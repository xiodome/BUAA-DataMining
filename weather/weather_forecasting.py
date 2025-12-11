#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weather 时间序列预测任务完整实现
================================

面向《数据挖掘导论2025》大作业任务 3，针对 Weather 数据集，实现以下步骤：

3.0 问题形式化：以过去 2 小时（12 个时间点）各气象指标为输入，预测下一时间点室外温度（OT）。
3.1 数据预处理：列名清洗、时间排序、缺失值插补、滑动窗口切分、训练/测试集划分、特征标准化。
3.2 建模思路：使用 Gradient Boosting Regressor 进行回归预测，理由是对特征尺度不敏感、可捕捉非线性、
    对较小数据集表现稳定，并能解释特征重要度。
3.3 评估效果：输出训练/测试集 MAE、RMSE、R²、MAPE，可选导出预测结果与指标 JSON。

运行示例（PowerShell）::

    python weather_forecasting.py `
        --data-path "G:/大学/大三上/数据挖掘/时间序列预测任务/weather.csv" `
        --history 12 `
        --forecast-horizon 1 `
        --train-ratio 0.8 `
        --pred-output forecasts.csv `
        --metrics-output metrics.json

依赖::

    pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    dataframe: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    time_column: str


@dataclass
class WindowedSamples:
    X: np.ndarray  # [n_samples, history, n_features]
    y: np.ndarray  # [n_samples, ]
    timestamps: pd.Series


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weather 时间序列预测基准脚本（Gradient Boosting 回归）"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="weather.csv 文件路径。",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="latin-1",
        help="CSV 编码，Weather 数据通常为 latin-1。",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=12,
        help="滑动窗口历史长度（时间步数）。默认 12 (2 小时)。",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="预测步长，默认预测下一时间点。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="按时间顺序划分训练集比例。",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="date",
        help="时间戳列名称（清洗后默认为 date）。",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="ot",
        help="预测目标列（清洗后默认 ot）。",
    )
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=None,
        help="可选：保存测试集预测结果的 CSV 路径。",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="可选：保存评估指标 JSON 路径。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gbrt_default",
        choices=["gbrt_default", "gbrt_light", "random_forest", "hist_gbrt", "ridge"],
        help="选择回归模型：gbrt_default(原始超参)、gbrt_light(浅树)、random_forest、hist_gbrt、ridge。",
    )
    parser.add_argument(
        "--save-window-data",
        type=Path,
        default=None,
        help="可选：保存滑动窗口样本为 .npz 文件。",
    )
    return parser


# ---------------------------------------------------------------------------
# 数据预处理（3.1）
# ---------------------------------------------------------------------------
def sanitize_column(name: str) -> str:
    """将列名转换为可读的 ASCII 形式。"""
    ascii_name = (
        name.replace("μ", "u")
        .replace("²", "2")
        .replace("°", "")
        .replace("·", "")
        .replace("¯", "")
        .replace("ð", "d")
    )
    ascii_name = re.sub(r"[^0-9A-Za-z]+", "_", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    return ascii_name.lower()


def load_dataset(
    csv_path: Path,
    encoding: str,
    time_column: str,
    target_column: str,
) -> DatasetInfo:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")

    df_raw = pd.read_csv(csv_path, encoding=encoding)
    df = df_raw.rename(columns={c: sanitize_column(c) for c in df_raw.columns})

    if time_column not in df.columns:
        raise KeyError(f"未找到时间列 {time_column}，请检查列名。")
    if target_column not in df.columns:
        raise KeyError(f"未找到目标列 {target_column}，请检查列名。")

    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)

    numeric_cols: List[str] = []
    for col in df.columns:
        if col == time_column:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_cols.append(col)

    df = df.set_index(time_column)
    df.interpolate(method="time", inplace=True, limit_direction="both")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    if df[target_column].isna().any():
        raise ValueError("目标列仍存在缺失值，无法训练模型。")

    feature_cols = [col for col in numeric_cols if col != target_column]
    return DatasetInfo(
        dataframe=df.reset_index(),
        feature_columns=feature_cols,
        target_column=target_column,
        time_column=time_column,
    )


def build_windows(
    info: DatasetInfo,
    history: int,
    forecast_horizon: int,
) -> WindowedSamples:
    df = info.dataframe
    features = df[info.feature_columns].to_numpy(dtype=np.float32)
    targets = df[info.target_column].to_numpy(dtype=np.float32)
    timestamps = df[info.time_column]

    if history <= 0 or forecast_horizon <= 0:
        raise ValueError("history 与 forecast_horizon 均须为正整数。")

    max_start = len(df) - forecast_horizon
    if max_start <= history:
        raise ValueError("数据量不足以生成所需滑动窗口，请调整参数。")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[pd.Timestamp] = []

    for idx in range(history, max_start + 1):
        X_list.append(features[idx - history : idx])
        target_idx = idx + forecast_horizon - 1
        y_list.append(targets[target_idx])
        ts_list.append(timestamps.iloc[target_idx])

    X = np.stack(X_list)
    y = np.asarray(y_list)
    ts = pd.Series(ts_list, name="timestamp")
    return WindowedSamples(X=X, y=y, timestamps=ts)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    ts: pd.Series,
    train_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须位于 (0, 1) 之间。")

    n_samples = len(X)
    split_idx = max(1, min(int(n_samples * train_ratio), n_samples - 1))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, ts_test


def standardize(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    n_train, history, n_feat = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(n_train, history * n_feat)
    X_test_2d = X_test.reshape(n_test, history * n_feat)

    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 模型训练与评估（3.2/3.3）
# ---------------------------------------------------------------------------
def train_model(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray
) -> object:
    """根据名称选择模型并训练。"""
    if model_name == "gbrt_light":
        model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.85,
            random_state=42,
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "hist_gbrt":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_depth=8,
            l2_regularization=0.0,
            random_state=42,
        )
    elif model_name == "ridge":
        model = Ridge(alpha=1.0)
    else:  # gbrt_default
        model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)
    return model


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None)))
    ) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE(%)": mape}


# ---------------------------------------------------------------------------
# 导出工具
# ---------------------------------------------------------------------------
def save_predictions(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    df = pd.DataFrame(
        {"timestamp": timestamps, "y_true": y_true, "y_pred": y_pred, "error": y_pred - y_true}
    )
    df.to_csv(output_path, index=False)
    print(f"[INFO] 预测结果已保存：{output_path.resolve()}")


def save_metrics(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 指标已保存：{output_path.resolve()}")


def save_window_npz(windowed: WindowedSamples, path: Path) -> None:
    np.savez_compressed(
        path,
        X=windowed.X,
        y=windowed.y,
        timestamps=windowed.timestamps.astype(str).to_numpy(),
    )
    print(f"[INFO] 滑动窗口数据已保存：{path.resolve()}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    print("[STEP] 数据加载与预处理（3.1）")
    dataset = load_dataset(
        csv_path=args.data_path,
        encoding=args.encoding,
        time_column=args.time_column,
        target_column=args.target_column,
    )
    print(
        f"[INFO] 样本数 {len(dataset.dataframe)}，特征列 {len(dataset.feature_columns)}，目标列 {dataset.target_column}"
    )

    windowed = build_windows(
        dataset,
        history=args.history,
        forecast_horizon=args.forecast_horizon,
    )
    print(
        f"[INFO] 滑动窗口生成完毕：样本数 {len(windowed.X)}，窗口维度 {windowed.X.shape[1:]}。"
    )

    if args.save_window_data:
        save_window_npz(windowed, args.save_window_data)

    X_train, X_test, y_train, y_test, ts_test = chronological_split(
        windowed.X, windowed.y, windowed.timestamps, args.train_ratio
    )
    print(f"[INFO] 训练集 {len(X_train)} 条，测试集 {len(X_test)} 条（按时间切分）。")

    X_train_scaled, X_test_scaled, _ = standardize(X_train, X_test)
    print("[INFO] 特征标准化完成。")

    print(f"[STEP] 训练模型（3.2）：{args.model}")
    model = train_model(args.model, X_train_scaled, y_train)
    print("[INFO] 模型训练完成。")

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_metrics = evaluate_metrics(y_train, train_pred)
    test_metrics = evaluate_metrics(y_test, test_pred)

    print("\n=== 训练集指标（3.3） ===")
    for k, v in train_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    print("\n=== 测试集指标（3.3） ===")
    for k, v in test_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    if args.pred_output:
        save_predictions(ts_test, y_test, test_pred, args.pred_output)

    if args.metrics_output:
        save_metrics({"train": train_metrics, "test": test_metrics}, args.metrics_output)


if __name__ == "__main__":
    main()