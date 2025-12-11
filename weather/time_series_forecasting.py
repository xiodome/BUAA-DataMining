#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weather 时间序列预测任务完整实现脚本
===================================

该脚本针对课程《数据挖掘导论2025》大作业任务 3（时间序列预测任务），
围绕以下要求给出可直接运行的基准方案：

3.0 问题的形式化描述
    - 输入：Weather 气象站连续 10min 采样的 21 个气象指标。
    - 目标：利用过去 2 小时（默认 12 个时间点）的观测，预测下一时间点的室外温度 OT。
    - 输出：对 OT 的回归预测值以及模型评估指标。

3.1 数据预处理 + 训练/测试划分
    - 清洗列名、解析时间戳、缺失值插补。
    - 使用滑动窗口构造样本，按时间顺序划分训练 / 测试集。
    - 特征标准化，避免量纲差异影响模型。

3.2 构建时间序列预测模型（设计思路）
    - 采用 Gradient Boosting 回归器（树模型，能捕捉非线性关系，鲁棒性好）。
    - 滑动窗口展开后作为模型输入，预测下一时刻 OT。

3.3 模型效果评估
    - 输出训练/测试集上的 MAE、RMSE、R²。
    - 可选保存预测结果、误差分析数据，辅助撰写报告。

运行示例（Windows PowerShell）::

    python time_series_forecasting.py `
        --data-path "G:/大学/大三上/数据挖掘/时间序列预测任务/weather.csv" `
        --history 12 `
        --forecast-horizon 1 `
        --train-ratio 0.8 `
        --pred-output weather_predictions.csv `
        --metrics-output metrics.json

依赖：
    pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------
@dataclass
class DatasetDescription:
    dataframe: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    time_column: str


@dataclass
class WindowedDataset:
    X: np.ndarray  # [n_samples, history, n_features]
    y: np.ndarray  # [n_samples, ]
    timestamps: pd.Series


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weather 时间序列预测基准方案（Gradient Boosting 回归）"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="weather.csv 数据文件路径。",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="latin-1",
        help="CSV 编码，Weather 数据默认 latin-1。",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=12,
        help="滑动窗口历史长度（过去多少个时间点），默认 12（两小时）。",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="预测步长：1 表示预测下一时间点。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（时间顺序划分）。",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="date",
        help="时间戳列名称（文件中为 date）。",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="ot",
        help="预测目标列名称（清洗后统一小写，默认 ot）。",
    )
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=None,
        help="可选：保存测试集预测结果 CSV 的路径。",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="可选：保存评估指标 JSON 的路径。",
    )
    parser.add_argument(
        "--save-window-data",
        type=Path,
        default=None,
        help="可选：保存滑动窗口后的 numpy 数据 (.npz)。",
    )
    return parser


# ---------------------------------------------------------------------------
# 数据预处理（3.1）
# ---------------------------------------------------------------------------
def sanitize_column(name: str) -> str:
    """将原始列名转换为 ASCII，方便后续处理。"""
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


def load_and_prepare_dataset(
    csv_path: Path,
    encoding: str,
    time_column: str,
    target_column: str,
) -> DatasetDescription:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")

    df_raw = pd.read_csv(csv_path, encoding=encoding)
    rename_map = {col: sanitize_column(col) for col in df_raw.columns}
    df = df_raw.rename(columns=rename_map)

    if time_column not in df.columns:
        raise KeyError(
            f"未找到时间戳列 {time_column}。可用列：{list(df.columns)[:10]}..."
        )
    if target_column not in df.columns:
        raise KeyError(
            f"未找到目标列 {target_column}。可用列：{list(df.columns)[:10]}..."
        )

    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)

    numeric_cols = []
    for col in df.columns:
        if col == time_column:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_cols.append(col)

    # 缺失值插补：双向填充 + 线性插值
    df = df.set_index(time_column)
    df.interpolate(method="time", inplace=True, limit_direction="both")
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    if df[target_column].isna().any():
        raise ValueError("目标列仍存在缺失值，请检查数据质量。")

    feature_cols = [col for col in numeric_cols if col != target_column]
    return DatasetDescription(
        dataframe=df.reset_index(),
        feature_columns=feature_cols,
        target_column=target_column,
        time_column=time_column,
    )


def build_sliding_windows(
    data: DatasetDescription,
    history: int,
    forecast_horizon: int,
) -> WindowedDataset:
    df = data.dataframe
    features = df[data.feature_columns].to_numpy(dtype=np.float32)
    targets = df[data.target_column].to_numpy(dtype=np.float32)
    timestamps = df[data.time_column]

    if history <= 0:
        raise ValueError("history 必须为正整数。")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon 必须为正整数。")

    max_start = len(df) - forecast_horizon
    if max_start <= history:
        raise ValueError("数据量不足以生成所需滑动窗口。")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[pd.Timestamp] = []

    for idx in range(history, max_start + 1):
        window = features[idx - history : idx]
        target_idx = idx + forecast_horizon - 1
        X_list.append(window)
        y_list.append(targets[target_idx])
        ts_list.append(timestamps.iloc[target_idx])

    X = np.stack(X_list)
    y = np.asarray(y_list)
    ts = pd.Series(ts_list, name="timestamp")
    return WindowedDataset(X=X, y=y, timestamps=ts)


def chronological_split(
    windowed: WindowedDataset, train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须位于 (0, 1) 之间。")

    n_samples = len(windowed.X)
    split_idx = max(1, min(int(n_samples * train_ratio), n_samples - 1))

    X_train, X_test = windowed.X[:split_idx], windowed.X[split_idx:]
    y_train, y_test = windowed.y[:split_idx], windowed.y[split_idx:]
    ts_test = windowed.timestamps.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, ts_test


def reshape_and_scale(
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
def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray
) -> GradientBoostingRegressor:
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


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None)))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE(%)": mape}


# ---------------------------------------------------------------------------
# 报告&导出
# ---------------------------------------------------------------------------
def export_predictions(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    output_df = pd.DataFrame(
        {"timestamp": timestamps, "y_true": y_true, "y_pred": y_pred, "error": y_pred - y_true}
    )
    output_df.to_csv(output_path, index=False)
    print(f"[INFO] 预测结果已保存：{output_path.resolve()}")


def export_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 指标已保存：{output_path.resolve()}")


def export_npz(windowed: WindowedDataset, path: Path) -> None:
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

    print("[STEP] 读取与预处理数据（3.1）")
    dataset = load_and_prepare_dataset(
        csv_path=args.data_path,
        encoding=args.encoding,
        time_column=args.time_column,
        target_column=args.target_column,
    )
    print(
        f"[INFO] 数据行数 {len(dataset.dataframe)}，特征列 {len(dataset.feature_columns)}，目标列 {dataset.target_column}"
    )

    windowed = build_sliding_windows(
        dataset,
        history=args.history,
        forecast_horizon=args.forecast_horizon,
    )
    print(
        f"[INFO] 滑动窗口生成完成：样本数 {len(windowed.X)}，窗口形状 {windowed.X.shape[1:]}。"
    )

    if args.save_window_data:
        export_npz(windowed, args.save_window_data)

    X_train, X_test, y_train, y_test, ts_test = chronological_split(
        windowed, args.train_ratio
    )
    print(
        f"[INFO] 训练集 {len(X_train)} 条，测试集 {len(X_test)} 条（保持时间顺序）。"
    )

    X_train_scaled, X_test_scaled, _ = reshape_and_scale(X_train, X_test)
    print("[INFO] 特征标准化完成。")

    print("[STEP] 训练 Gradient Boosting 回归模型（3.2）")
    model = train_gradient_boosting(X_train_scaled, y_train)
    print("[INFO] 模型训练完成。")

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_metrics = evaluate_regression(y_train, train_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    print("\n=== 训练集指标（3.3） ===")
    for k, v in train_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    print("\n=== 测试集指标（3.3） ===")
    for k, v in test_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    if args.pred_output:
        export_predictions(ts_test, y_test, test_pred, args.pred_output)

    if args.metrics_output:
        export_metrics({"train": train_metrics, "test": test_metrics}, args.metrics_output)


if __name__ == "__main__":
    main()

