"""
cluster_solution.py
====================

该脚本针对 `Cluster/dataset` 图像集完成以下任务：

1. **问题形式化**：将图像聚类视为无监督学习任务，目标是依据外观特征将 600 张图片划分为 6 个簇。
2. **图像特征处理**：对每张图像提取多模态特征（颜色直方图、纹理 LBP、梯度 HOG），组成数值特征向量。
3. **聚类算法选择**：提供 KMeans 与 层次聚类（Ward）两种算法，并在主流程中默认采用 KMeans。
4. **聚类效果评估**：若提供 `cluster_labels.json` 则与真实标签比较，输出 Silhouette、Davies-Bouldin、ARI、NMI 等指标，并给出 t-SNE 可视化。

使用方式（示例）::

    python cluster_solution.py ^
        --dataset Cluster/dataset ^
        --labels Cluster/cluster_labels.json ^
        --output-dir Cluster/outputs

依赖库：
    numpy, pillow, scikit-image, scikit-learn, matplotlib
可通过 `pip install -r requirements.txt` 安装（若无 requirements.txt，可运行脚本顶端的 `REQUIRED_PACKAGES` 提示）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from skimage import color, exposure
from skimage.feature import hog, local_binary_pattern

import importlib.util


REQUIRED_PACKAGES = [
    "numpy",
    "pillow",
    "scikit-image",
    "scikit-learn",
    "matplotlib",
]


def _is_package_installed(package_name: str) -> bool:
    normalized = package_name.replace("-", "_")
    return importlib.util.find_spec(normalized) is not None


def _resolve_dir(path_str: str) -> Path:
    """将相对路径解析为绝对路径，兼容以脚本目录为起点的写法。"""
    path = Path(path_str).expanduser()
    if path.is_dir():
        return path.resolve()
    alt = Path(__file__).resolve().parent / path_str
    if alt.is_dir():
        return alt.resolve()
    raise FileNotFoundError(f"数据集目录不存在: {path_str}")


def _resolve_labels_path(dataset_dir: Path, labels_arg: str) -> Optional[Path]:
    """优先使用命令行提供的标签路径；否则尝试 dataset 同级目录下的 cluster_labels.json。"""
    if labels_arg:
        candidate = Path(labels_arg).expanduser()
        if candidate.exists():
            return candidate.resolve()
        alt = Path(__file__).resolve().parent / labels_arg
        if alt.exists():
            return alt.resolve()
        raise FileNotFoundError(f"标签文件不存在: {labels_arg}")

    auto = dataset_dir.parent / "cluster_labels.json"
    return auto.resolve() if auto.exists() else None


@dataclass(frozen=True)
class ImageRecord:
    """图像路径与可选标签。"""

    path: Path
    label: Optional[str] = None


def list_image_files(dataset_dir: Path, extensions: Sequence[str] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    files = [p for p in sorted(dataset_dir.iterdir()) if p.suffix.lower() in extensions]
    if not files:
        raise FileNotFoundError(f"在 {dataset_dir} 未找到图像文件，支持后缀: {extensions}")
    return files


def load_labels(labels_path: Path) -> Dict[str, str]:
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


def prepare_records(dataset_dir: Path, labels_path: Optional[Path]) -> List[ImageRecord]:
    image_paths = list_image_files(dataset_dir)
    labels: Dict[str, str] = {}
    if labels_path and labels_path.exists():
        labels = load_labels(labels_path)
    records = [
        ImageRecord(path=img_path, label=labels.get(img_path.name))
        for img_path in image_paths
    ]
    return records


def extract_color_histograms(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """提取 RGB 与 HSV 直方图并拼接。"""
    histograms: List[np.ndarray] = []
    # RGB 通道
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 255), density=True)
        histograms.append(hist)
    # HSV 空间
    hsv = color.rgb2hsv(image)
    ranges = [(0, 1), (0, 1), (0, 1)]
    for channel in range(3):
        hist, _ = np.histogram(hsv[:, :, channel], bins=bins, range=ranges[channel], density=True)
        histograms.append(hist)
    return np.concatenate(histograms)


def extract_texture_features(image_gray: np.ndarray) -> np.ndarray:
    """提取 LBP + HOG 纹理特征。"""
    lbp = local_binary_pattern(image_gray, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), density=True)

    hog_vec = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.concatenate([lbp_hist, hog_vec])


def extract_features(record: ImageRecord, image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """对单张图像提取特征。"""
    image = Image.open(record.path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    image_np = np.asarray(image, dtype=np.uint8)
    gray = color.rgb2gray(image_np)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    color_feat = extract_color_histograms(image_np)
    texture_feat = extract_texture_features(gray)
    return np.concatenate([color_feat, texture_feat])


def build_feature_matrix(records: Sequence[ImageRecord]) -> Tuple[np.ndarray, List[str]]:
    features = [extract_features(rec) for rec in records]
    feature_matrix = np.vstack(features)
    filenames = [rec.path.name for rec in records]
    return feature_matrix, filenames


def encode_labels(records: Sequence[ImageRecord]) -> Tuple[Optional[np.ndarray], List[str]]:
    labels = [rec.label for rec in records]
    if any(label is None for label in labels):
        return None, []
    unique = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(unique)}
    encoded = np.array([mapping[label] for label in labels], dtype=np.int64)
    return encoded, unique


def run_kmeans(features: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=0.95, random_state=random_state)
    features_reduced = pca.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=500, random_state=random_state)
    pred_labels = kmeans.fit_predict(features_reduced)

    return pred_labels, {
        "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
        "n_components": int(pca.n_components_),
    }, features_reduced


def run_agglomerative(features_reduced: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(features_reduced)


def evaluate_clustering(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    unique_clusters = np.unique(pred_labels)
    metrics: Dict[str, float] = {}
    if len(unique_clusters) < 2:
        return metrics
    metrics["silhouette"] = float(silhouette_score(embeddings, pred_labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings, pred_labels))
    if true_labels is not None:
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(true_labels, pred_labels))
        metrics["normalized_mutual_info"] = float(
            normalized_mutual_info_score(true_labels, pred_labels)
        )
        metrics["clustering_accuracy"] = float(compute_clustering_accuracy(true_labels, pred_labels))
    return metrics


def run_tsne(
    embeddings: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> np.ndarray:
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE 需要至少 3 个样本。")
    safe_perplexity = min(perplexity, max(2.0, n_samples - 1))
    if safe_perplexity != perplexity:
        print(f"提示：自动将 t-SNE perplexity 从 {perplexity} 调整为 {safe_perplexity} 以适配样本量。")

    tsne_common_kwargs = dict(
        n_components=2,
        random_state=random_state,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
    )
    # scikit-learn 1.8.0 将 n_iter 更名为 max_iter，这里兼容两种写法
    try:
        tsne = TSNE(**tsne_common_kwargs, n_iter=n_iter)
    except TypeError:
        tsne = TSNE(**tsne_common_kwargs, max_iter=n_iter)
    return tsne.fit_transform(embeddings)


def plot_clusters(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    filenames: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="tab10", s=35, alpha=0.85)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    for idx in range(len(filenames)):
        if idx % max(1, len(filenames) // 200) == 0:  # 避免过度标注
            plt.annotate(filenames[idx], (embedding_2d[idx, 0], embedding_2d[idx, 1]), fontsize=6, alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_clustering_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """使用匈牙利算法求解最佳簇-类映射，返回聚类精度。"""
    if true_labels.size == 0 or pred_labels.size == 0:
        return float("nan")
    cont = contingency_matrix(true_labels, pred_labels)
    if cont.size == 0:
        return float("nan")
    row_ind, col_ind = linear_sum_assignment(-cont)
    matched = cont[row_ind, col_ind].sum()
    total = cont.sum()
    return float(matched / total) if total > 0 else float("nan")


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    header = ["Algorithm"] + sorted({metric for metrics in metrics_dict.values() for metric in metrics})
    col_widths = [max(len(h), 12) for h in header]
    divider = "+".join("-" * (width + 2) for width in col_widths)
    print(divider)
    print("| " + " | ".join(h.ljust(width) for h, width in zip(header, col_widths)) + " |")
    print(divider)
    for algo, metrics in metrics_dict.items():
        row = [algo] + [f"{metrics.get(metric, np.nan):.4f}" for metric in header[1:]]
        print("| " + " | ".join(val.ljust(width) for val, width in zip(row, col_widths)) + " |")
    print(divider)


def dump_predictions(
    output_dir: Path,
    filenames: Sequence[str],
    labels_pred: Dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for algo_name, preds in labels_pred.items():
        output_file = output_dir / f"{algo_name}_predictions.csv"
        with output_file.open("w", encoding="utf-8") as f:
            f.write("filename,predicted_cluster\n")
            for name, label in zip(filenames, preds):
                f.write(f"{name},{label}\n")


def main(args: argparse.Namespace) -> None:
    dataset_dir = _resolve_dir(args.dataset)
    labels_path = _resolve_labels_path(dataset_dir, args.labels or "")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("Cluster/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 准备数据 ===")
    if labels_path:
        print(f"使用标签文件: {labels_path}")
    else:
        print("未找到标签文件，跳过有监督指标。")

    records = prepare_records(dataset_dir, labels_path)
    features, filenames = build_feature_matrix(records)
    true_labels, label_names = encode_labels(records)

    print(f"样本数: {len(records)}, 特征维度: {features.shape[1]}")
    if label_names:
        print(f"真实类别: {label_names}")
    else:
        print("未提供真实标签，部分指标将不可用。")

    print("\n=== KMeans 聚类 ===")
    kmeans_labels, pca_info, embeddings = run_kmeans(features, n_clusters=args.n_clusters, random_state=args.random_state)
    print(f"PCA 主成分数量: {pca_info['n_components']}, 累计解释方差: {pca_info['explained_variance_ratio']:.4f}")

    print("\n=== 层次聚类 (Ward) ===")
    aggl_labels = run_agglomerative(embeddings, n_clusters=args.n_clusters)

    metrics_results = {
        "kmeans": evaluate_clustering(embeddings, kmeans_labels, true_labels),
        "agglomerative": evaluate_clustering(embeddings, aggl_labels, true_labels),
    }
    print("\n=== 指标汇总 ===")
    print_metrics_table(metrics_results)

    print("\n=== 生成可视化 ===")
    embedding_2d = run_tsne(embeddings, random_state=args.random_state)
    plot_clusters(
        embedding_2d,
        kmeans_labels,
        filenames,
        output_dir / "kmeans_tsne.png",
        title="KMeans 聚类结果 (t-SNE)",
    )
    plot_clusters(
        embedding_2d,
        aggl_labels,
        filenames,
        output_dir / "agglomerative_tsne.png",
        title="Agglomerative 聚类结果 (t-SNE)",
    )
    print(f"可视化与预测已保存到: {output_dir.resolve()}")

    dump_predictions(output_dir, filenames, {"kmeans": kmeans_labels, "agglomerative": aggl_labels})

    if args.save_metrics:
        metrics_path = output_dir / "metrics_summary.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_results, f, indent=4, ensure_ascii=False)
        print(f"指标已写入: {metrics_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster 数据集聚类脚本")
    parser.add_argument("--dataset", type=str, required=True, help="图像数据集目录（包含 001.png 等文件）")
    parser.add_argument("--labels", type=str, default="", help="可选：真实标签 JSON 路径")
    parser.add_argument("--output-dir", type=str, default="", help="输出结果目录，默认为 Cluster/outputs")
    parser.add_argument("--n-clusters", type=int, default=6, help="聚类类别数，默认 6")
    parser.add_argument("--random-state", type=int, default=42, help="随机数种子")
    parser.add_argument("--save-metrics", action="store_true", help="是否保存指标 JSON 文件")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not os.environ.get("SKIP_PACKAGE_HINT", ""):
        missing = [pkg for pkg in REQUIRED_PACKAGES if not _is_package_installed(pkg)]
        if missing:
            print(
                "提示：检测到以下依赖可能未安装 -> "
                + ", ".join(missing)
                + "\n可执行: pip install " + " ".join(missing)
            )
    main(args)