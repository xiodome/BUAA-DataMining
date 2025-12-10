"""
图像异常检测任务
基于 PatchCore 的异常检测流程，附加多指标评估与特征可视化
"""

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


class ImageAnomalyDataset(Dataset):
    """按类别加载图像并返回图像/标签/路径。"""

    def __init__(self, root_dir: Union[str, Path], category: str, split: str,
                 transform: Optional[transforms.Compose] = None) -> None:
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.images: List[str] = []
        self.labels: List[int] = []
        self._extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP")
        self._load_split()
        if not self.images:
            raise ValueError(f"未找到任何图像，请检查路径: {self.root_dir / category / split}")

    def _collect(self, directory: Path, label: int) -> None:
        if not directory.exists():
            return
        for pattern in self._extensions:
            for path in sorted(directory.glob(pattern)):
                self.images.append(str(path))
                self.labels.append(label)

    def _load_split(self) -> None:
        print(f"加载数据: {self.category}/{self.split}")
        if self.split == "train":
            good_dir = self.root_dir / self.category / "train" / "good"
            bad_dir = self.root_dir / self.category / "train" / "bad"
            self._collect(good_dir, 0)
            self._collect(bad_dir, 1)
        else:
            test_dir = self.root_dir / self.category / "test"
            good_dir = test_dir / "good"
            bad_dir = test_dir / "bad"
            if good_dir.exists() or bad_dir.exists():
                self._collect(good_dir, 0)
                self._collect(bad_dir, 1)
            else:
                for pattern in self._extensions:
                    for path in sorted(test_dir.glob(pattern)):
                        self.images.append(str(path))
                        self.labels.append(-1)
        total = len(self.images)
        normal = sum(1 for label in self.labels if label == 0)
        anomaly = sum(1 for label in self.labels if label == 1)
        print(f"  样本总数: {total} (normal={normal}, anomaly={anomaly})")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"警告: 无法加载图像 {image_path}: {exc}")
            image = Image.new("RGB", (224, 224), color="white")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label, image_path


class ResNetEmbedder(nn.Module):
    """截断的 Wide-ResNet50-2 用作 PatchCore 特征提取 backbone。"""

    def __init__(self, use_pretrained: bool = True) -> None:
        super().__init__()
        if use_pretrained:
            try:
                backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            except Exception as exc:
                print(f"⚠ 预训练权重加载失败，使用随机初始化: {exc}")
                backbone = models.wide_resnet50_2(weights=None)
        else:
            backbone = models.wide_resnet50_2(weights=None)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


class PatchCore:
    """PatchCore 实现，负责构建记忆库并输出异常分数。"""

    def __init__(self, backbone: nn.Module, device: torch.device,
                 max_patches: int = 120_000) -> None:
        self.backbone = backbone.to(device)
        self.device = device
        self.max_patches = max_patches
        self.index: Optional[faiss.IndexFlatL2] = None
        self.train_vectors: Optional[np.ndarray] = None
        self.train_paths: List[str] = []

    @staticmethod
    def _embedding_concat(maps: List[torch.Tensor]) -> torch.Tensor:
        resized = [F.interpolate(fmap, size=(28, 28), mode="bilinear", align_corners=False) for fmap in maps]
        return torch.cat(resized, dim=1)

    def build_memory_bank(self, loader: DataLoader) -> None:
        self.backbone.eval()
        embedding_list: List[np.ndarray] = []
        vector_list: List[np.ndarray] = []
        path_list: List[str] = []

        for images, _, paths in tqdm(loader, desc="构建记忆库", leave=False):
            images = images.to(self.device)
            with torch.no_grad():
                feats = self.backbone(images)
                emb = self._embedding_concat(feats)
            emb_np = emb.detach().cpu().numpy()
            emb_np = emb_np.reshape(emb_np.shape[0], emb_np.shape[1], -1)
            emb_np = np.transpose(emb_np, (0, 2, 1))
            embedding_list.append(emb_np)
            vector_list.append(emb_np.mean(axis=1))
            path_list.extend(list(paths))

        if not embedding_list:
            raise ValueError("记忆库构建失败，未提取到任何特征。")

        patches = np.concatenate(embedding_list, axis=0)
        patches = patches.reshape(-1, patches.shape[-1]).astype(np.float32)

        if len(patches) > self.max_patches:
            choice = np.random.choice(len(patches), self.max_patches, replace=False)
            patches = patches[choice]

        self.index = faiss.IndexFlatL2(patches.shape[-1])
        self.index.add(patches)
        self.train_vectors = np.vstack(vector_list)
        self.train_paths = path_list
        print(f"记忆库构建完成，共包含 {self.index.ntotal} 个 patch 样本")

    def predict_batch(self, images: torch.Tensor) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        if self.index is None:
            raise RuntimeError("记忆库尚未构建，请先调用 build_memory_bank。")
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(images.to(self.device))
            emb = self._embedding_concat(feats)
        emb_np = emb.detach().cpu().numpy()
        emb_np = emb_np.reshape(emb_np.shape[0], emb_np.shape[1], -1)
        emb_np = np.transpose(emb_np, (0, 2, 1))

        scores: List[float] = []
        heatmaps: List[np.ndarray] = []
        vectors = emb_np.mean(axis=1)

        for patches in emb_np:
            distances, _ = self.index.search(patches.astype(np.float32), k=1)
            distances = distances.reshape(28, 28)
            score = float(distances.max())
            norm_map = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            scores.append(score)
            heatmaps.append(norm_map)

        return scores, heatmaps, vectors


class Evaluator:
    """评估并可视化异常检测结果。"""

    def __init__(self) -> None:
        self.history: Dict[str, Dict[str, object]] = {}

    def evaluate(self, category: str, y_true: np.ndarray, scores: np.ndarray,
                 store_history: bool = True) -> Dict[str, float]:
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, scores)

        youden_index = tpr - fpr
        best_idx = int(np.argmax(youden_index))
        best_threshold = float(roc_thresholds[best_idx])
        y_pred = (scores >= best_threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        report = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], digits=4)

        print(f"{category} 评估指标:")
        metrics_df = pd.DataFrame({
            "ROC-AUC": [roc_auc],
            "PR-AUC": [pr_auc],
            "Precision": [precision],
            "Recall": [recall],
            "Specificity": [specificity],
            "F1": [f1],
            "Accuracy": [accuracy],
            "Balanced Accuracy": [balanced_acc],
            "MCC": [mcc],
            "Best Threshold": [best_threshold],
        })
        print(metrics_df.to_string(index=False))
        print("分类报告:")
        print(report)
        print("混淆矩阵:")
        print(cm)

        if store_history:
            self.history[category] = {
                "scores": scores,
                "y_true": y_true,
                "y_pred": y_pred,
                "fpr": fpr,
                "tpr": tpr,
                "precision_curve": precision_curve,
                "recall_curve": recall_curve,
                "roc_thresholds": roc_thresholds,
                "pr_thresholds": pr_thresholds,
                "best_threshold": best_threshold,
                "metrics": {
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "f1": f1,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "mcc": mcc,
                },
            }

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "mcc": mcc,
            "best_threshold": best_threshold,
        }

    def plot_curves(self, output_dir: Union[str, Path] = "results") -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for category, data in self.history.items():
            scores = np.asarray(data["scores"])
            y_true = np.asarray(data["y_true"])
            y_pred = np.asarray(data["y_pred"])
            threshold = float(data["best_threshold"])

            plt.figure(figsize=(6, 5))
            plt.plot(data["fpr"], data["tpr"], label=f"AUC = {data['metrics']['roc_auc']:.4f}", linewidth=2)
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {category}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output / f"{category}_roc.png", dpi=300)
            plt.close()

            plt.figure(figsize=(6, 5))
            plt.plot(data["recall_curve"], data["precision_curve"], label=f"AP = {data['metrics']['pr_auc']:.4f}", linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {category}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output / f"{category}_pr.png", dpi=300)
            plt.close()

            plt.figure(figsize=(6, 5))
            sns.histplot(scores[y_true == 0], bins=40, color="steelblue", alpha=0.6, label="Normal")
            sns.histplot(scores[y_true == 1], bins=40, color="indianred", alpha=0.6, label="Anomaly")
            plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
            plt.xlabel("Anomaly Score")
            plt.ylabel("Frequency")
            plt.title(f"Score Distribution - {category}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / f"{category}_score_distribution.png", dpi=300)
            plt.close()

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {category}")
            plt.tight_layout()
            plt.savefig(output / f"{category}_confusion.png", dpi=300)
            plt.close()


def plot_feature_space(category: str, train_vectors: Optional[np.ndarray], test_vectors: Optional[np.ndarray],
                       test_labels: np.ndarray, results_dir: Path, max_points: int = 2000) -> None:
    if train_vectors is None or test_vectors is None:
        return
    if test_vectors.size == 0:
        return

    train_labels = np.zeros(len(train_vectors), dtype=np.int64)
    features = np.vstack([train_vectors, test_vectors])
    labels = np.concatenate([train_labels, test_labels])
    splits = np.concatenate([
        np.full(len(train_vectors), "Train"),
        np.full(len(test_vectors), "Test"),
    ])

    if features.shape[0] > max_points:
        indices = np.random.choice(features.shape[0], max_points, replace=False)
        features = features[indices]
        labels = labels[indices]
        splits = splits[indices]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    coords = PCA(n_components=2).fit_transform(features_scaled)

    label_map = {0: "Normal", 1: "Anomaly"}
    label_names = np.array([label_map.get(int(label), "Unknown") for label in labels])
    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Label": label_names,
        "Split": splits,
    })

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Label", style="Split",
                    palette={"Normal": "#3182bd", "Anomaly": "#e6550d", "Unknown": "#969696"},
                    s=45, alpha=0.8, edgecolor="none")
    plt.title(f"Feature Space PCA - {category}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(results_dir / f"{category}_feature_pca.png", dpi=300)
    plt.close()


def save_top_anomaly_visualizations(tag: str, paths: List[str], heatmaps: List[np.ndarray],
                                    scores: np.ndarray, labels: Optional[np.ndarray],
                                    results_dir: Path, top_k: int = 6,
                                    title: Optional[str] = None) -> None:
    if not paths or not heatmaps:
        return
    k = min(top_k, len(paths))
    top_indices = np.argsort(scores)[-k:][::-1]

    ncols = 3
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2))
    axes = axes.flatten()

    for ax in axes[k:]:
        ax.axis("off")

    for rank, (idx, ax) in enumerate(zip(top_indices, axes), start=1):
        image = Image.open(paths[idx]).convert("RGB")
        heatmap = heatmaps[idx]
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
        ax.imshow(image)
        ax.imshow(np.array(heatmap_img) / 255.0, cmap="jet", alpha=0.45)
        label_text = "Unknown"
        if labels is not None:
            if labels[idx] == 1:
                label_text = "Anomaly"
            elif labels[idx] == 0:
                label_text = "Normal"
        ax.set_title(f"#{rank} score={scores[idx]:.4f}\\nlabel={label_text}")
        ax.axis("off")

    display_title = title or tag
    fig.suptitle(f"Top-{k} Anomaly Heatmaps - {display_title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / f"{tag}_top_anomalies.png", dpi=300)
    plt.close()


def save_good_bad_examples(category: str, dataset: ImageAnomalyDataset, results_dir: Path,
                           num_examples: int = 3) -> None:
    good_paths = [path for path, label in zip(dataset.images, dataset.labels) if label == 0]
    bad_paths = [path for path, label in zip(dataset.images, dataset.labels) if label == 1]
    if not good_paths or not bad_paths:
        return

    n_good = min(num_examples, len(good_paths))
    n_bad = min(num_examples, len(bad_paths))
    n = min(n_good, n_bad)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(n * 4.0, 2 * 4.0))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]]) if isinstance(axes, np.ndarray) else np.array([[axes], [axes]])

    for idx in range(n):
        image = Image.open(good_paths[idx]).convert("RGB")
        axes[0, idx].imshow(image)
        axes[0, idx].set_title(f"Good #{idx + 1}")
        axes[0, idx].axis("off")

    for idx in range(n):
        image = Image.open(bad_paths[idx]).convert("RGB")
        axes[1, idx].imshow(image)
        axes[1, idx].set_title(f"Anomaly #{idx + 1}")
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("Train Good", fontsize=12)
    axes[1, 0].set_ylabel("Train Bad", fontsize=12)
    fig.suptitle(f"{category} Good vs. Bad Examples (Train)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / f"{category}_good_vs_bad_examples.png", dpi=300)
    plt.close()


def run_inference(loader: DataLoader, patchcore: PatchCore, desc: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray], Optional[np.ndarray]]:
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_paths: List[str] = []
    all_heatmaps: List[np.ndarray] = []
    all_vectors: List[np.ndarray] = []

    for images, labels, paths in tqdm(loader, desc=desc, leave=False):
        scores, heatmaps, vectors = patchcore.predict_batch(images)
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())
        all_paths.extend(list(paths))
        all_heatmaps.extend(heatmaps)
        all_vectors.append(vectors)

    scores_array = np.array(all_scores, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)
    vectors_array = np.vstack(all_vectors) if all_vectors else None
    return scores_array, labels_array, all_paths, all_heatmaps, vectors_array


def check_data_structure(data_root: Union[str, Path]) -> bool:
    print("=" * 60)
    print("检查数据结构...")
    print("=" * 60)
    root = Path(data_root)
    if not root.exists():
        print(f"❌ 数据根目录不存在: {root}")
        print(f"   当前工作目录: {Path.cwd()}")
        return False
    print(f"✓ 数据根目录存在: {root}")
    ok = True
    for category in ("hazelnut", "zipper"):
        category_dir = root / category
        print(f"类别: {category}")
        if not category_dir.exists():
            print(f"  ❌ 缺失目录 {category_dir}")
            ok = False
            continue
        for sub in ("train/good", "train/bad", "test"):
            path = category_dir / sub
            exists = path.exists()
            count = len(list(path.glob("**/*.*"))) if exists else 0
            prefix = "✓" if exists else "❌"
            print(f"  {prefix} {sub} -> {count} 个文件")
            if not exists and sub != "train/bad":
                ok = False
    return ok


def run_patchcore_for_category(category: str, data_root: Union[str, Path], transform: transforms.Compose,
                               device: torch.device, results_dir: Path, batch_size: int, num_workers: int,
                               evaluator: Evaluator) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    print("-" * 60)
    print(f"处理类别: {category}")
    print("-" * 60)

    train_dataset = ImageAnomalyDataset(data_root, category, split="train", transform=transform)
    save_good_bad_examples(category, train_dataset, results_dir)
    normal_indices = [idx for idx, label in enumerate(train_dataset.labels) if label == 0]
    if not normal_indices:
        raise ValueError(f"类别 {category} 的训练集中没有正常样本，无法训练 PatchCore。")
    normal_subset = Subset(train_dataset, normal_indices)
    normal_loader = DataLoader(normal_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=device.type == "cuda")

    train_loader_full = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=device.type == "cuda")

    test_dataset = ImageAnomalyDataset(data_root, category, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=device.type == "cuda")

    patchcore = PatchCore(ResNetEmbedder(use_pretrained=True), device=device)
    patchcore.build_memory_bank(normal_loader)

    train_scores, train_labels, train_paths, train_heatmaps, train_vectors = run_inference(
        train_loader_full, patchcore, desc=f"训练推理 {category}")
    save_top_anomaly_visualizations(
        f"{category}_train",
        train_paths,
        train_heatmaps,
        train_scores,
        train_labels if train_labels.size and not np.any(train_labels == -1) else None,
        results_dir,
        title=f"{category} Train",
    )

    train_metrics: Optional[Dict[str, float]]
    if np.any(train_labels == -1):
        train_metrics = None
        train_threshold = float(np.percentile(train_scores, 90))
        train_pred = (train_scores >= train_threshold).astype(int)
        train_df = pd.DataFrame({
            "image_path": train_paths,
            "anomaly_score": train_scores,
            "predicted_label": train_pred,
        })
        train_df.to_csv(results_dir / f"{category}_train_predictions.csv", index=False)
        print(f"训练集缺少标签，基于 90 分位数阈值得到预测: {results_dir / (category + '_train_predictions.csv')}")
    else:
        train_metrics = evaluator.evaluate(f"{category} (Train)", train_labels, train_scores, store_history=False)
        train_threshold = train_metrics["best_threshold"]
        train_pred = (train_scores >= train_threshold).astype(int)
        train_df = pd.DataFrame({
            "image_path": train_paths,
            "true_label": train_labels,
            "anomaly_score": train_scores,
            "predicted_label": train_pred,
        })
        train_df.to_csv(results_dir / f"{category}_train_predictions.csv", index=False)
        print(f"训练集预测结果已保存: {results_dir / (category + '_train_predictions.csv')}")

    test_scores, test_labels, test_paths, test_heatmaps, test_vectors = run_inference(
        test_loader, patchcore, desc=f"推理 {category}")

    label_array_for_viz: Optional[np.ndarray]
    if np.any(test_labels == -1):
        label_array_for_viz = None
    else:
        label_array_for_viz = test_labels

    save_top_anomaly_visualizations(
        category,
        test_paths,
        test_heatmaps,
        test_scores,
        label_array_for_viz,
        results_dir,
        title=f"{category} Test",
    )

    if label_array_for_viz is None:
        threshold = float(np.percentile(test_scores, 90))
        predicted = (test_scores >= threshold).astype(int)
        results_df = pd.DataFrame({
            "image_path": test_paths,
            "anomaly_score": test_scores,
            "predicted_label": predicted,
        })
        results_df.to_csv(results_dir / f"{category}_predictions.csv", index=False)
        print(f"测试集缺少标签，基于 90 分位数阈值保存预测: {results_dir / (category + '_predictions.csv')}")
        plot_feature_space(category, patchcore.train_vectors, test_vectors, predicted, results_dir)
        return train_metrics, None

    test_metrics = evaluator.evaluate(category, test_labels, test_scores)
    best_threshold = test_metrics["best_threshold"]
    predicted = (test_scores >= best_threshold).astype(int)
    results_df = pd.DataFrame({
        "image_path": test_paths,
        "true_label": test_labels,
        "anomaly_score": test_scores,
        "predicted_label": predicted,
    })
    results_df.to_csv(results_dir / f"{category}_predictions.csv", index=False)
    print(f"预测结果已保存: {results_dir / (category + '_predictions.csv')}")

    plot_feature_space(category, patchcore.train_vectors, test_vectors, test_labels, results_dir)
    return train_metrics, test_metrics


def main() -> None:
    data_root = "Image_Anomaly_Detection"
    categories = ("hazelnut", "zipper")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 0

    print("=" * 60)
    print("图像异常检测任务 (PatchCore)")
    print("=" * 60)

    if not check_data_structure(data_root):
        print("数据结构检查失败，终止运行")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    evaluator = Evaluator()
    summary_train: List[Dict[str, float]] = []
    summary_test: List[Dict[str, float]] = []

    for category in categories:
        train_metrics, test_metrics = run_patchcore_for_category(
            category=category,
            data_root=data_root,
            transform=transform,
            device=device,
            results_dir=results_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            evaluator=evaluator,
        )
        if train_metrics is not None:
            entry_train = {"category": category}
            entry_train.update({k: v for k, v in train_metrics.items() if k != "best_threshold"})
            summary_train.append(entry_train)
        if test_metrics is not None:
            entry_test = {"category": category}
            entry_test.update({k: v for k, v in test_metrics.items() if k != "best_threshold"})
            summary_test.append(entry_test)

    if evaluator.history:
        print("生成评估图表...")
        evaluator.plot_curves(results_dir)

    if summary_train:
        print("=" * 60)
        print("训练集总结指标")
        print("=" * 60)
        train_df = pd.DataFrame(summary_train)
        print(train_df.to_string(index=False))

    if summary_test:
        print("=" * 60)
        print("测试集总结指标")
        print("=" * 60)
        test_df = pd.DataFrame(summary_test)
        print(test_df.to_string(index=False))

    print("任务完成，结果已输出到 results/ 目录。")


if __name__ == "__main__":
    main()
