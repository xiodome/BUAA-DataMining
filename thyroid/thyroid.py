import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置中文字体以避免可视化乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class UnsupervisedDiseaseDetector:
    """基于孤立森林的无监督疾病检测器"""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model: Optional[IsolationForest] = None
        self.results: dict[str, object] = {}
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None

    def load_data(self, train_path: str, test_path: str) -> list[str]:
        print('=' * 60)
        print('1. 数据加载与探索性分析')
        print('=' * 60)

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print()
        print(f'训练集形状: {train_df.shape}')
        print(f'测试集形状: {test_df.shape}')

        feature_cols = [col for col in train_df.columns if col.startswith('feature')]
        if not feature_cols:
            raise ValueError('未找到以 feature 开头的特征列。')

        self.X_train = train_df[feature_cols].to_numpy()
        self.X_test = test_df[feature_cols].to_numpy()
        self.y_test = test_df['label'].to_numpy()

        print()
        print(f'特征维度: {len(feature_cols)}')
        print(f'训练集样本数: {len(self.X_train)} (全部为正常样本)')
        print(f'测试集样本数: {len(self.X_test)}')
        positives = int(np.sum(self.y_test == 1))
        pos_rate = float(np.mean(self.y_test)) * 100
        print(f'测试集中患病样本数: {positives} ({pos_rate:.2f}%)')
        print(f'测试集中正常样本数: {len(self.X_test) - positives} ({100 - pos_rate:.2f}%)')

        return feature_cols

    def explore_data(self, feature_cols: list[str]) -> None:
        if self.X_train is None or self.X_test is None or self.y_test is None:
            raise RuntimeError('请先调用 load_data 加载数据。')

        print()
        print('=' * 60)
        print('2. 数据特征分析')
        print('=' * 60)

        print()
        print('训练集特征统计:')
        print(pd.DataFrame(self.X_train, columns=feature_cols).describe())

        print()
        print(f'训练集缺失值: {int(np.isnan(self.X_train).sum())}')
        print(f'测试集缺失值: {int(np.isnan(self.X_test).sum())}')

        rows = 2
        cols = int(np.ceil(len(feature_cols) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for idx, col in enumerate(feature_cols):
            axes[idx].hist(self.X_train[:, idx], bins=50, alpha=0.7, label='训练集(正常)', color='steelblue')
            axes[idx].hist(self.X_test[self.y_test == 0, idx], bins=50, alpha=0.5, label='测试集(正常)', color='seagreen')
            axes[idx].hist(self.X_test[self.y_test == 1, idx], bins=50, alpha=0.5, label='测试集(患病)', color='indianred')
            axes[idx].set_title(f'{col} 分布')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        for idx in range(len(feature_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('result/feature_distributions.png', dpi=300, bbox_inches='tight')
        print()
        print('特征分布图已保存为: result/feature_distributions.png')
        plt.close()

        correlation_matrix = np.corrcoef(self.X_train.T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                    xticklabels=feature_cols, yticklabels=feature_cols,
                    cmap='coolwarm', center=0)
        plt.title('训练集特征相关性矩阵')
        plt.tight_layout()
        plt.savefig('result/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print('特征相关性矩阵已保存为: result/correlation_matrix.png')
        plt.close()

    def preprocess_data(self) -> None:
        if self.X_train is None or self.X_test is None:
            raise RuntimeError('请先调用 load_data 加载数据。')

        print()
        print('=' * 60)
        print('3. 数据预处理')
        print('=' * 60)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print()
        print('数据已进行标准化处理')
        print(f'标准化后训练集均值: {self.X_train_scaled.mean(axis=0)}')
        print(f'标准化后训练集标准差: {self.X_train_scaled.std(axis=0)}')

    def train_model(self, contamination: float = 0.05, n_estimators: int = 200) -> None:
        if self.X_train_scaled is None:
            raise RuntimeError('请先调用 preprocess_data 进行标准化。')

        print()
        print('=' * 60)
        print('4. 模型训练')
        print('=' * 60)
        print()
        print(f'Isolation Forest 训练中, contamination={contamination:.3f}, n_estimators={n_estimators}')
        print()

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train_scaled)
        print('模型训练完成!')

    def evaluate_model(self) -> None:
        if self.model is None or self.X_test_scaled is None or self.y_test is None:
            raise RuntimeError('请确认模型已训练且测试数据已准备。')

        print()
        print('=' * 60)
        print('5. 模型评估')
        print('=' * 60)

        raw_pred = self.model.predict(self.X_test_scaled)
        y_pred = (raw_pred == -1).astype(int)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)

        try:
            scores = -self.model.score_samples(self.X_test_scaled)
            auc = roc_auc_score(self.y_test, scores)
            avg_precision = average_precision_score(self.y_test, scores)
        except ValueError:
            scores = None
            auc = None
            avg_precision = None

        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'scores': scores
        }

        self.print_results()
        self.plot_results()

    def print_results(self) -> None:
        if not self.results:
            raise RuntimeError('未找到评估结果。')

        result = self.results
        summary = pd.DataFrame({
            'Accuracy': [f"{result['accuracy']:.4f}"],
            'Precision': [f"{result['precision']:.4f}"],
            'Recall': [f"{result['recall']:.4f}"],
            'F1-Score': [f"{result['f1']:.4f}"],
            'AUC-ROC': [f"{result['auc']:.4f}" if result['auc'] else 'N/A'],
            'Avg Precision': [f"{result['avg_precision']:.4f}" if result['avg_precision'] else 'N/A']
        })

        print()
        print('孤立森林评估指标:')
        print(summary)

        tn, fp, fn, tp = result['confusion_matrix'].ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        print()
        print('-' * 60)
        print('混淆矩阵')
        print('-' * 60)
        print(f'真阴性(TN): {tn}, 假阳性(FP): {fp}')
        print(f'假阴性(FN): {fn}, 真阳性(TP): {tp}')
        print(f'特异度(Specificity): {specificity:.4f}')
        print(f"灵敏度(Sensitivity/Recall): {result['recall']:.4f}")

    def plot_results(self) -> None:
        if not self.results:
            raise RuntimeError('未找到评估结果。')

        cm = self.results['confusion_matrix']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测正常', '预测患病'],
                    yticklabels=['实际正常', '实际患病'])
        plt.title('Isolation Forest 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('result/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print()
        print('混淆矩阵图已保存为: result/confusion_matrix.png')
        plt.close()

        if self.results['scores'] is not None:
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(self.y_test, self.results['scores'])
            auc = self.results['auc']
            plt.plot(fpr, tpr, label=f'Isolation Forest (AUC={auc:.4f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
            plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
            plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
            plt.title('ROC曲线', fontsize=14, fontweight='bold')
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('result/roc_curve.png', dpi=300, bbox_inches='tight')
            print('ROC曲线图已保存为: result/roc_curve.png')
            plt.close()

            plt.figure(figsize=(10, 8))
            precision, recall, _ = precision_recall_curve(self.y_test, self.results['scores'])
            avg_prec = self.results['avg_precision']
            plt.plot(recall, precision, label=f'Isolation Forest (AP={avg_prec:.4f})', linewidth=2)
            plt.xlabel('召回率 (Recall)', fontsize=12)
            plt.ylabel('精确率 (Precision)', fontsize=12)
            plt.title('Precision-Recall曲线', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('result/pr_curve.png', dpi=300, bbox_inches='tight')
            print('PR曲线图已保存为: result/pr_curve.png')
            plt.close()

        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(self.X_test_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        scatter = axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                                  c=self.y_test, cmap='RdYlGn_r',
                                  alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[0].set_title('真实标签', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[0], label='0=正常, 1=患病')

        scatter = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                                  c=self.results['predictions'], cmap='RdYlGn_r',
                                  alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[1].set_title('Isolation Forest 预测', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[1], label='0=正常, 1=患病')

        plt.tight_layout()
        plt.savefig('result/pca_visualization.png', dpi=300, bbox_inches='tight')
        print('PCA可视化图已保存为: result/pca_visualization.png')
        plt.close()

    def get_model(self) -> IsolationForest:
        if self.model is None:
            raise RuntimeError('模型尚未训练。')
        return self.model


def main() -> None:
    print()
    print('=' * 60)
    print('无监督疾病判断任务')
    print('=' * 60)

    detector = UnsupervisedDiseaseDetector()

    train_path = 'thyroid/train-set.csv'
    test_path = 'thyroid/test-set.csv'

    feature_cols = detector.load_data(train_path, test_path)
    detector.explore_data(feature_cols)
    detector.preprocess_data()
    detector.train_model(contamination=0.024, n_estimators=1000)
    detector.evaluate_model()
    _ = detector.get_model()

    print()
    print('=' * 60)
    print('任务完成!')
    print('=' * 60)
    print()
    print('生成的文件:')
    print('- feature_distributions.png: 特征分布图')
    print('- correlation_matrix.png: 特征相关性矩阵')
    print('- confusion_matrix.png: 混淆矩阵')
    print('- roc_curve.png: ROC曲线 (如可计算)')
    print('- pr_curve.png: Precision-Recall曲线 (如可计算)')
    print('- pca_visualization.png: PCA降维可视化')


if __name__ == '__main__':
    main()
