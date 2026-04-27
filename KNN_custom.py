"""
KNN_custom.py — собственная реализация классификатора k-ближайших соседей.

Особенности реализации:
    - Поддержка трёх метрик: евклидова, манхэттенская, косинусная.
    - Два режима взвешивания: равные веса и веса, обратно пропорциональные расстоянию.
    - Векторизованный расчёт расстояний через numpy (без цикла по обучающим объектам).
    - Собственный расчёт метрик (accuracy, precision, recall, F1) без sklearn.

Примечание: из-за O(n * m * d) сложности по памяти и времени при предсказании
полный датасет PaySim (6 млн строк) не подходит для кастомного KNN напрямую.
В __main__ используется подвыборка для демонстрации корректности реализации.
"""

import os
from collections import Counter
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset_manager import DatasetManager

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class KNNCustom:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'euclidean',
    ) -> None:
        """
        Собственная реализация классификатора k-ближайших соседей.

        Параметры:
            n_neighbors (int): Количество ближайших соседей.
            weights (str): Стратегия взвешивания:
                - 'uniform': все соседи имеют равный вес;
                - 'distance': вес обратно пропорционален расстоянию.
            metric (str): Метрика расстояния:
                - 'euclidean': евклидово расстояние √Σ(a-b)²;
                - 'manhattan': манхэттенское расстояние Σ|a-b|;
                - 'cosine': косинусное расстояние 1 - cos(a,b).
        """
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Сохраняет обучающую выборку (ленивое обучение — вся работа при predict).

        Параметры:
            X_train: Матрица признаков обучающих объектов.
            y_train: Вектор меток классов.
        """
        self.X_train  = np.array(X_train, dtype=np.float64)
        self.y_train  = np.array(y_train)
        self.classes_ = np.unique(self.y_train)

    def _distances_to_all(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет расстояния от одного тестового объекта до всех обучающих
        векторизованным способом (без Python-цикла по обучающим объектам).

        Параметры:
            x (ndarray): Тестовый объект — вектор признаков формы (d,).

        Возвращает:
            ndarray: Вектор расстояний формы (n_train,).

        Исключения:
            ValueError: если указана неизвестная метрика.
        """
        if self.metric == 'euclidean':
            diff = self.X_train - x
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))

        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)

        elif self.metric == 'cosine':
            dots        = self.X_train @ x
            norms_train = np.linalg.norm(self.X_train, axis=1)
            norm_x      = np.linalg.norm(x)
            denom       = norms_train * norm_x
            denom       = np.where(denom == 0, 1e-10, denom)
            return 1.0 - dots / denom

        else:
            raise ValueError(
                f"Неизвестная метрика: '{self.metric}'. "
                "Доступные: 'euclidean', 'manhattan', 'cosine'."
            )

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для объектов тестовой выборки.

        Алгоритм для каждого тестового объекта:
            1. Вычислить расстояния до всех обучающих объектов.
            2. Выбрать k ближайших.
            3. Взвесить их метки (равномерно или по расстоянию).
            4. Вернуть класс с наибольшим суммарным весом.

        Параметры:
            X_test: Матрица признаков тестовых объектов.

        Возвращает:
            ndarray: Вектор предсказанных меток.

        Исключения:
            RuntimeError: если модель не была обучена методом fit().
        """
        if self.X_train is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_test_arr  = np.array(X_test, dtype=np.float64)
        predictions = np.empty(len(X_test_arr), dtype=self.y_train.dtype)

        for i, x in enumerate(X_test_arr):
            distances  = self._distances_to_all(x)
            k_idx      = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            k_labels   = self.y_train[k_idx]
            k_distances = distances[k_idx]

            if self.weights == 'distance':
                w = 1.0 / (k_distances + 1e-8)
            else:
                w = np.ones(self.n_neighbors)

            vote = Counter()
            for label, weight in zip(k_labels, w):
                vote[label] += weight

            predictions[i] = max(vote, key=vote.get)

        return predictions

    def calculate_accuracy(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> float:
        """
        Вычисляет долю правильных предсказаний (Accuracy).

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.

        Возвращает:
            float: Значение Accuracy ∈ [0, 1].
        """
        y_true = np.array(y_true)
        return float(np.sum(y_true == y_pred) / len(y_true))

    def calculate_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Строит матрицу ошибок (confusion matrix) без использования sklearn.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.

        Возвращает:
            ndarray: Матрица [n_classes x n_classes].
        """
        y_true       = np.array(y_true)
        n            = len(self.classes_)
        cls_to_idx   = {c: i for i, c in enumerate(self.classes_)}
        matrix       = np.zeros((n, n), dtype=int)

        for t, p in zip(y_true, y_pred):
            matrix[cls_to_idx[t], cls_to_idx[p]] += 1

        return matrix

    def _per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Вычисляет precision, recall и F1 для каждого класса отдельно.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.

        Возвращает:
            Dict: {класс: {'precision', 'recall', 'f1', 'support'}}.
        """
        matrix  = self.calculate_confusion_matrix(y_true, y_pred)
        metrics = {}

        for i, cls in enumerate(self.classes_):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            fn = matrix[i, :].sum() - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            metrics[cls] = {
                'precision': prec,
                'recall':    rec,
                'f1':        f1,
                'support':   int(tp + fn),
            }
        return metrics

    def calculate_precision(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисляет Precision.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.
            average (str): 'binary', 'macro', 'weighted' или 'micro'.

        Возвращает:
            float: Значение Precision.
        """
        y_true  = np.array(y_true)
        metrics = self._per_class_metrics(y_true, y_pred)

        if average == 'binary':
            return metrics[self.classes_[-1]]['precision']

        precs    = [m['precision'] for m in metrics.values()]
        supports = [m['support']   for m in metrics.values()]

        if average == 'macro':
            return float(np.mean(precs))
        elif average == 'weighted':
            return float(np.average(precs, weights=supports))
        elif average == 'micro':
            cm = self.calculate_confusion_matrix(y_true, y_pred)
            tp = np.diag(cm).sum()
            fp = (cm.sum(axis=0) - np.diag(cm)).sum()
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        else:
            raise ValueError("average: 'binary', 'macro', 'weighted', 'micro'.")

    def calculate_recall(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисляет Recall.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.
            average (str): 'binary', 'macro', 'weighted' или 'micro'.

        Возвращает:
            float: Значение Recall ∈ [0, 1].
        """
        y_true  = np.array(y_true)
        metrics = self._per_class_metrics(y_true, y_pred)

        if average == 'binary':
            return metrics[self.classes_[-1]]['recall']

        recs     = [m['recall']  for m in metrics.values()]
        supports = [m['support'] for m in metrics.values()]

        if average == 'macro':
            return float(np.mean(recs))
        elif average == 'weighted':
            return float(np.average(recs, weights=supports))
        elif average == 'micro':
            cm = self.calculate_confusion_matrix(y_true, y_pred)
            tp = np.diag(cm).sum()
            fn = (cm.sum(axis=1) - np.diag(cm)).sum()
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        else:
            raise ValueError("average: 'binary', 'macro', 'weighted', 'micro'.")

    def calculate_f1(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисляет F1-меру.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.
            average (str): Стратегия усреднения.

        Возвращает:
            float: Значение F1-score ∈ [0, 1].
        """
        p = self.calculate_precision(y_true, y_pred, average)
        r = self.calculate_recall(y_true, y_pred, average)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def get_metrics_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> Dict[str, float]:
        """
        Формирует сводный отчёт по метрикам классификации.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.
            average (str): Стратегия усреднения.

        Возвращает:
            Dict[str, float]: accuracy, precision, recall, f1.
        """
        return {
            'accuracy':  self.calculate_accuracy(y_true, y_pred),
            'precision': self.calculate_precision(y_true, y_pred, average),
            'recall':    self.calculate_recall(y_true, y_pred, average),
            'f1':        self.calculate_f1(y_true, y_pred, average),
        }

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        title: str = 'KNN (собственная реализация)',
    ) -> None:
        """
        Строит и сохраняет матрицу ошибок в виде тепловой карты.

        Параметры:
            y_true: Истинные метки.
            y_pred: Предсказанные метки.
            title (str): Заголовок графика.
        """
        cm = self.calculate_confusion_matrix(np.array(y_true), y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Нормальная', 'Мошенническая'],
            yticklabels=['Нормальная', 'Мошенническая'],
        )
        plt.title(f'Матрица ошибок — {title}', fontweight='bold')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'cm_knn_custom.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {path}")


# ── Точка входа ───────────────────────────────────────────────────────

if __name__ == '__main__':
    manager = DatasetManager()
    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train_full, y_train_full = manager.get_training_data()
    X_test_full,  y_test_full  = manager.get_testing_data()

    # Подвыборка: кастомный KNN работает O(n*m*d) — на полном датасете слишком долго.
    # Стратифицированная подвыборка 50/50 гарантирует наличие мошенников в обеих выборках.
    N_EACH_TRAIN = 2500   # по классу, итого ~5000 train
    N_EACH_TEST  = 500    # по классу, итого ~1000 test

    y_train_arr = np.array(y_train_full)
    y_test_arr  = np.array(y_test_full)
    rng = np.random.default_rng(42)

    idx_tr_fraud  = np.where(y_train_arr == 1)[0]
    idx_tr_normal = np.where(y_train_arr == 0)[0]
    idx_tr = np.concatenate([
        rng.choice(idx_tr_fraud,  min(N_EACH_TRAIN, len(idx_tr_fraud)),  replace=False),
        rng.choice(idx_tr_normal, min(N_EACH_TRAIN, len(idx_tr_normal)), replace=False),
    ])

    idx_te_fraud  = np.where(y_test_arr == 1)[0]
    idx_te_normal = np.where(y_test_arr == 0)[0]
    idx_te = np.concatenate([
        rng.choice(idx_te_fraud,  min(N_EACH_TEST, len(idx_te_fraud)),  replace=False),
        rng.choice(idx_te_normal, min(N_EACH_TEST, len(idx_te_normal)), replace=False),
    ])

    X_train_s = X_train_full.iloc[idx_tr].values
    y_train_s = y_train_arr[idx_tr]
    X_test_s  = X_test_full.iloc[idx_te].values
    y_test_s  = y_test_arr[idx_te]

    print(f"\nСтратифицированная подвыборка: train={len(idx_tr)}, test={len(idx_te)}")
    print(f"Мошенников в train: {y_train_s.sum()}  |  в test: {y_test_s.sum()}")

    knn = KNNCustom(n_neighbors=5, weights='distance', metric='euclidean')
    knn.fit(X_train_s, y_train_s)

    print("Предсказание...")
    y_pred = knn.predict(X_test_s)

    report = knn.get_metrics_report(y_test_s, y_pred)
    print("\nМетрики KNN (собственная реализация):")
    for metric, value in report.items():
        print(f"  {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(knn.calculate_confusion_matrix(y_test_s, y_pred))

    knn.plot_confusion_matrix(y_test_s, y_pred)
