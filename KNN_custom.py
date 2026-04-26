import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Optional, Dict
from collections import Counter
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
        Собственная реализация классификатора k-ближайших соседей (KNN).

        Параметры:
            n_neighbors (int): Количество ближайших соседей (по умолчанию 5).
            weights (str): Стратегия взвешивания:
                - 'uniform': все соседи имеют равный вес;
                - 'distance': вес обратно пропорционален расстоянию.
            metric (str): Метрика расстояния:
                - 'euclidean': евклидово расстояние;
                - 'manhattan': манхэттенское расстояние;
                - 'cosine': косинусное расстояние.
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
        Сохраняет обучающую выборку в памяти модели (ленивое обучение).

        Параметры:
            X_train (DataFrame/ndarray): Матрица признаков обучающих объектов.
            y_train (Series/ndarray): Вектор меток классов.
        """
        self.X_train  = np.array(X_train)
        self.y_train  = np.array(y_train)
        self.classes_ = np.unique(self.y_train)

    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычисляет расстояние между двумя точками в заданной метрике.

        Параметры:
            a (ndarray): Первая точка (вектор признаков).
            b (ndarray): Вторая точка (вектор признаков).

        Возвращает:
            float: Расстояние между a и b.

        Исключения:
            ValueError: если указана неизвестная метрика.
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        elif self.metric == 'cosine':
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - np.dot(a, b) / (norm_a * norm_b)
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}. "
                             f"Доступные: 'euclidean', 'manhattan', 'cosine'.")

    def _distances_vectorized(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет расстояния от одного объекта до всех объектов обучающей выборки
        векторизованным способом (без цикла по обучающим объектам).

        Параметры:
            x (ndarray): Тестовый объект (вектор признаков).

        Возвращает:
            ndarray: Вектор расстояний длиной len(X_train).
        """
        if self.metric == 'euclidean':
            diff = self.X_train - x
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'cosine':
            norms_train = np.linalg.norm(self.X_train, axis=1)
            norm_x      = np.linalg.norm(x)
            if norm_x == 0:
                return np.ones(len(self.X_train))
            dots = self.X_train @ x
            denom = norms_train * norm_x
            denom[denom == 0] = 1e-10
            return 1.0 - dots / denom
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}.")

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для объектов тестовой выборки.

        Параметры:
            X_test (DataFrame/ndarray): Матрица признаков тестовых объектов.

        Возвращает:
            ndarray: Вектор предсказанных меток.

        Исключения:
            RuntimeError: если модель не была обучена методом fit().
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_test_arr  = np.array(X_test)
        predictions = []

        for x in X_test_arr:
            distances  = self._distances_vectorized(x)
            k_indices  = np.argsort(distances)[:self.n_neighbors]
            k_labels   = self.y_train[k_indices]
            k_distances = distances[k_indices]

            if self.weights == 'distance':
                weights = 1.0 / (k_distances + 1e-8)
            else:
                weights = np.ones(self.n_neighbors)

            counter = Counter()
            for label, w in zip(k_labels, weights):
                counter[label] += w

            predictions.append(max(counter, key=counter.get))

        return np.array(predictions)

    def calculate_accuracy(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> float:
        """
        Вычисляет точность (Accuracy) — долю правильных предсказаний.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

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
        Строит матрицу ошибок (confusion matrix).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица [n_classes x n_classes].
        """
        y_true    = np.array(y_true)
        n_classes = len(self.classes_)
        matrix    = np.zeros((n_classes, n_classes), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for true, pred in zip(y_true, y_pred):
            matrix[class_to_idx[true], class_to_idx[pred]] += 1

        return matrix

    def _calculate_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """
        Вычисляет метрики precision, recall и F1 для каждого класса отдельно.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            Dict: Для каждого класса — precision, recall, f1, support.
        """
        matrix  = self.calculate_confusion_matrix(y_true, y_pred)
        metrics = {}

        for i, class_label in enumerate(self.classes_):
            tp = matrix[i, i]
            fp = np.sum(matrix[:, i]) - tp
            fn = np.sum(matrix[i, :]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            metrics[class_label] = {
                'precision': precision,
                'recall':    recall,
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
        Вычисляет метрику Precision (точность предсказания классов).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения ('binary', 'macro', 'weighted', 'micro').

        Возвращает:
            float: Значение Precision.
        """
        y_true   = np.array(y_true)
        metrics  = self._calculate_class_metrics(y_true, y_pred)
        positive = self.classes_[-1]  # при бинарной классификации — класс 1

        if average == 'binary':
            return metrics[positive]['precision']

        precisions = [m['precision'] for m in metrics.values()]
        supports   = [m['support']   for m in metrics.values()]

        if average == 'macro':
            return float(np.mean(precisions))
        elif average == 'weighted':
            return float(np.average(precisions, weights=supports))
        elif average == 'micro':
            matrix = self.calculate_confusion_matrix(y_true, y_pred)
            tp     = np.sum(np.diag(matrix))
            fp     = np.sum(matrix, axis=0) - np.diag(matrix)
            return float(tp / (tp + np.sum(fp)))
        else:
            raise ValueError("Доступные варианты: 'binary', 'macro', 'weighted', 'micro'.")

    def calculate_recall(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисляет метрику Recall (полнота — доля найденных объектов класса).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения ('binary', 'macro', 'weighted', 'micro').

        Возвращает:
            float: Значение Recall ∈ [0, 1].
        """
        y_true   = np.array(y_true)
        metrics  = self._calculate_class_metrics(y_true, y_pred)
        positive = self.classes_[-1]

        if average == 'binary':
            return metrics[positive]['recall']

        recalls  = [m['recall']  for m in metrics.values()]
        supports = [m['support'] for m in metrics.values()]

        if average == 'macro':
            return float(np.mean(recalls))
        elif average == 'weighted':
            return float(np.average(recalls, weights=supports))
        elif average == 'micro':
            matrix = self.calculate_confusion_matrix(y_true, y_pred)
            tp     = np.sum(np.diag(matrix))
            fn     = np.sum(matrix, axis=1) - np.diag(matrix)
            return float(tp / (tp + np.sum(fn)))
        else:
            raise ValueError("Доступные варианты: 'binary', 'macro', 'weighted', 'micro'.")

    def calculate_f1(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисляет F1-меру (гармоническое среднее Precision и Recall).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения.

        Возвращает:
            float: Значение F1-меры ∈ [0, 1].
        """
        prec = self.calculate_precision(y_true, y_pred, average)
        rec  = self.calculate_recall(y_true, y_pred, average)
        return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    def get_metrics_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> Dict[str, float]:
        """
        Генерирует сводный отчёт по метрикам классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Метод усреднения.

        Возвращает:
            Dict[str, float]: Метрики: accuracy, precision, recall, f1.
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
        title: str = 'KNN Custom',
    ) -> None:
        """
        Строит и сохраняет матрицу ошибок в виде тепловой карты.

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            title (str): Заголовок графика.
        """
        cm = self.calculate_confusion_matrix(np.array(y_true), y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Нормальная', 'Мошенническая'],
                    yticklabels=['Нормальная', 'Мошенническая'])
        plt.title(f'Матрица ошибок — {title}', fontweight='bold')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'cm_knn_custom.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {path}")


if __name__ == '__main__':
    # Для кастомного KNN берём подвыборку — полный датасет займёт слишком много времени
    # из-за попарного вычисления расстояний O(n*m*d)
    manager = DatasetManager()
    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train_full, y_train_full = manager.get_training_data()
    X_test_full,  y_test_full  = manager.get_testing_data()

    # Случайная подвыборка для демонстрации кастомного KNN
    SAMPLE_TRAIN = 5000
    SAMPLE_TEST  = 1000
    idx_tr = np.random.RandomState(42).choice(len(X_train_full), SAMPLE_TRAIN, replace=False)
    idx_te = np.random.RandomState(42).choice(len(X_test_full),  SAMPLE_TEST,  replace=False)

    X_train_s = X_train_full.iloc[idx_tr].values
    y_train_s = np.array(y_train_full)[idx_tr]
    X_test_s  = X_test_full.iloc[idx_te].values
    y_test_s  = np.array(y_test_full)[idx_te]

    print(f"\nПодвыборка для кастомного KNN: train={SAMPLE_TRAIN}, test={SAMPLE_TEST}")

    knn = KNNCustom(n_neighbors=5, weights='distance', metric='euclidean')
    knn.fit(X_train_s, y_train_s)

    print("Предсказание...")
    y_pred = knn.predict(X_test_s)

    report = knn.get_metrics_report(y_test_s, y_pred)
    print("\nОтчёт о метриках классификации (KNN custom):")
    for metric, value in report.items():
        print(f"  {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(knn.calculate_confusion_matrix(y_test_s, y_pred))

    knn.plot_confusion_matrix(y_test_s, y_pred, title='KNN (собственная реализация)')
