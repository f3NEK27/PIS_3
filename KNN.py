import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Optional, Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)
from dataset_manager import DatasetManager

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class KNNClassifier:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'minkowski',
    ) -> None:
        """
        Классификатор k-ближайших соседей (KNN) на основе sklearn.

        Параметры:
            n_neighbors (int): Количество соседей (по умолчанию 5).
            weights (str): Стратегия взвешивания:
                - 'uniform': равные веса;
                - 'distance': вес обратно пропорционален расстоянию.
            metric (str): Метрика для расчёта расстояний (по умолчанию 'minkowski').
        """
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.model: Optional[KNeighborsClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Обучение модели на обучающих данных.

        Параметры:
            X_train (DataFrame/ndarray): Матрица признаков обучающей выборки.
            y_train (Series/ndarray): Вектор целевых меток.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказание классов для новых данных.

        Параметры:
            X_test (DataFrame/ndarray): Матрица признаков тестовой выборки.

        Возвращает:
            ndarray: Массив предсказанных меток.

        Исключения:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала выполните обучение модели (fit()).")
        return self.model.predict(X_test)

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Возвращает вероятности принадлежности к каждому классу.

        Параметры:
            X_test (DataFrame/ndarray): Матрица признаков тестовой выборки.

        Возвращает:
            ndarray: Массив вероятностей формы (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Сначала выполните обучение модели (fit()).")
        return self.model.predict_proba(X_test)

    def calculate_accuracy(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> float:
        """
        Вычисление метрики Accuracy (доля правильных предсказаний).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            float: Значение Accuracy ∈ [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    def calculate_precision(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисление метрики Precision (точность).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('binary', 'macro', 'weighted').

        Возвращает:
            float: Значение Precision.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_recall(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисление метрики Recall (полнота).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения ('binary', 'macro', 'weighted').

        Возвращает:
            float: Значение Recall ∈ [0, 1].
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_f1(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        average: str = 'binary',
    ) -> float:
        """
        Вычисление F1-меры (гармоническое среднее Precision и Recall).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения.

        Возвращает:
            float: Значение F1-score ∈ [0, 1].
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_roc_auc(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: np.ndarray,
    ) -> float:
        """
        Вычисление площади под ROC-кривой (ROC-AUC).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_prob (ndarray): Вероятности положительного класса.

        Возвращает:
            float: Значение ROC-AUC ∈ [0, 1].
        """
        return roc_auc_score(y_true, y_prob)

    def calculate_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Построение матрицы ошибок (confusion matrix).

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.

        Возвращает:
            ndarray: Матрица [2 x 2] для бинарной классификации.
        """
        return confusion_matrix(y_true, y_pred)

    def get_metrics_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'binary',
    ) -> Dict[str, float]:
        """
        Генерация сводного отчёта по основным метрикам классификации.

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            y_prob (ndarray, optional): Вероятности положительного класса для ROC-AUC.
            average (str): Стратегия усреднения.

        Возвращает:
            Dict[str, float]: Словарь метрик: accuracy, precision, recall, f1, roc_auc.
        """
        report = {
            'accuracy':  self.calculate_accuracy(y_true, y_pred),
            'precision': self.calculate_precision(y_true, y_pred, average),
            'recall':    self.calculate_recall(y_true, y_pred, average),
            'f1':        self.calculate_f1(y_true, y_pred, average),
        }
        if y_prob is not None:
            report['roc_auc'] = self.calculate_roc_auc(y_true, y_prob)
        return report

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        title: str = 'KNN',
    ) -> None:
        """
        Строит и сохраняет матрицу ошибок в виде тепловой карты.

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            title (str): Заголовок графика.
        """
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Нормальная', 'Мошенническая'],
                    yticklabels=['Нормальная', 'Мошенническая'])
        plt.title(f'Матрица ошибок — {title}', fontweight='bold')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f'cm_knn.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {path}")

    def plot_roc_curve(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: np.ndarray,
        title: str = 'KNN',
    ) -> None:
        """
        Строит и сохраняет ROC-кривую модели.

        Параметры:
            y_true (Series/ndarray): Истинные метки.
            y_prob (ndarray): Вероятности положительного класса.
            title (str): Заголовок графика.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'{title} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Случайный классификатор')
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC-кривая — {title}', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'roc_knn.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC-кривая сохранена: {path}")


if __name__ == '__main__':
    manager = DatasetManager()
    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train, y_train = manager.get_training_data()
    X_test,  y_test  = manager.get_testing_data()

    knn = KNNClassifier(n_neighbors=5, weights='distance', metric='minkowski')
    print("\nОбучение KNN...")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]

    report = knn.get_metrics_report(y_test, y_pred, y_prob)
    print("\nОтчёт о метриках классификации (KNN sklearn):")
    for metric, value in report.items():
        print(f"  {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(knn.calculate_confusion_matrix(y_test, y_pred))

    knn.plot_confusion_matrix(y_test, y_pred, title='KNN (sklearn)')
    knn.plot_roc_curve(y_test, y_prob, title='KNN (sklearn)')
