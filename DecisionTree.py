import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Optional, Dict, List
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)
from dataset_manager import DatasetManager

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class DecisionTreeModel:
    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        class_weight: Optional[str] = 'balanced',
    ) -> None:
        """
        Инициализирует классификатор на основе дерева решений.

        Параметры:
            criterion (str): Критерий оценки качества разбиения:
                - 'gini': индекс Джини;
                - 'entropy': информационный выигрыш по Шеннону.
            max_depth (Optional[int]): Максимальная глубина дерева.
                None — дерево строится до исчерпания выборки.
            class_weight (Optional[str]): Веса классов:
                - 'balanced': автоматически обратно пропорциональны частоте класса
                  (важно при дисбалансе классов в датасете PaySim).
                - None: все классы равноценны.
        """
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.class_weight = class_weight
        self.model: Optional[DecisionTreeClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Обучает модель дерева решений на обучающих данных.

        Параметры:
            X_train (DataFrame/ndarray): Матрица признаков обучающей выборки.
            y_train (Series/ndarray): Вектор истинных меток классов.
        """
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Предсказывает метки классов для новых объектов.

        Параметры:
            X_test (DataFrame/ndarray): Матрица признаков тестовой выборки.

        Возвращает:
            ndarray: Предсказанные метки классов.

        Исключения:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель с помощью fit().")
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
            raise RuntimeError("Сначала обучите модель с помощью fit().")
        return self.model.predict_proba(X_test)

    def calculate_accuracy(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
    ) -> float:
        """
        Вычисляет метрику Accuracy (долю правильных классификаций).

        Параметры:
            y_true (ndarray): Истинные метки.
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
        Вычисляет метрику Precision (точность предсказания классов).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения ('binary', 'macro', 'weighted').

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
        Вычисляет метрику Recall (полноту) — долю найденных положительных объектов.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения.

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
        Вычисляет F1-меру — гармоническое среднее Precision и Recall.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Тип усреднения.

        Возвращает:
            float: Значение F1 ∈ [0, 1].
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_roc_auc(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: np.ndarray,
    ) -> float:
        """
        Вычисляет площадь под ROC-кривой (ROC-AUC).

        Параметры:
            y_true (ndarray): Истинные метки.
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
        Строит матрицу ошибок (confusion matrix).

        Параметры:
            y_true (ndarray): Истинные метки.
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
        Генерирует словарь с основными метриками классификации.

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            y_prob (ndarray, optional): Вероятности для расчёта ROC-AUC.
            average (str): Стратегия усреднения.

        Возвращает:
            Dict[str, float]: Метрики: accuracy, precision, recall, f1, roc_auc.
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

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Возвращает важность признаков по критерию Gini Importance.

        Параметры:
            feature_names (Optional[List[str]]): Названия признаков.
                Если None, используются имена из обученной модели.

        Возвращает:
            DataFrame: Таблица признаков и их важности, отсортированная по убыванию.

        Исключения:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель с помощью fit().")

        names = feature_names or list(getattr(self.model, 'feature_names_in_', []))
        return pd.DataFrame({
            'feature':    names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        title: str = 'Дерево решений',
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
        path = os.path.join(PLOTS_DIR, 'cm_decision_tree.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {path}")

    def plot_roc_curve(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: np.ndarray,
        title: str = 'Дерево решений',
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
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'{title} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Случайный классификатор')
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC-кривая — {title}', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'roc_decision_tree.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC-кривая сохранена: {path}")

    def plot_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 12,
    ) -> None:
        """
        Строит горизонтальную столбчатую диаграмму важности признаков.

        Параметры:
            feature_names (Optional[List[str]]): Названия признаков.
            top_n (int): Сколько наиболее важных признаков отображать.
        """
        df_imp = self.get_feature_importance(feature_names).head(top_n)
        df_imp = df_imp.sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(df_imp['feature'], df_imp['importance'],
                 color='#e74c3c', edgecolor='black', alpha=0.85)
        plt.title('Важность признаков — Дерево решений', fontweight='bold', fontsize=13)
        plt.xlabel('Gini Importance')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'feature_importance_dt.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График важности признаков сохранён: {path}")

    def plot(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth_display: int = 4,
    ) -> None:
        """
        Визуализирует структуру дерева решений (первые уровни) и сохраняет в файл.

        Параметры:
            feature_names (list): Названия признаков.
            class_names (list): Названия классов.
            max_depth_display (int): Максимальная глубина для отображения.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            filled=True,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth_display,
            impurity=True,
            fontsize=9,
        )
        plt.title(f'Дерево решений (первые {max_depth_display} уровня)', fontweight='bold')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'decision_tree_plot.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Визуализация дерева сохранена: {path}")


if __name__ == '__main__':
    manager = DatasetManager()
    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train, y_train = manager.get_training_data()
    X_test,  y_test  = manager.get_testing_data()

    dt = DecisionTreeModel(criterion='gini', max_depth=10, class_weight='balanced')
    print("\nОбучение дерева решений...")
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]

    report = dt.get_metrics_report(y_test, y_pred, y_prob)
    print("\nОтчёт о метриках классификации (Дерево решений):")
    for metric, value in report.items():
        print(f"  {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(dt.calculate_confusion_matrix(y_test, y_pred))

    print("\nВажность признаков:")
    print(dt.get_feature_importance(X_train.columns.tolist()).to_string(index=False))

    dt.plot_confusion_matrix(y_test, y_pred)
    dt.plot_roc_curve(y_test, y_prob)
    dt.plot_feature_importance(feature_names=X_train.columns.tolist())
    dt.plot(
        feature_names=X_train.columns.tolist(),
        class_names=['Нормальная', 'Мошенническая'],
    )
