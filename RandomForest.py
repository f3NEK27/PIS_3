import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Optional, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)
from dataset_manager import DatasetManager

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        max_features: Optional[str] = 'sqrt',
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
    ) -> None:
        """
        Инициализирует классификатор на основе случайного леса.

        Параметры:
            n_estimators (int): Количество деревьев в лесу.
            criterion (str): Критерий оценки качества разбиения:
                - 'gini': индекс Джини;
                - 'entropy': информационный выигрыш по Шеннону.
            max_depth (Optional[int]): Максимальная глубина деревьев.
                None — деревья строятся до исчерпания выборки.
            max_features (str): Число признаков для выбора при разбиении:
                - 'sqrt': корень из общего числа признаков;
                - 'log2': логарифм по основанию 2.
            class_weight (Optional[str]): Веса классов ('balanced' или None).
                'balanced' рекомендуется при сильном дисбалансе (как в PaySim).
            random_state (int): Seed для воспроизводимости результатов.
        """
        self.n_estimators = n_estimators
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Обучает модель случайного леса на обучающих данных.

        Параметры:
            X_train (DataFrame/ndarray): Матрица признаков обучающей выборки.
            y_train (Series/ndarray): Вектор истинных меток классов.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
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
        Вычисляет метрику Accuracy — долю правильно классифицированных объектов.

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
        Вычисляет метрику Precision — точность предсказания положительного класса.

        Параметры:
            y_true (ndarray): Истинные метки.
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
        Вычисляет метрику Recall — полноту (долю найденных мошеннических транзакций).

        Параметры:
            y_true (ndarray): Истинные метки.
            y_pred (ndarray): Предсказанные метки.
            average (str): Стратегия усреднения.

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
            average (str): Стратегия усреднения.

        Возвращает:
            float: Значение F1-метрики ∈ [0, 1].
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
        Строит матрицу ошибок по результатам классификации.

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
        Возвращает сводный отчёт по метрикам классификации.

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

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность признаков в обученной модели, отсортированную по убыванию.

        Возвращает:
            DataFrame: Таблица с признаками и их важностью.

        Исключения:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
        return pd.DataFrame({
            'feature':    self.model.feature_names_in_,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        title: str = 'Random Forest',
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
        path = os.path.join(PLOTS_DIR, 'cm_random_forest.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Матрица ошибок сохранена: {path}")

    def plot_roc_curve(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: np.ndarray,
        title: str = 'Random Forest',
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
        plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'{title} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Случайный классификатор')
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC-кривая — {title}', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'roc_random_forest.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC-кривая сохранена: {path}")

    def plot_feature_importance(self, top_n: int = 12) -> None:
        """
        Строит горизонтальную столбчатую диаграмму важности признаков.

        Параметры:
            top_n (int): Количество наиболее важных признаков для отображения.
        """
        df_imp = self.get_feature_importance().head(top_n)
        df_imp = df_imp.sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(df_imp['feature'], df_imp['importance'],
                 color='#2ecc71', edgecolor='black', alpha=0.85)
        plt.title('Важность признаков — Random Forest', fontweight='bold', fontsize=13)
        plt.xlabel('Gini Importance')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'feature_importance_rf.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График важности признаков сохранён: {path}")

    def plot_tree(
        self,
        tree_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth_display: int = 4,
    ) -> None:
        """
        Визуализирует структуру одного дерева из случайного леса.

        Параметры:
            tree_idx (int): Индекс дерева для отображения.
            feature_names (list): Список имён признаков.
            class_names (list): Список имён классов.
            max_depth_display (int): Максимальная глубина для отображения.
        """
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model.estimators_[tree_idx],
            filled=True,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth_display,
            impurity=True,
            fontsize=9,
        )
        plt.title(f'Дерево №{tree_idx} из Random Forest (первые {max_depth_display} уровня)',
                  fontweight='bold')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f'rf_tree_{tree_idx}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Визуализация дерева #{tree_idx} сохранена: {path}")


if __name__ == '__main__':
    manager = DatasetManager()
    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train, y_train = manager.get_training_data()
    X_test,  y_test  = manager.get_testing_data()

    rf = RandomForestModel(
        n_estimators=100,
        criterion='gini',
        max_depth=15,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
    )
    print("\nОбучение Random Forest...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    report = rf.get_metrics_report(y_test, y_pred, y_prob)
    print("\nОтчёт о метриках классификации (Random Forest):")
    for metric, value in report.items():
        print(f"  {metric}: {value:.4f}")

    print("\nМатрица ошибок:")
    print(rf.calculate_confusion_matrix(y_test, y_pred))

    print("\nВажность признаков:")
    print(rf.get_feature_importance().to_string(index=False))

    rf.plot_confusion_matrix(y_test, y_pred)
    rf.plot_roc_curve(y_test, y_prob)
    rf.plot_feature_importance()
    rf.plot_tree(
        tree_idx=0,
        feature_names=X_train.columns.tolist(),
        class_names=['Нормальная', 'Мошенническая'],
    )
