"""
dataset_manager.py — загрузка, предобработка и визуализация датасета PaySim.

Датасет: синтетические финансовые транзакции для обнаружения мошенничества.
Источник: https://www.kaggle.com/datasets/ealaxi/paysim1
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class DatasetManager:
    def __init__(self, csv_path: str = 'PS_20174392719_1491204439457_log.csv') -> None:
        """
        Инициализирует менеджер датасета PaySim.

        Параметры:
            csv_path (str): Путь к CSV-файлу датасета.
        """
        self.csv_path = csv_path
        self.df: Optional[DataFrame] = None
        self.features: Optional[DataFrame] = None
        self.target: Optional[Series] = None
        self.scaled_features: Optional[DataFrame] = None
        self.stats: Dict = {}

        self.X_train: Optional[DataFrame] = None
        self.X_test: Optional[DataFrame] = None
        self.y_train: Optional[Series] = None
        self.y_test: Optional[Series] = None

        self._load_data()
        self._engineer_features()
        self._extract_features_target()

    # ── Загрузка ─────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """
        Загружает CSV-файл датасета PaySim в self.df.

        Выбрасывает:
            FileNotFoundError: если файл не найден по указанному пути.
        """
        print(f"Загружаем датасет: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Данные загружены: {self.df.shape[0]:,} строк, {self.df.shape[1]} столбцов")

    # ── Инжиниринг признаков ──────────────────────────────────────────

    def _engineer_features(self) -> None:
        """
        Создаёт новые признаки и удаляет нечисловые столбцы.

        Новые признаки:
            type_encoded        — числовое кодирование типа операции (LabelEncoder).
            balance_diff_orig   — изменение баланса отправителя (new - old).
            balance_diff_dest   — изменение баланса получателя (new - old).
            orig_zeroed         — 1, если счёт отправителя обнулился после транзакции.
            suspicious_type     — 1, если тип операции CASH_OUT или TRANSFER
                                  (только эти типы содержат мошенничество в датасете).
            amount_balance_error— |amount + balance_diff_orig|: при честной операции ~0.

        Удаляются: nameOrig, nameDest (строковые ID), type (заменён на type_encoded),
                   isFlaggedFraud (дублирует часть информации о мошенничестве).
        """
        le = LabelEncoder()
        self.df['type_encoded']         = le.fit_transform(self.df['type'])
        self.df['balance_diff_orig']    = self.df['newbalanceOrig'] - self.df['oldbalanceOrg']
        self.df['balance_diff_dest']    = self.df['newbalanceDest'] - self.df['oldbalanceDest']
        self.df['orig_zeroed']          = (
            (self.df['oldbalanceOrg'] > 0) & (self.df['newbalanceOrig'] == 0)
        ).astype(int)
        self.df['suspicious_type']      = self.df['type'].isin(
            ['CASH_OUT', 'TRANSFER']
        ).astype(int)
        self.df['amount_balance_error'] = (
            self.df['amount'] + self.df['balance_diff_orig']
        ).abs()

        self.df.drop(columns=['nameOrig', 'nameDest', 'type', 'isFlaggedFraud'], inplace=True)
        print("Инжиниринг признаков выполнен: создано 6 новых признаков.")

    def _extract_features_target(self) -> None:
        """Разделяет DataFrame на матрицу признаков и вектор меток."""
        self.target   = self.df['isFraud'].copy()
        self.features = self.df.drop(columns=['isFraud']).copy()

    # ── Статистика ────────────────────────────────────────────────────

    def compute_basic_statistics(self) -> Dict:
        """
        Вычисляет базовые статистики и сохраняет в self.stats.

        Ключи словаря:
            'describe'           — описательные статистики признаков.
            'correlation_matrix' — матрица корреляций.
            'class_distribution' — количество объектов каждого класса.

        Возвращает:
            Dict: словарь со статистиками.
        """
        self.stats['describe']           = self.features.describe().T
        self.stats['correlation_matrix'] = self.features.corr()
        self.stats['class_distribution'] = (
            self.target.value_counts().sort_index().to_frame(name='count')
        )
        return self.stats

    # ── Предобработка ─────────────────────────────────────────────────

    def preprocess(self, drop_duplicates: bool = True, drop_na: bool = True) -> None:
        """
        Выполняет предобработку данных:
            1. Удаление дублирующихся строк (если drop_duplicates=True).
            2. Удаление строк с пропущенными значениями (если drop_na=True).
            3. Масштабирование признаков через StandardScaler.

        Параметры:
            drop_duplicates (bool): Удалять ли дубликаты.
            drop_na (bool): Удалять ли строки с пропусками.

        После выполнения self.scaled_features содержит масштабированные признаки.
        """
        df_proc = pd.concat([self.features, self.target], axis=1)

        if drop_duplicates:
            before = len(df_proc)
            df_proc.drop_duplicates(inplace=True)
            df_proc.reset_index(drop=True, inplace=True)
            print(f"Удалено дубликатов: {before - len(df_proc)}")

        if drop_na:
            before = len(df_proc)
            df_proc.dropna(inplace=True)
            df_proc.reset_index(drop=True, inplace=True)
            print(f"Удалено строк с пропусками: {before - len(df_proc)}")

        self.target   = df_proc['isFraud'].copy()
        self.features = df_proc.drop(columns=['isFraud']).copy()

        scaler = StandardScaler()
        self.scaled_features = pd.DataFrame(
            scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index,
        )
        print(f"Предобработка завершена. Итого: {len(self.features):,} строк.")

    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Параметры:
            test_size (float): Доля тестовых данных (по умолчанию 0.2).
            random_state (int): Seed для воспроизводимости.
            stratify (bool): Сохранять ли соотношение классов в обеих выборках.

        Выбрасывает:
            RuntimeError: если preprocess() не был вызван.
        """
        if self.scaled_features is None:
            raise RuntimeError("Сначала выполните preprocess().")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_features,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.target if stratify else None,
        )
        train_pct = int((1 - test_size) * 100)
        test_pct  = int(test_size * 100)
        print(f"Данные разделены (стратифицировано, {train_pct}/{test_pct}):")
        print(f"  Обучение:    {len(self.X_train):>10,} строк  "
              f"(мошенников: {self.y_train.sum():,}, {self.y_train.mean()*100:.3f}%)")
        print(f"  Тестирование:{len(self.X_test):>10,} строк  "
              f"(мошенников: {self.y_test.sum():,}, {self.y_test.mean()*100:.3f}%)")

    def balance_classes(
        self,
        sampler: str = 'SMOTE',
        sampling_strategy: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """
        Балансирует классы в обучающей выборке.

        Параметры:
            sampler (str): Метод балансировки:
                - 'SMOTE': генерирует синтетические примеры миноритарного класса.
                - 'undersampling': случайно удаляет примеры мажоритарного класса.
            sampling_strategy (float): Целевое соотношение меньшинства к большинству.
            random_state (int): Seed для воспроизводимости.

        Выбрасывает:
            RuntimeError: если split_data() не был вызван.
            ValueError: если передан неизвестный метод.
        """
        if self.X_train is None:
            raise RuntimeError("Сначала выполните split_data().")

        print(f"\nДо балансировки ({sampler}):")
        print(f"  Нормальных:    {(self.y_train == 0).sum():,}")
        print(f"  Мошеннических: {(self.y_train == 1).sum():,}")

        if sampler == 'SMOTE':
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif sampler == 'undersampling':
            resampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        else:
            raise ValueError("Доступные методы: 'SMOTE', 'undersampling'.")

        self.X_train, self.y_train = resampler.fit_resample(self.X_train, self.y_train)

        counts = pd.Series(self.y_train).value_counts().sort_index()
        print(f"\nПосле балансировки ({sampler}):")
        print(f"  Нормальных:    {counts[0]:,}")
        print(f"  Мошеннических: {counts[1]:,}  ({counts[1]/len(self.y_train)*100:.2f}%)")

    def get_training_data(self) -> Tuple[DataFrame, Series]:
        """Возвращает обучающие данные (X_train, y_train)."""
        return self.X_train, self.y_train

    def get_testing_data(self) -> Tuple[DataFrame, Series]:
        """Возвращает тестовые данные (X_test, y_test)."""
        return self.X_test, self.y_test

    def get_preprocessed_data(self) -> Tuple[DataFrame, Series]:
        """
        Возвращает масштабированные признаки и метки.

        Выбрасывает:
            RuntimeError: если preprocess() не был вызван.
        """
        if self.scaled_features is None:
            raise RuntimeError("Сначала выполните preprocess().")
        return self.scaled_features, self.target

    # ── Визуализация ──────────────────────────────────────────────────

    def visualize_class_distribution(
        self,
        figsize: Tuple[int, int] = (8, 8),
        title: str = "Распределение транзакций по классам",
        colors: Optional[List[str]] = None,
        autopct: str = "%1.2f%%",
        startangle: int = 90,
    ) -> None:
        """
        Строит и сохраняет круговую диаграмму распределения классов.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры в дюймах.
            title (str): Заголовок диаграммы.
            colors (Optional[List[str]]): Цвета секторов.
            autopct (str): Формат отображения процентов.
            startangle (int): Угол начала первой секции.
        """
        if 'class_distribution' not in self.stats:
            self.compute_basic_statistics()

        sizes  = self.stats['class_distribution']['count'].tolist()
        labels = ['Нормальная (0)', 'Мошенническая (1)']
        colors = colors or ['#2ecc71', '#e74c3c']

        plt.figure(figsize=figsize)
        plt.pie(
            sizes, labels=labels, colors=colors,
            autopct=autopct, startangle=startangle,
            explode=(0, 0.08),
            textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
        )
        plt.title(title, fontsize=14, pad=20)
        plt.axis('equal')
        self._save('class_distribution.png')

    def visualize_distributions(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """
        Строит гистограммы распределений числовых признаков и сохраняет в файл.

        Бинарные флаги (orig_zeroed, suspicious_type) исключены как неинформативные
        для гистограммы.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры в дюймах.
        """
        binary_cols  = {'orig_zeroed', 'suspicious_type'}
        cols_to_plot = [c for c in self.features.columns if c not in binary_cols]
        n_cols       = 3
        n_rows       = (len(cols_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(cols_to_plot):
            cap  = self.features[col].quantile(0.99)
            data = self.features[col].clip(upper=cap)
            axes[i].hist(data, bins=40, edgecolor='black', color='#3498db', alpha=0.85)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('Значение')
            axes[i].set_ylabel('Частота')

        for j in range(len(cols_to_plot), len(axes)):
            axes[j].axis('off')

        plt.suptitle('Распределения числовых признаков', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save('distributions.png')

    def visualize_scatter_matrix(
        self,
        with_target: bool = True,
        figsize: Tuple[int, int] = (14, 12),
        sample_size: int = 3000,
    ) -> None:
        """
        Строит матрицу рассеивания для числовых признаков и сохраняет в файл.

        Поскольку датасет содержит >6 млн строк, используется случайная
        стратифицированная подвыборка: поровну нормальных и мошеннических транзакций.
        Значения обрезаются по 99-му перцентилю для читаемости осей.

        Параметры:
            with_target (bool): Раскрашивать ли точки по классу isFraud.
            figsize (Tuple[int, int]): Размер фигуры в дюймах.
            sample_size (int): Суммарное число строк в подвыборке.
        """
        # Исключаем бинарные флаги и кодированный тип — малоинформативны на scatter
        excluded     = {'orig_zeroed', 'suspicious_type', 'type_encoded'}
        selected     = [c for c in self.features.columns if c not in excluded][:5]

        df_plot              = self.features[selected].copy()
        df_plot['isFraud']   = self.target.values

        # Стратифицированная подвыборка 50/50
        n_each    = sample_size // 2
        fraud_s   = df_plot[df_plot['isFraud'] == 1].sample(
            min(n_each, (df_plot['isFraud'] == 1).sum()), random_state=42
        )
        normal_s  = df_plot[df_plot['isFraud'] == 0].sample(
            min(n_each, (df_plot['isFraud'] == 0).sum()), random_state=42
        )
        df_sample = pd.concat([fraud_s, normal_s]).sample(frac=1, random_state=42)

        # Обрезка выбросов по 99-му перцентилю
        for col in selected:
            cap = df_sample[col].quantile(0.99)
            df_sample[col] = df_sample[col].clip(upper=cap)

        if with_target:
            df_sample['Класс'] = df_sample['isFraud'].map(
                {0: 'Нормальная', 1: 'Мошенническая'}
            )
            pair_grid = sns.pairplot(
                df_sample[selected + ['Класс']],
                hue='Класс',
                palette={'Нормальная': '#2ecc71', 'Мошенническая': '#e74c3c'},
                diag_kind='hist',
                plot_kws={'alpha': 0.4, 's': 15},
                height=figsize[0] / len(selected),
            )
            pair_grid.fig.suptitle(
                'Матрица рассеивания признаков (раскраска по классу)',
                y=1.01, fontsize=14, fontweight='bold',
            )
            pair_grid.fig.set_size_inches(figsize)
        else:
            from pandas.plotting import scatter_matrix as pd_scatter_matrix
            fig, _ = plt.subplots(figsize=figsize)
            plt.close(fig)
            pd_scatter_matrix(
                df_sample[selected], figsize=figsize,
                diagonal='hist', alpha=0.4, color='#3498db',
            )
            plt.suptitle(
                'Матрица рассеивания признаков', y=1.01,
                fontsize=14, fontweight='bold',
            )

        self._save('scatter_matrix.png')
        print(f"  (подвыборка: {len(df_sample)} строк из {len(self.features):,})")

    def visualize_fraud_by_type(self) -> None:
        """
        Строит столбчатую диаграмму доли мошенничества по типу операции.

        Для корректного отображения читает столбец 'type' из исходного CSV,
        поскольку он был удалён в ходе инжиниринга признаков.
        """
        df_raw     = pd.read_csv(self.csv_path, usecols=['type', 'isFraud'])
        fraud_rate = df_raw.groupby('type')['isFraud'].mean() * 100

        plt.figure(figsize=(9, 5))
        bars = plt.bar(
            fraud_rate.index, fraud_rate.values,
            color='#e74c3c', edgecolor='black', alpha=0.85,
        )
        for bar, val in zip(bars, fraud_rate.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}%', ha='center', va='bottom', fontsize=10,
            )
        plt.title(
            'Доля мошеннических транзакций по типу операции (%)',
            fontweight='bold', fontsize=13,
        )
        plt.ylabel('Доля мошенничества (%)')
        plt.xlabel('Тип операции')
        plt.tight_layout()
        self._save('fraud_by_type.png')

    def visualize_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 9)) -> None:
        """
        Строит тепловую карту корреляций между признаками и сохраняет в файл.

        Параметры:
            figsize (Tuple[int, int]): Размер фигуры в дюймах.
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.features.corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.7},
        )
        plt.title('Матрица корреляции признаков', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save('correlation_heatmap.png')

    # ── Вспомогательный метод ─────────────────────────────────────────

    def _save(self, filename: str) -> None:
        """Сохраняет текущую фигуру matplotlib в папку PLOTS_DIR и закрывает её."""
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сохранён: {path}")


# ── Точка входа ───────────────────────────────────────────────────────

if __name__ == '__main__':
    manager = DatasetManager()

    stats = manager.compute_basic_statistics()
    print("\nОписание признаков:")
    print(stats['describe'].to_string())
    print("\nРаспределение по классам:")
    print(stats['class_distribution'])

    manager.visualize_class_distribution()
    manager.visualize_fraud_by_type()
    manager.visualize_distributions()
    manager.visualize_scatter_matrix(with_target=True)
    manager.visualize_correlation_heatmap()

    manager.preprocess()
    manager.split_data(test_size=0.2, stratify=True)
    manager.balance_classes(sampler='SMOTE')

    X_train, y_train = manager.get_training_data()
    X_test,  y_test  = manager.get_testing_data()
    print("\nПервые строки предобработанной обучающей выборки:")
    print(X_train.head(10).to_string())
    print(f"\nГотово: X_train {X_train.shape}, X_test {X_test.shape}")
