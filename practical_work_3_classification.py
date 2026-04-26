"""
========================================================================
Практическая работа №3: «Классификации в анализе данных»
Дисциплина: Проектирование интеллектуальных систем

Предметная область: Обнаружение мошеннических финансовых транзакций
Датасет: PaySim — https://www.kaggle.com/datasets/ealaxi/paysim1

Структура:
    1. Загрузка и предобработка данных
    2. Разведочный анализ (EDA) + визуализации
    3. Подготовка признаков
    4. Бинарная классификация (основное задание)
    5. Несбалансированная классификация (доп. задание 1)
    6. Ансамбль моделей (доп. задание 2)
    7. Предсказательная аналитика (доп. задание 3)
    8. Итоговое сравнение
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Сохранение графиков в файлы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from time import time

warnings.filterwarnings('ignore')

# Sklearn — предобработка и разбивка
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Базовые классификаторы
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Ансамблевые модели
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

# Работа с несбалансированными данными
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    imblearn_available = True
except ImportError:
    imblearn_available = False
    print("imbalanced-learn не установлен. Установите: pip install imbalanced-learn")

# XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Метрики
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

# Настройки графиков
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Путь к CSV-файлу датасета
DATA_PATH = 'PS_20174392719_1491204439457_log.csv'

# Хранилище результатов всех моделей
results = []


# ════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ════════════════════════════════════════════════════════════════════

def section(title):
    """Печатает заголовок раздела."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """
    Обучает модель, предсказывает на тесте, считает метрики.
    Результат добавляется в глобальный список results.
    Возвращает: (обученная_модель, y_pred, y_prob)
    """
    t0 = time()
    model.fit(X_tr, y_tr)
    train_time = round(time() - t0, 2)

    y_pred = model.predict(X_te)

    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        roc_auc = roc_auc_score(y_te, y_prob)
    except Exception:
        y_prob = None
        roc_auc = None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    results.append({
        'Модель':             name,
        'Accuracy':           acc,
        'Precision':          prec,
        'Recall':             rec,
        'F1-score':           f1,
        'ROC-AUC':            roc_auc,
        'Время обучения (с)': train_time,
    })

    print(f"\n  Модель: {name}")
    print(f"    Accuracy:        {acc:.4f}")
    print(f"    Precision:       {prec:.4f}")
    print(f"    Recall:          {rec:.4f}")
    print(f"    F1-score:        {f1:.4f}")
    if roc_auc is not None:
        print(f"    ROC-AUC:         {roc_auc:.4f}")
    print(f"    Время обучения:  {train_time} с")
    print()
    print(classification_report(
        y_te, y_pred,
        target_names=['Нормальная', 'Мошенническая'],
        zero_division=0,
    ))

    return model, y_pred, y_prob


def save_confusion_matrix(y_true, y_pred, title, filename):
    """Строит и сохраняет матрицу ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Нормальная', 'Мошенническая'],
                yticklabels=['Нормальная', 'Мошенническая'])
    plt.title(f'Матрица ошибок — {title}', fontweight='bold')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Матрица ошибок сохранена: {path}")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 1: ЗАГРУЗКА ДАННЫХ
# ════════════════════════════════════════════════════════════════════
section("1. ЗАГРУЗКА ДАННЫХ")

print(f"  Читаем файл: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"  Загружено: {df.shape[0]:,} строк, {df.shape[1]} столбцов")

print("\n  Первые 3 строки:")
print(df.head(3).to_string())

print("\n  Описание столбцов:")
col_desc = {
    'step':           'Единица времени (1 час)',
    'type':           'Тип операции: CASH-IN/OUT, DEBIT, PAYMENT, TRANSFER',
    'amount':         'Сумма транзакции',
    'nameOrig':       'ID отправителя',
    'oldbalanceOrg':  'Баланс отправителя ДО',
    'newbalanceOrig': 'Баланс отправителя ПОСЛЕ',
    'nameDest':       'ID получателя',
    'oldbalanceDest': 'Баланс получателя ДО',
    'newbalanceDest': 'Баланс получателя ПОСЛЕ',
    'isFraud':        'ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (1 = мошенничество)',
    'isFlaggedFraud': 'Флаг системы безопасности',
}
for col, desc in col_desc.items():
    print(f"    {col:20s} — {desc}")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 2: ПРЕДОБРАБОТКА И EDA
# ════════════════════════════════════════════════════════════════════
section("2. ПРЕДОБРАБОТКА И РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)")

# Пропуски
print("  Пропущенные значения:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("    Пропущенных значений нет")
else:
    print(missing[missing > 0])

# Дубли
dups = df.duplicated().sum()
print(f"\n  Дублирующихся строк: {dups}")
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"    Дубли удалены. Новый размер: {df.shape}")
else:
    print("    Дублей нет")

# Дисбаланс классов
fraud_cnt = df['isFraud'].value_counts()
fraud_pct = df['isFraud'].value_counts(normalize=True) * 100
print(f"\n  Распределение классов:")
print(f"    Нормальные (0):      {fraud_cnt[0]:>10,}  ({fraud_pct[0]:.4f}%)")
print(f"    Мошеннические (1):   {fraud_cnt[1]:>10,}  ({fraud_pct[1]:.4f}%)")
print(f"    Соотношение дисбаланса: {fraud_cnt[0]//fraud_cnt[1]}:1")

# Мошенничество по типам
print("\n  Мошенничество по типам транзакций:")
type_fraud = (df.groupby('type')['isFraud']
              .agg(Мошеннических='sum', Всего='count', Доля='mean')
              .assign(Доля=lambda x: (x['Доля'] * 100).round(3)))
print(type_fraud.to_string())

# Базовая статистика
print("\n  Статистика числовых признаков:")
print(df[['amount', 'oldbalanceOrg', 'newbalanceOrig',
          'oldbalanceDest', 'newbalanceDest']].describe().round(2).to_string())

# Графики EDA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Нормальные', 'Мошеннические'], fraud_cnt.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.85)
axes[0].set_title('Количество транзакций по классам', fontweight='bold')
axes[0].set_ylabel('Количество')
for i, v in enumerate(fraud_cnt.values):
    axes[0].text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(fraud_cnt.values,
            labels=[f'Нормальные\n{fraud_pct[0]:.2f}%',
                    f'Мошеннические\n{fraud_pct[1]:.2f}%'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.2f%%', startangle=90, explode=(0, 0.1))
axes[1].set_title('Доля классов', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution.png'), dpi=150)
plt.close()

fraud_rate = df.groupby('type')['isFraud'].mean() * 100
plt.figure(figsize=(9, 5))
plt.bar(fraud_rate.index, fraud_rate.values,
        color='#e74c3c', edgecolor='black', alpha=0.85)
plt.title('Доля мошеннических транзакций по типу (%)', fontweight='bold')
plt.ylabel('Доля мошенничества (%)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_fraud_by_type.png'), dpi=150)
plt.close()

num_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'isFraud']
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Корреляционная матрица', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_correlation.png'), dpi=150)
plt.close()

print("\n  Графики EDA сохранены в папку plots/")
print("  Вывод: мошенничество встречается ТОЛЬКО в CASH_OUT и TRANSFER.")
print("  Вывод: данные сильно несбалансированы — нужны специальные метрики.")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 3: ИНЖИНИРИНГ ПРИЗНАКОВ И ПОДГОТОВКА ДАННЫХ
# ════════════════════════════════════════════════════════════════════
section("3. ИНЖИНИРИНГ ПРИЗНАКОВ И ПОДГОТОВКА ДАННЫХ")

df_model = df.copy()

# Кодирование категориального признака
le = LabelEncoder()
df_model['type_encoded'] = le.fit_transform(df_model['type'])
print("  Кодирование типов транзакций (LabelEncoder):")
for i, t in enumerate(le.classes_):
    print(f"    {t} -> {i}")

# Новые признаки
df_model['balance_diff_orig']    = df_model['newbalanceOrig'] - df_model['oldbalanceOrg']
df_model['balance_diff_dest']    = df_model['newbalanceDest'] - df_model['oldbalanceDest']
df_model['orig_zeroed']          = (
    (df_model['oldbalanceOrg'] > 0) & (df_model['newbalanceOrig'] == 0)
).astype(int)
df_model['suspicious_type']      = df_model['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
df_model['amount_balance_error'] = abs(df_model['amount'] + df_model['balance_diff_orig'])

print("\n  Новые признаки (Feature Engineering):")
print("    balance_diff_orig    — изменение баланса отправителя")
print("    balance_diff_dest    — изменение баланса получателя")
print("    orig_zeroed          — баланс отправителя обнулился (0/1)")
print("    suspicious_type      — подозрительный тип операции (0/1)")
print("    amount_balance_error — несоответствие суммы и баланса")

FEATURE_COLS = [
    'step', 'type_encoded', 'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_diff_orig', 'balance_diff_dest',
    'orig_zeroed', 'suspicious_type', 'amount_balance_error',
]

X = df_model[FEATURE_COLS]
y = df_model['isFraud']

print(f"\n  X: {X.shape}  |  y: {y.shape}")

# Стратифицированное разбиение 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Разбивка (стратифицированная 80/20):")
print(f"    Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"    Доля мошенничества — train: {y_train.mean()*100:.4f}%"
      f"  |  test: {y_test.mean()*100:.4f}%")

# Масштабирование для ЛР и НБ
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("  Данные масштабированы (StandardScaler) для ЛР и НБ")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 4: БИНАРНАЯ КЛАССИФИКАЦИЯ — БАЗОВЫЕ МОДЕЛИ
# ════════════════════════════════════════════════════════════════════
section("4. БИНАРНАЯ КЛАССИФИКАЦИЯ — БАЗОВЫЕ МОДЕЛИ")

print("""
  Метрики оценки:
    Accuracy  — доля верных предсказаний (ненадёжна при дисбалансе!)
    Precision — доля истинных мошенников среди помеченных моделью
    Recall    — доля найденных мошенников из всех реальных
    F1-score  — гармоническое среднее Precision и Recall
    ROC-AUC   — площадь под ROC-кривой (1.0 = идеальная модель)
""")

# 4.1 Логистическая регрессия (линейный, энергичный классификатор)
print("  --- 4.1 Логистическая регрессия (линейная модель) ---")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model, lr_pred, lr_prob = evaluate_model(
    'Логистическая регрессия', lr,
    X_train_sc, y_train, X_test_sc, y_test
)
save_confusion_matrix(y_test, lr_pred, 'Логистическая регрессия', '04_cm_lr.png')

# 4.2 Дерево решений (нелинейный, энергичный классификатор)
print("  --- 4.2 Дерево решений (нелинейная модель) ---")
dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
dt_model, dt_pred, dt_prob = evaluate_model(
    'Дерево решений', dt,
    X_train, y_train, X_test, y_test
)
save_confusion_matrix(y_test, dt_pred, 'Дерево решений', '05_cm_dt.png')

# 4.3 Наивный Байес (вероятностный классификатор)
print("  --- 4.3 Наивный Байес (вероятностная модель) ---")
nb = GaussianNB()
nb_model, nb_pred, nb_prob = evaluate_model(
    'Наивный Байес', nb,
    X_train_sc, y_train, X_test_sc, y_test
)

# Важность признаков по дереву решений
feat_imp = pd.Series(
    dt_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("  Важность признаков (Дерево решений, Gini Importance):")
for feat, imp in feat_imp.items():
    bar = '#' * int(imp * 60)
    print(f"    {feat:25s} {imp:.4f}  {bar}")

plt.figure(figsize=(10, 6))
feat_imp.sort_values().plot(kind='barh', color='#3498db', edgecolor='black', alpha=0.85)
plt.title('Важность признаков — Дерево решений', fontweight='bold')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_feature_importance.png'), dpi=150)
plt.close()
print(f"\n  Вывод: ключевые признаки — amount, balance_diff, orig_zeroed.")
print("  Мошенники обнуляют счёт жертвы и переводят крупные суммы.")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 5: НЕСБАЛАНСИРОВАННАЯ КЛАССИФИКАЦИЯ (ДОП. ЗАДАНИЕ 1)
# ════════════════════════════════════════════════════════════════════
section("5. НЕСБАЛАНСИРОВАННАЯ КЛАССИФИКАЦИЯ (Доп. задание 1)")

# Демонстрация проблемы
print("  Демонстрация проблемы дисбаланса классов:")
naive_pred = np.zeros(len(y_test), dtype=int)
print(f"    Наивный классификатор (всегда предсказывает 0 = нормально):")
print(f"      Accuracy:  {accuracy_score(y_test, naive_pred):.4f}  <- выглядит хорошо!")
print(f"      Recall:    {recall_score(y_test, naive_pred):.4f}    <- находит 0% мошенников!")
print(f"      F1-score:  {f1_score(y_test, naive_pred):.4f}    <- абсолютно бесполезен")
print("  ВЫВОД: Accuracy = ~99.87% при том, что ни один мошенник не найден.")
print("  Именно поэтому нужны F1, Recall, ROC-AUC и техники балансировки!")

if imblearn_available:
    # SMOTE
    print(f"\n  --- SMOTE (генерация синтетических примеров меньшинства) ---")
    print(f"    До SMOTE: {X_train.shape[0]:,} записей, "
          f"мошенников {y_train.sum():,} ({y_train.mean()*100:.3f}%)")

    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"    После SMOTE: {X_train_smote.shape[0]:,} записей, "
          f"мошенников {y_train_smote.sum():,} ({y_train_smote.mean()*100:.2f}%)")

    dt_smote = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt_smote_model, dt_smote_pred, dt_smote_prob = evaluate_model(
        'Дерево решений + SMOTE', dt_smote,
        X_train_smote, y_train_smote, X_test, y_test
    )
    save_confusion_matrix(y_test, dt_smote_pred,
                          'Дерево решений + SMOTE', '07_cm_smote.png')

    # RandomUnderSampler
    print("  --- RandomUnderSampler (недовыборка мажоритарного класса) ---")
    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    print(f"    После UnderSampling: {X_train_rus.shape[0]:,} записей")

    dt_rus = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt_rus_model, dt_rus_pred, dt_rus_prob = evaluate_model(
        'Дерево решений + UnderSampling', dt_rus,
        X_train_rus, y_train_rus, X_test, y_test
    )

    # Сравнение методов балансировки
    bal_names = ['Дерево решений', 'Дерево решений + SMOTE',
                 'Дерево решений + UnderSampling']
    bal_df = pd.DataFrame([r for r in results if r['Модель'] in bal_names])
    print("\n  Сравнение методов балансировки:")
    print(bal_df[['Модель', 'Precision', 'Recall', 'F1-score']].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    short = [n.replace('Дерево решений', 'ДР') for n in bal_df['Модель']]
    for ax, metric, color in zip(axes,
                                  ['Recall', 'Precision', 'F1-score'],
                                  ['#e74c3c', '#3498db', '#2ecc71']):
        ax.bar(short, bal_df[metric], color=color, edgecolor='black', alpha=0.85)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle('Сравнение методов балансировки', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_balance_comparison.png'), dpi=150)
    plt.close()
    print("\n  Вывод: SMOTE улучшает Recall — находим больше мошенников.")
    print("  UnderSampling быстрее, но теряет часть полезных данных.")
else:
    print("\n  imbalanced-learn не установлен, раздел 5 пропущен.")
    print("  Установите: pip install imbalanced-learn")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 6: АНСАМБЛЬ МОДЕЛЕЙ (ДОП. ЗАДАНИЕ 2)
# ════════════════════════════════════════════════════════════════════
section("6. АНСАМБЛЬ МОДЕЛЕЙ (Доп. задание 2)")

print("""
  Виды ансамблей:
    Бэггинг    — однородный ансамбль, деревья обучаются на разных подвыборках
    Бустинг    — последовательный ансамбль, каждая модель исправляет ошибки прошлой
    Голосование— параллельный ансамбль из разных моделей, усреднение предсказаний
    Стэкинг    — мета-классификатор обучается на предсказаниях базовых моделей
""")

# 6.1 Однородный ансамбль — Random Forest (бэггинг)
print("  --- 6.1 Random Forest (однородный ансамбль, бэггинг) ---")
print("  100 деревьев, каждое обучается на случайной подвыборке строк и признаков")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model, rf_pred, rf_prob = evaluate_model(
    'Random Forest (бэггинг)', rf,
    X_train, y_train, X_test, y_test
)
save_confusion_matrix(y_test, rf_pred, 'Random Forest', '09_cm_rf.png')

# 6.2 Последовательный ансамбль — Бустинг
print("  --- 6.2 Бустинг (последовательный ансамбль) ---")
if xgb_available:
    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
    gb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pw, random_state=42,
        n_jobs=-1, eval_metric='logloss', verbosity=0
    )
    gb_name = 'XGBoost (бустинг)'
    print("  Используется XGBoost с учётом дисбаланса через scale_pos_weight")
else:
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb_name = 'Gradient Boosting (бустинг)'
    print("  Используется sklearn GradientBoostingClassifier")

gb_model, gb_pred, gb_prob = evaluate_model(
    gb_name, gb,
    X_train, y_train, X_test, y_test
)
save_confusion_matrix(y_test, gb_pred, gb_name, '10_cm_boost.png')

# 6.3 Параллельный ансамбль — Voting (мягкое голосование)
print("  --- 6.3 VotingClassifier (параллельный ансамбль, soft voting) ---")
print("  Состав: Логистическая регрессия + Дерево решений + Наивный Байес")
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')),
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')),
        ('nb', GaussianNB()),
    ],
    voting='soft', n_jobs=-1
)
voting_model, voting_pred, voting_prob = evaluate_model(
    'Voting (ЛР + ДР + НБ)', voting,
    X_train_sc, y_train, X_test_sc, y_test
)

# 6.4 Гетерогенный ансамбль — Stacking
print("  --- 6.4 StackingClassifier (гетерогенный ансамбль) ---")
print("  Базовые модели: ЛР + ДР + НБ")
print("  Мета-классификатор обучается на их предсказаниях: Логистическая регрессия")
stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')),
        ('dt', DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced')),
        ('nb', GaussianNB()),
    ],
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=3, n_jobs=-1
)
stacking_model, stacking_pred, stacking_prob = evaluate_model(
    'Stacking (мета-ЛР)', stacking,
    X_train_sc, y_train, X_test_sc, y_test
)

# ROC-кривые
plt.figure(figsize=(12, 7))
roc_models = [
    ('Логистическая регрессия', lr_prob),
    ('Дерево решений',          dt_prob),
    (gb_name,                   gb_prob),
    ('Random Forest',           rf_prob),
    ('Voting',                  voting_prob),
    ('Stacking',                stacking_prob),
]
if imblearn_available:
    roc_models.insert(2, ('ДР + SMOTE', dt_smote_prob))

colors_roc = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#e67e22']
for (name, prob), color in zip(roc_models, colors_roc):
    if prob is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Случайный классификатор')
plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC-кривые всех классификаторов', fontweight='bold', fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_roc_curves.png'), dpi=150)
plt.close()
print(f"\n  ROC-кривые сохранены: {PLOTS_DIR}/11_roc_curves.png")
print("  Вывод: бустинг и Random Forest показывают наивысший ROC-AUC.")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 7: ПРЕДСКАЗАТЕЛЬНАЯ АНАЛИТИКА (ДОП. ЗАДАНИЕ 3)
# ════════════════════════════════════════════════════════════════════
section("7. ПРЕДСКАЗАТЕЛЬНАЯ АНАЛИТИКА — Многоклассовая (Доп. задание 3)")

print("""
  Задача: предсказать КАТЕГОРИЮ РИСКА транзакции (3 класса):
    0 — нормальная транзакция
    1 — мошенничество малой суммы (ниже медианы мошеннических транзакций)
    2 — мошенничество крупной суммы (выше медианы)

  Отличие от регрессии: предсказываем категорию риска, а не точную сумму.
  Это позволяет банку приоритизировать проверку подозрительных транзакций.
""")

fraud_median = df_model[df_model['isFraud'] == 1]['amount'].median()
print(f"  Медиана суммы мошеннических транзакций: {fraud_median:,.0f}")


def assign_risk(row):
    if row['isFraud'] == 0:
        return 0
    return 1 if row['amount'] <= fraud_median else 2


df_model['risk_category'] = df_model.apply(assign_risk, axis=1)

cat_names = {
    0: 'Нормальная транзакция',
    1: f'Мошенничество: малая сумма (<= {fraud_median:,.0f})',
    2: f'Мошенничество: крупная сумма (> {fraud_median:,.0f})',
}
print("\n  Категории риска:")
for k, v in cat_names.items():
    cnt = (df_model['risk_category'] == k).sum()
    print(f"    Класс {k}: {v} — {cnt:,} записей ({cnt/len(df_model)*100:.3f}%)")

# Обучение многоклассового Random Forest
X_mc = df_model[FEATURE_COLS]
y_mc = df_model['risk_category']

X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(
    X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc
)

rf_mc = RandomForestClassifier(
    n_estimators=100, max_depth=12,
    class_weight='balanced', random_state=42, n_jobs=-1
)
print("\n  Обучение многоклассового Random Forest...")
t0 = time()
rf_mc.fit(X_tr_mc, y_tr_mc)
print(f"  Время обучения: {time()-t0:.1f} с")

y_pred_mc = rf_mc.predict(X_te_mc)
print("\n  Отчёт многоклассовой классификации (3 класса):")
print(classification_report(
    y_te_mc, y_pred_mc,
    target_names=['Нормальная', 'Мошенн. малая', 'Мошенн. крупная'],
    zero_division=0,
))

# Матрица ошибок
cm_mc = confusion_matrix(y_te_mc, y_pred_mc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Нормальная', 'Мош.\nмалая', 'Мош.\nкрупная'],
            yticklabels=['Нормальная', 'Мош.\nмалая', 'Мош.\nкрупная'])
plt.title('Матрица ошибок — Многоклассовая классификация риска', fontweight='bold')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_multiclass_cm.png'), dpi=150)
plt.close()
print(f"  График сохранён: {PLOTS_DIR}/12_multiclass_cm.png")

# Симуляция для новых транзакций
print("\n  Симуляция: предсказание категории риска для новых транзакций")
print("  " + "-" * 60)

new_tx = pd.DataFrame({
    'step':                [1,       200,     350,    10],
    'type_encoded':        [4,       1,       4,      2],
    'amount':              [5000,    850000,  18000,  200],
    'oldbalanceOrg':       [5000,    850000,  18000,  5000],
    'newbalanceOrig':      [0,       0,       0,      4800],
    'oldbalanceDest':      [100000,  0,       50000,  1000],
    'newbalanceDest':      [105000,  850000,  68000,  1200],
    'balance_diff_orig':   [-5000,  -850000, -18000,  -200],
    'balance_diff_dest':   [5000,    850000,  18000,   200],
    'orig_zeroed':         [1,       1,       1,       0],
    'suspicious_type':     [1,       1,       1,       0],
    'amount_balance_error':[0,       0,       0,       0],
})
tx_desc = [
    'TRANSFER 5,000    — счёт обнулился',
    'CASH_OUT 850,000  — счёт обнулился',
    'TRANSFER 18,000   — счёт обнулился',
    'PAYMENT  200      — обычная покупка',
]

preds_mc = rf_mc.predict(new_tx)
probs_mc = rf_mc.predict_proba(new_tx)

for i, (desc, pred, prob) in enumerate(zip(tx_desc, preds_mc, probs_mc)):
    flag = '[МОШЕННИЧЕСТВО]' if pred > 0 else '[НОРМА]'
    print(f"\n  Транзакция {i+1}: {desc}")
    print(f"    {flag} -> {cat_names[pred]}")
    print(f"    Вероятности: Норм.={prob[0]:.3f} | Мош.мал.={prob[1]:.3f} | Мош.кр.={prob[2]:.3f}")

print("\n  Вывод: модель корректно классифицирует тип риска транзакции.")
print("  Крупные мошенничества детектируются уверенней — они более выражены.")


# ════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 8: ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════
section("8. ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")

results_df = pd.DataFrame(results).sort_values('F1-score', ascending=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 130)
pd.set_option('display.float_format', '{:.4f}'.format)

print(results_df[[
    'Модель', 'Accuracy', 'Precision', 'Recall',
    'F1-score', 'ROC-AUC', 'Время обучения (с)'
]].to_string(index=False))

best = results_df.iloc[0]
print(f"\n  Лучшая модель по F1-score: {best['Модель']}")
print(f"    F1-score: {best['F1-score']:.4f}  |  ROC-AUC: {best['ROC-AUC']:.4f}")

# Итоговый график
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, metric in zip(axes.flatten(), ['F1-score', 'Recall', 'Precision', 'ROC-AUC']):
    data = results_df[results_df[metric].notnull()].sort_values(metric, ascending=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    bars = ax.barh(data['Модель'], data[metric],
                   color=colors, edgecolor='black', alpha=0.85)
    ax.set_title(metric, fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, data[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8, fontweight='bold')

plt.suptitle('Итоговое сравнение всех классификаторов',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '13_final_comparison.png'), dpi=150)
plt.close()
print(f"\n  Итоговый график сохранён: {PLOTS_DIR}/13_final_comparison.png")


# ════════════════════════════════════════════════════════════════════
#  ВЫВОДЫ
# ════════════════════════════════════════════════════════════════════
section("ВЫВОДЫ ПО ПРАКТИЧЕСКОЙ РАБОТЕ")

print("""
  1. ПРЕДМЕТНАЯ ОБЛАСТЬ
     Обнаружение мошеннических финансовых транзакций (Fraud Detection).
     Датасет PaySim: ~6.3 млн записей, 11 исходных признаков.

  2. ПРЕДОБРАБОТКА (Раздел 1-3)
     - Пропусков и дублей нет
     - Кодирование типа транзакции через LabelEncoder
     - Созданы 5 новых признаков: balance_diff, orig_zeroed, suspicious_type и др.
     - Стратифицированное разбиение 80/20 для корректной оценки

  3. БИНАРНАЯ КЛАССИФИКАЦИЯ (Раздел 4)
     Обучены 3 базовые модели: Логистическая регрессия, Дерево решений, Наивный Байес.
     Accuracy вводит в заблуждение при дисбалансе — нужны F1 и Recall.

  4. НЕСБАЛАНСИРОВАННАЯ КЛАССИФИКАЦИЯ (Раздел 5, Доп. задание 1)
     Дисбаланс ~773:1 делает Accuracy бесполезной метрикой.
     SMOTE улучшает Recall, UnderSampling быстрее, class_weight — хороший базовый метод.

  5. АНСАМБЛЬ МОДЕЛЕЙ (Раздел 6, Доп. задание 2)
     Реализованы все 4 вида ансамблей:
       - Бэггинг:     Random Forest (100 деревьев)
       - Бустинг:     XGBoost / Gradient Boosting
       - Голосование: VotingClassifier (ЛР + ДР + НБ)
       - Стэкинг:     StackingClassifier с мета-ЛР
     Бустинг и Random Forest дают наивысший ROC-AUC.

  6. ПРЕДСКАЗАТЕЛЬНАЯ АНАЛИТИКА (Раздел 7, Доп. задание 3)
     Многоклассовая классификация категорий риска (3 класса) позволяет
     не только обнаружить мошенничество, но и оценить его серьёзность.

  Все графики сохранены в папку: plots/
""")

print("=" * 70)
print("  Практическая работа №3 выполнена успешно!")
print("=" * 70)
