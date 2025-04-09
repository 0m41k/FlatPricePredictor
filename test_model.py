import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import SORTED_DATA_DIR
from sklearn.feature_selection import f_classif


def analyze_feature_importance(data):
    print("\n=== Анализ значимости факторов ===")

    # Разделение данных на числовые и категориальные признаки
    numeric_features = ['Количество комнат', 'Площадь', 'Этаж']
    categorical_features = ['Район', 'Тип дома', 'Ремонт', 'Мебель']

    # Целевая переменная
    target = 'Расценки на аренду'
    X = data.drop(target, axis=1)
    y = data[target]

    # Корреляция для числовых признаков
    numeric_corr = data[numeric_features + [target]].corr()[target].drop(target)
    numeric_corr_df = numeric_corr.reset_index()
    numeric_corr_df.columns = ['Признак', 'Значение']

    # ANOVA для категориальных признаков
    anova_scores = []
    n = len(data)  # Количество наблюдений
    for feature in categorical_features:
        groups = [y[data[feature] == category] for category in data[feature].unique()]
        f_stat, _ = f_classif(pd.get_dummies(data[feature]), y)
        k = len(data[feature].unique())  # Количество групп (категорий)
        r_squared = f_stat[0] / (f_stat[0] + n - k)  # Преобразование F-статистики в R²
        correlation = r_squared ** 0.5  # Преобразование R² в корреляцию
        anova_scores.append((feature, correlation))

    anova_df = pd.DataFrame(anova_scores, columns=['Признак', 'Значение'])

    # Объединение результатов
    importance_df = pd.concat([numeric_corr_df, anova_df], ignore_index=True)

    # Определение значимости
    importance_df['Значимость'] = importance_df.apply(
        lambda row: 'Значимый' if abs(row['Значение']) > 0.15 else 'Незначимый',
        axis=1
    )

    # Вывод таблицы значимости
    print("\nТаблица значимости факторов:")
    print(importance_df[['Признак', 'Значение', 'Значимость']])


def test(model):
    print("\n=== Тестирование модели ===")

    # Загрузка тестовых данных
    X_test = pd.read_csv(f"{SORTED_DATA_DIR}X_test.csv")
    y_test = pd.read_csv(f"{SORTED_DATA_DIR}y_test.csv").values.ravel()

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)

    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Сравнение реальных и предсказанных значений
    comparison = pd.DataFrame({
        'Реальная цена': y_test,
        'Предсказанная цена': y_pred
    })

    # Форматируем числа для удобства чтения
    comparison['Реальная цена'] = comparison['Реальная цена'].apply(lambda x: f"{x:,.2f}")
    comparison['Предсказанная цена'] = comparison['Предсказанная цена'].apply(lambda x: f"{x:,.2f}")

    print("\nСравнение реальных и предсказанных цен (первые 10 строк):")
    print(comparison.head(10))

    # Анализ значимости факторов
    full_data = pd.read_csv("FlatPricePredictor/data/input_data.csv")  # Загружаем полные данные
    analyze_feature_importance(full_data)