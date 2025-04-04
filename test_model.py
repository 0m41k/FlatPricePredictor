import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import SORTED_DATA_DIR

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