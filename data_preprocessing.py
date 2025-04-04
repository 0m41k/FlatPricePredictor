import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = "FlatPricePredictor/data/input_data.csv"
SORTED_DATA_DIR = "FlatPricePredictor/data/sorted_data/"

def preprocess_and_sort_data():
    print("=== Предобработка и сортировка данных ===")

    # Создание директории, если она не существует
    os.makedirs(SORTED_DATA_DIR, exist_ok=True)

    # Чтение данных
    data = pd.read_csv(DATA_PATH)

    # Проверка наличия всех нужных столбцов
    required_columns = ['Район', 'Количество комнат', 'Площадь', 'Тип дома', 'Этаж', 'Ремонт', 'Мебель', 'Цена покупки']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"В данных отсутствуют следующие столбцы: {missing_columns}")

    # Разделение данных на числовые и категориальные признаки
    numeric_features = ['Количество комнат', 'Площадь', 'Этаж', 'Ремонт', 'Мебель']
    categorical_features = ['Район', 'Тип дома']

    # Создание преобразователя
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Разделение данных на признаки (X) и целевую переменную (y)
    X = data.drop('Цена покупки', axis=1)
    y = data['Цена покупки']

    # Разделение данных на обучающую и тестовую выборки (95% - обучение, 5% - тест)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Отладочная информация
    print("Столбцы в X_train:", X_train.columns)
    print("Столбцы в X_test:", X_test.columns)

    # Сохранение обработанных данных
    X_train.to_csv(f"{SORTED_DATA_DIR}X_train.csv", index=False)
    X_test.to_csv(f"{SORTED_DATA_DIR}X_test.csv", index=False)
    y_train.to_csv(f"{SORTED_DATA_DIR}y_train.csv", index=False)
    y_test.to_csv(f"{SORTED_DATA_DIR}y_test.csv", index=False)

    print("Данные успешно предобработаны и сохранены.")