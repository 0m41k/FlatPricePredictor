import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = "FlatPricePredictor/data/input_data.csv"
SORTED_DATA_DIR = "FlatPricePredictor/data/sorted_data/"

def preprocess_and_sort_data():
    print("=== Предобработка и сортировка данных ===")

    # Создание директории, если она не существует
    try:
        os.makedirs(SORTED_DATA_DIR, exist_ok=True)
        print(f"Директория '{SORTED_DATA_DIR}' успешно создана.")
    except Exception as e:
        print(f"Ошибка при создании директории: {e}")
        return

    # Чтение данных
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Данные успешно загружены из файла '{DATA_PATH}'.")
    except Exception as e:
        print(f"Ошибка при чтении файла '{DATA_PATH}': {e}")
        return

    # Проверка наличия всех нужных столбцов
    required_columns = ['Район', 'Количество комнат', 'Площадь', 'Тип дома', 'Этаж', 'Цена покупки']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"В данных отсутствуют следующие столбцы: {missing_columns}")
        return

    # Разделение данных на числовые и категориальные признаки
    numeric_features = ['Количество комнат', 'Площадь', 'Этаж']
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

    # Разделение данных на обучающую и тестовую выборки (80% - обучение, 20% - тест)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Данные успешно разделены на обучающую и тестовую выборки.")
    except Exception as e:
        print(f"Ошибка при разделении данных: {e}")
        return

    # Сохранение обработанных данных
    try:
        X_train.to_csv(f"{SORTED_DATA_DIR}X_train.csv", index=False)
        X_test.to_csv(f"{SORTED_DATA_DIR}X_test.csv", index=False)
        y_train.to_csv(f"{SORTED_DATA_DIR}y_train.csv", index=False)
        y_test.to_csv(f"{SORTED_DATA_DIR}y_test.csv", index=False)
        print("Обработанные данные успешно сохранены.")
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")