import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_preprocessing import SORTED_DATA_DIR

def train():
    print("=== Обучение модели ===")

    # Загрузка предобработанных данных
    X_train = pd.read_csv(f"{SORTED_DATA_DIR}X_train.csv")
    y_train = pd.read_csv(f"{SORTED_DATA_DIR}y_train.csv").values.ravel()

    # Отладочная информация
    print("Столбцы в X_train:", X_train.columns)

    # Разделение данных на числовые и категориальные признаки
    numeric_features = ['Количество комнат', 'Площадь', 'Этаж', 'Ремонт', 'Мебель']
    categorical_features = ['Район', 'Тип дома' ]

    # Создание преобразователя
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Создание pipeline с предобработкой и моделью линейной регрессии
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Обучение модели
    model.fit(X_train, y_train)

    print("Модель успешно обучена.")
    return model, preprocessor