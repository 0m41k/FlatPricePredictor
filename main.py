import sys
from PyQt5.QtWidgets import QApplication
from train_model import train
from test_model import test
from visualize_data import visualize
from gui import PricePredictionApp
from data_preprocessing import preprocess_and_sort_data
import pandas as pd

DATA_PATH = "FlatPricePredictor/data/input_data.csv"

def main():
    print("=== Запуск программы ===")
    # Предобработка данных
    preprocess_and_sort_data()

    # Загрузка уникальных значений для выпадающих списков
    data = pd.read_csv(DATA_PATH)
    unique_districts = data['Район'].unique().tolist()  # Убедитесь, что здесь "Район"
    unique_house_types = data['Тип дома'].unique().tolist()

    # Обучение модели
    model, preprocessor = train()  # Получаем модель и preprocessors

    # Тестирование модели в консоли
    test(model)  # Передаем только модель

    # Визуализация данных
    visualize()

    # Запуск графического интерфейса
    app = QApplication(sys.argv)
    window = PricePredictionApp(model, unique_districts, unique_house_types)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()