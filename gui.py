from PyQt5.QtWidgets import QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox, QComboBox
import pandas as pd

class PricePredictionApp(QMainWindow):
    def __init__(self, model, unique_districts, unique_house_types):
        super().__init__()
        self.model = model
        self.unique_districts = unique_districts
        self.unique_house_types = unique_house_types
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Предсказание цены покупки жилья")
        self.setGeometry(100, 100, 400, 400)

        # Создаем виджеты
        layout = QVBoxLayout()

        # Район (выпадающий список)
        self.district_label = QLabel("Район:")
        self.district_input = QComboBox()
        self.district_input.addItems(self.unique_districts)

        # Количество комнат (текстовое поле)
        self.rooms_label = QLabel("Количество комнат:")
        self.rooms_input = QLineEdit()

        # Площадь (текстовое поле)
        self.area_label = QLabel("Площадь (кв.м):")
        self.area_input = QLineEdit()

        # Тип дома (выпадающий список)
        self.house_type_label = QLabel("Тип дома:")
        self.house_type_input = QComboBox()
        self.house_type_input.addItems(self.unique_house_types)

        # Этаж (текстовое поле)
        self.floor_label = QLabel("Этаж:")
        self.floor_input = QLineEdit()

        # Ремонт (выпадающий список: Нет - 2, Есть - 1)
        self.repair_label = QLabel("Ремонт:")
        self.repair_input = QComboBox()
        self.repair_input.addItems(["Нет", "Есть"])

        # Мебель (выпадающий список: Нет - 2, Есть - 1)
        self.furniture_label = QLabel("Мебель:")
        self.furniture_input = QComboBox()
        self.furniture_input.addItems(["Нет", "Есть"])

        # Кнопка предсказания
        self.predict_button = QPushButton("Предсказать")
        self.predict_button.clicked.connect(self.predict_price)

        # Добавляем виджеты в layout
        layout.addWidget(self.district_label)
        layout.addWidget(self.district_input)
        layout.addWidget(self.rooms_label)
        layout.addWidget(self.rooms_input)
        layout.addWidget(self.area_label)
        layout.addWidget(self.area_input)
        layout.addWidget(self.house_type_label)
        layout.addWidget(self.house_type_input)
        layout.addWidget(self.floor_label)
        layout.addWidget(self.floor_input)
        layout.addWidget(self.repair_label)
        layout.addWidget(self.repair_input)
        layout.addWidget(self.furniture_label)
        layout.addWidget(self.furniture_input)
        layout.addWidget(self.predict_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def predict_price(self):
        try:
            # Получаем данные из полей ввода
            district = self.district_input.currentText()
            rooms = int(self.rooms_input.text())
            area = float(self.area_input.text())
            house_type = self.house_type_input.currentText()
            floor = int(self.floor_input.text())

            # Получаем значения для ремонта и мебели
            repair = 1 if self.repair_input.currentText() == "Есть" else 2
            furniture = 1 if self.furniture_input.currentText() == "Есть" else 2

            # Создаем DataFrame из введенных данных
            input_data = pd.DataFrame({
                'Район': [district],
                'Количество комнат': [rooms],
                'Площадь': [area],
                'Тип дома': [house_type],
                'Этаж': [floor],
                'Ремонт': [repair],
                'Мебель': [furniture]
            })

            # Получаем предсказание
            prediction = self.model.predict(input_data)[0]

            # Форматируем предсказанную цену
            formatted_prediction = f"{prediction:,.0f}"

            # Выводим результат в диалоговом окне
            QMessageBox.information(self, "Результат", f"Расценка на аренду: {formatted_prediction} рублей")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")