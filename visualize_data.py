import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from data_preprocessing import DATA_PATH

def visualize():
    print("\n=== Визуализация данных ===")

    # Загрузка данных
    data = pd.read_csv(DATA_PATH)

    # Настройка стиля seaborn
    sns.set(style="whitegrid")

    # Создание подграфиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Визуализация данных о ценах на квартиры", fontsize=16)

    # График 1: Круговая диаграмма доли квартир по районам
    district_counts = data['Район'].value_counts()
    axes[0, 0].pie(district_counts, labels=district_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
    axes[0, 0].set_title("Доля квартир по районам")
    axes[0, 0].axis('equal')  # Круговая диаграмма будет кругом, а не эллипсом

    # График 2: Зависимость цены от площади с линией тренда
    sns.scatterplot(x='Площадь', y='Расценки на аренду', data=data, ax=axes[0, 1], color='orange')

    # Настройка графика
    axes[0, 1].set_title("Зависимость цены от площади")
    axes[0, 1].set_xlabel("Площадь (кв.м)")
    axes[0, 1].set_ylabel("Расценка на аренду")
    axes[0, 1].legend()

    # График 3: Зависимость цены от количества комнат
    sns.boxplot(x='Количество комнат', y='Расценки на аренду', data=data, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title("Зависимость цены от количества комнат")
    axes[1, 0].set_xlabel("Количество комнат")
    axes[1, 0].set_ylabel("Расценка на аренду")

    # График 4: Столбчатая диаграмма средней цены по типам домов
    avg_price_by_house_type = data.groupby('Тип дома')['Расценки на аренду'].mean().sort_values()
    sns.barplot(x=avg_price_by_house_type.index, y=avg_price_by_house_type.values, ax=axes[1, 1], palette='Blues_d')
    axes[1, 1].set_title("Средняя цена квартир по типам домов")
    axes[1, 1].set_xlabel("Тип дома")
    axes[1, 1].set_ylabel("Средняя расценка на аренду")
    axes[1, 1].tick_params(axis='x', rotation=45)  # Поворот подписей по оси X

    # Настройка макета
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()