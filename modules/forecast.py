import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objs as go


class SalesForecast:
    """Класс для прогнозирования месячного объёма продаж"""

    def __init__(self, data_path='data/Food_apps.csv'):
        """Загрузка данных и подготовка"""
        self.df = pd.read_csv(data_path)
        self.X = self.df[['avg_equipment_price', 'production_capacity',
                          'is_automation', 'energy_consumption', 'equipment_weight']]
        self.y = self.df['monthly_equipment_sales']

        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model = None
        self.is_trained = False

    def train(self):
        """Обучение модели линейной регрессии"""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

        # Расчёт метрик
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        self.r2_train = r2_score(self.y_train, y_pred_train)
        self.r2_test = r2_score(self.y_test, y_pred_test)
        self.mae = mean_absolute_error(self.y_test, y_pred_test)

        # Коэффициенты модели
        self.coefficients = {
            'intercept': self.model.intercept_,
            'avg_equipment_price': self.model.coef_[0],
            'production_capacity': self.model.coef_[1],
            'is_automation': self.model.coef_[2],
            'energy_consumption': self.model.coef_[3],
            'equipment_weight': self.model.coef_[4]
        }

        return self.r2_train, self.r2_test, self.mae

    def predict(self, price, capacity, automation, energy, weight):
        # Убираем автоматическое обучение!
        if not self.is_trained:
            raise Exception("Модель не обучена. Сначала нажмите 'Обучить модель'.")

        features = np.array([[price, capacity, automation, energy, weight]])
        prediction = self.model.predict(features)[0]
        return round(prediction, 0)

    def get_dependence_plot(self, feature_name, feature_range, fixed_values=None):
        """
        Построение графика зависимости целевого показателя от выбранного признака
        feature_name: 'avg_equipment_price', 'production_capacity', 'energy_consumption', 'equipment_weight'
        feature_range: список значений (например, list(range(5, 21)))
        fixed_values: словарь с фиксированными значениями остальных признаков
        """
        if not self.is_trained:
            self.train()

        # Значения по умолчанию для фиксированных признаков (средние по датасету)
        if fixed_values is None:
            fixed_values = {
                'avg_equipment_price': self.X['avg_equipment_price'].mean(),
                'production_capacity': self.X['production_capacity'].mean(),
                'is_automation': 0.5,
                'energy_consumption': self.X['energy_consumption'].mean(),
                'equipment_weight': self.X['equipment_weight'].mean()
            }

        predictions = []
        for val in feature_range:
            # Создаём массив признаков
            features = [
                fixed_values['avg_equipment_price'],
                fixed_values['production_capacity'],
                fixed_values['is_automation'],
                fixed_values['energy_consumption'],
                fixed_values['equipment_weight']
            ]

            # Заменяем выбранный признак
            if feature_name == 'avg_equipment_price':
                features[0] = val
            elif feature_name == 'production_capacity':
                features[1] = val
            elif feature_name == 'energy_consumption':
                features[3] = val
            elif feature_name == 'equipment_weight':
                features[4] = val

            pred = self.model.predict([features])[0]
            predictions.append(pred)

        # Названия признаков для подписей
        names = {
            'avg_equipment_price': 'Цена (млн руб.)',
            'production_capacity': 'Производительность (кг/час)',
            'energy_consumption': 'Энергопотребление (кВт·ч)',
            'equipment_weight': 'Вес (кг)'
        }

        # Создаём график
        fig = go.Figure(data=go.Scatter(
            x=feature_range,
            y=predictions,
            mode='lines+markers',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6, color='#2ecc71')
        ))

        fig.update_layout(
            title=f'Зависимость объёма продаж от {names.get(feature_name, feature_name)}',
            xaxis_title=names.get(feature_name, feature_name),
            yaxis_title='Прогнозируемый объём продаж (шт./мес)',
            hovermode='x'
        )

        return fig

    def get_correlation_plot(self):
        """Матрица корреляции"""
        corr_matrix = self.df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}'
        ))

        fig.update_layout(
            title='Матрица корреляции признаков',
            width=600,
            height=600
        )

        return fig

    def get_scatter_plot(self, feature_name):
        """Диаграмма рассеяния для выбранного признака"""
        names = {
            'avg_equipment_price': 'Цена (млн руб.)',
            'production_capacity': 'Производительность (кг/час)',
            'energy_consumption': 'Энергопотребление (кВт·ч)',
            'equipment_weight': 'Вес (кг)',
            'is_automation': 'Автоматизация (1 - да, 0 - нет)'
        }

        fig = go.Figure(data=go.Scatter(
            x=self.df[feature_name],
            y=self.df['monthly_equipment_sales'],
            mode='markers',
            marker=dict(color='#3498db', size=8, opacity=0.6)
        ))

        fig.update_layout(
            title=f'Зависимость продаж от {names.get(feature_name, feature_name)}',
            xaxis_title=names.get(feature_name, feature_name),
            yaxis_title='Месячный объём продаж (шт.)'
        )

        return fig