import pandas as pd
import numpy as np
import plotly.graph_objs as go


class QualityEvaluator:
    """Класс для оценки технического уровня продукции"""

    def __init__(self, product_type):
        self.product_type = product_type
        self.etalon_data = self._get_etalon()
        self.samples = []
        self.sample_names = []

    def _get_etalon(self):
        """Эталонные значения для каждого типа продукции"""
        etalons = {
            'bottling': {  # Линия розлива напитков (Krones)
                'Производительность (бут/час)': 16000,
                'Точность дозирования (%)': 99.5,
                'Энергопотребление (кВт·ч)': 45,
                'Ресурс (тыс. часов)': 50000,
                'Автоматизация (баллы)': 9
            },
            'packing': {  # Фасовочно-упаковочный автомат (Ishida)
                'Скорость упаковки (уп/мин)': 150,
                'Точность взвешивания (%)': 0.5,
                'Расход материала (м²/час)': 0.5,
                'Энергопотребление (кВт·ч)': 3.0,
                'Диапазон размеров (мм)': 300
            },
            'oven': {  # Печь для хлебопечения (MIWE)
                'Вместимость (противней/час)': 200,
                'Макс. температура (°C)': 350,
                'Равномерность выпечки (баллы)': 9,
                'Расход энергии (кВт·ч)': 25,
                'Время разогрева (мин)': 15
            }
        }
        return etalons.get(self.product_type, {})

    def _get_sample_default(self):
        """Значения образца (наша продукция)"""
        samples = {
            'bottling': {
                'Производительность (бут/час)': 12000,
                'Точность дозирования (%)': 98.5,
                'Энергопотребление (кВт·ч)': 50,
                'Ресурс (тыс. часов)': 40000,
                'Автоматизация (баллы)': 7
            },
            'packing': {
                'Скорость упаковки (уп/мин)': 100,
                'Точность взвешивания (%)': 1.0,
                'Расход материала (м²/час)': 0.7,
                'Энергопотребление (кВт·ч)': 4.0,
                'Диапазон размеров (мм)': 250
            },
            'oven': {
                'Вместимость (противней/час)': 150,
                'Макс. температура (°C)': 300,
                'Равномерность выпечки (баллы)': 7,
                'Расход энергии (кВт·ч)': 35,
                'Время разогрева (мин)': 25
            }
        }
        return samples.get(self.product_type, {})

    def add_sample(self, sample_data, sample_name="Образец 1"):
        """Добавление образца для сравнения"""
        self.samples.append(sample_data)
        self.sample_names.append(sample_name)

    def _is_stimulator(self, indicator_name):
        """Определение типа показателя (стимулятор/дестимулятор)"""
        # Дестимуляторы (чем меньше, тем лучше)
        desstimulators = ['Энергопотребление (кВт·ч)', 'Расход материала (м²/час)',
                          'Время разогрева (мин)', 'Расход энергии (кВт·ч)']
        return indicator_name not in desstimulators

    def calculate_relative(self, sample_data):
        """Расчёт относительных показателей q_i"""
        q_values = {}
        for indicator, etalon_value in self.etalon_data.items():
            sample_value = sample_data.get(indicator, 0)
            if etalon_value == 0 or sample_value == 0:
                q_values[indicator] = 0
                continue

            is_stim = self._is_stimulator(indicator)
            if is_stim:
                q = sample_value / etalon_value
            else:
                q = etalon_value / sample_value
            q_values[indicator] = round(q, 3)
        return q_values

    def calculate_integral_index(self, q_values):
        """Расчёт интегрального индекса Q (среднее арифметическое)"""
        values = list(q_values.values())
        if len(values) == 0:
            return 0
        return round(sum(values) / len(values), 3)

    def evaluate_all(self):
        """Оценка всех добавленных образцов"""
        results = []
        for i, sample in enumerate(self.samples):
            q_values = self.calculate_relative(sample)
            Q = self.calculate_integral_index(q_values)
            results.append({
                'name': self.sample_names[i],
                'q_values': q_values,
                'Q': Q
            })
        # Сортировка по Q (от лучшего к худшему)
        results.sort(key=lambda x: x['Q'], reverse=True)
        return results

    def get_radar_chart(self, results):
        """Построение радиальной диаграммы для лучшего образца"""
        if not results:
            return go.Figure()

        best = results[0]
        indicators = list(best['q_values'].keys())
        values = list(best['q_values'].values())

        # Замыкаем график
        indicators.append(indicators[0])
        values.append(values[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=indicators,
            fill='toself',
            name=best['name']
        ))

        max_val = max(max(values), 1.2)

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_val])
            ),
            title=f"Радиальная диаграмма - {best['name']}"
        )
        return fig

    def get_bar_chart(self, results):
        """Построение столбчатой диаграммы для всех образцов"""
        if not results:
            return go.Figure()

        names = [r['name'] for r in results]
        Q_values = [r['Q'] for r in results]

        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(names))]

        fig = go.Figure(data=go.Bar(
            x=names,
            y=Q_values,
            marker_color=colors,
            text=Q_values,
            textposition='auto'
        ))

        fig.update_layout(
            title="Интегральные индексы качества (Q)",
            xaxis_title="Образец",
            yaxis_title="Интегральный индекс",
            yaxis_range=[0, max(1.2, max(Q_values))]
        )
        return fig