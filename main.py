import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from modules.quality import QualityEvaluator
from modules.forecast import SalesForecast
from modules.transport import TransportSolver

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("ИАС для производителя оборудования для пищевой промышленности",
            style={'textAlign': 'center', 'color': '#2c3e50'}),

    dcc.Tabs(id='tabs', value='tab-quality', children=[
        dcc.Tab(label='Оценка технического уровня', value='tab-quality'),
        dcc.Tab(label='Прогнозирование продаж', value='tab-forecast'),
        dcc.Tab(label='Оптимизация поставок', value='tab-transport'),
    ]),

    html.Div(id='tabs-content', style={'padding': '20px'})
])


# Вкладка 1: Оценка технического уровня
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-quality':
        return html.Div([
            html.H3("Оценка технического уровня продукции"),
            html.Label("Выберите тип продукции:"),
            dcc.Dropdown(
                id='product-type',
                options=[
                    {'label': 'Линия розлива напитков (эталон: Krones)', 'value': 'bottling'},
                    {'label': 'Фасовочно-упаковочный автомат (эталон: Ishida)', 'value': 'packing'},
                    {'label': 'Печь для хлебопечения (эталон: MIWE)', 'value': 'oven'}
                ],
                value='bottling',
                style={'width': '60%', 'marginBottom': '20px'}
            ),
            html.Button("Рассчитать", id='calc-quality',
                        style={'backgroundColor': '#3498db', 'color': 'white',
                               'padding': '10px 20px', 'marginBottom': '20px'}),

            # Блок с эталонными значениями
            html.Div(id='etalon-info', style={'marginBottom': '20px', 'padding': '15px',
                                              'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

            dcc.Graph(id='radar-chart'),
            dcc.Graph(id='bar-chart')
        ])

    elif tab == 'tab-forecast':
        return html.Div([
            html.H3("Прогнозирование месячного объёма продаж"),

            html.H4("Обучение модели"),
            html.Button("Обучить модель", id='train-model',
                        style={'backgroundColor': '#3498db', 'color': 'white',
                               'padding': '10px 20px', 'marginBottom': '20px'}),
            html.Div(id='model-metrics', style={'marginBottom': '30px',
                                                'padding': '10px',
                                                'backgroundColor': '#ecf0f1'}),

            html.H4("График зависимости"),
            html.Label("Выберите признак:"),
            dcc.Dropdown(
                id='feature-select',
                options=[
                    {'label': 'Цена (млн руб.)', 'value': 'avg_equipment_price'},
                    {'label': 'Производительность (кг/час)', 'value': 'production_capacity'},
                    {'label': 'Энергопотребление (кВт·ч)', 'value': 'energy_consumption'},
                    {'label': 'Вес (кг)', 'value': 'equipment_weight'}
                ],
                value='avg_equipment_price',
                style={'width': '50%', 'marginBottom': '20px'}
            ),
            html.Label("Диапазон значений (через запятую, например: 5,20):"),
            dcc.Input(id='range-input', type='text', value='5,20',
                      style={'width': '200px', 'marginBottom': '20px'}),
            html.Button("Построить график", id='plot-dependence',
                        style={'backgroundColor': '#2ecc71', 'color': 'white',
                               'padding': '10px 20px', 'marginBottom': '20px'}),
            dcc.Graph(id='dependence-graph'),

            html.H4("Ручной прогноз"),
            html.Label("Цена (млн руб.):"),
            dcc.Input(id='input-price', type='number', value=8.5, style={'marginBottom': '10px', 'width': '200px'}),
            html.Br(),
            html.Label("Производительность (кг/час):"),
            dcc.Input(id='input-capacity', type='number', value=500, style={'marginBottom': '10px', 'width': '200px'}),
            html.Br(),
            html.Label("Автоматизация (1 - да, 0 - нет):"),
            dcc.Input(id='input-automation', type='number', value=1, style={'marginBottom': '10px', 'width': '200px'}),
            html.Br(),
            html.Label("Энергопотребление (кВт·ч):"),
            dcc.Input(id='input-energy', type='number', value=25, style={'marginBottom': '10px', 'width': '200px'}),
            html.Br(),
            html.Label("Вес (кг):"),
            dcc.Input(id='input-weight', type='number', value=800, style={'marginBottom': '20px', 'width': '200px'}),
            html.Br(),
            html.Button("Получить прогноз", id='predict-btn',
                        style={'backgroundColor': '#e67e22', 'color': 'white',
                               'padding': '10px 20px'}),
            html.Div(id='prediction-result', style={'marginTop': '20px', 'padding': '10px',
                                                    'backgroundColor': '#d5f5e3'})
        ])

    else:  # tab-transport
        return html.Div([
            html.H3("Оптимизация транспортных поставок"),
            html.P("Исходные данные:"),
            html.Ul([
                html.Li("Мощности площадок: P1 = 30 ед./квартал, P2 = 25 ед./квартал"),
                html.Li("Спрос комбинатов: FC1 = 15, FC2 = 12, FC3 = 20, FC4 = 10 ед./квартал"),
                html.Li("Цель: минимизация общих затрат")
            ]),
            html.Button("Решить транспортную задачу", id='solve-transport',
                        style={'backgroundColor': '#9b59b6', 'color': 'white',
                               'padding': '10px 20px', 'marginBottom': '20px'}),
            html.Div(id='transport-result', style={'marginBottom': '20px',
                                                   'padding': '10px',
                                                   'backgroundColor': '#ecf0f1'}),
            html.H4("Оптимальный план поставок:"),
            html.Div(id='transport-table')
        ])


# Новый callback для отображения эталонных значений
@app.callback(
    Output('etalon-info', 'children'),
    [Input('product-type', 'value')]
)
def show_etalon(product_type):
    try:
        evaluator = QualityEvaluator(product_type)
        etalon = evaluator._get_etalon()

        # Названия эталонов
        etalon_names = {
            'bottling': 'Krones (Германия)',
            'packing': 'Ishida (Япония)',
            'oven': 'MIWE (Германия)'
        }

        if not etalon:
            return html.Div("Нет данных для выбранного типа продукции")

        rows = []
        for indicator, value in etalon.items():
            rows.append(html.Tr([html.Td(indicator, style={'padding': '8px', 'fontWeight': 'bold'}),
                                 html.Td(f"{value}", style={'padding': '8px'})]))

        return html.Div([
            html.H5(f"📊 Эталонные значения ({etalon_names.get(product_type, '')}):",
                    style={'marginBottom': '10px'}),
            html.Table(rows, style={'borderCollapse': 'collapse', 'width': '100%', 'border': '1px solid #dee2e6'})
        ], style={'border': '1px solid #dee2e6', 'borderRadius': '5px', 'padding': '10px'})

    except Exception as e:
        return html.Div(f"Ошибка загрузки эталонных данных: {str(e)}")


# Callback для расчёта оценки ТУ
@app.callback(
    [Output('radar-chart', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('calc-quality', 'n_clicks')],
    [State('product-type', 'value')]
)
def update_quality(n_clicks, product_type):
    if n_clicks is None:
        return go.Figure(), go.Figure()

    evaluator = QualityEvaluator(product_type)

    sample_data = evaluator._get_sample_default()
    evaluator.add_sample(sample_data, "Наша продукция")

    if product_type == 'bottling':
        competitor_a = {
            'Производительность (бут/час)': 14000,
            'Точность дозирования (%)': 99.0,
            'Энергопотребление (кВт·ч)': 46,
            'Ресурс (тыс. часов)': 45000,
            'Автоматизация (баллы)': 8
        }
        competitor_b = {
            'Производительность (бут/час)': 10000,
            'Точность дозирования (%)': 98.0,
            'Энергопотребление (кВт·ч)': 55,
            'Ресурс (тыс. часов)': 35000,
            'Автоматизация (баллы)': 6
        }
        evaluator.add_sample(competitor_a, "Конкурент А")
        evaluator.add_sample(competitor_b, "Конкурент Б")

    elif product_type == 'packing':
        competitor_a = {
            'Скорость упаковки (уп/мин)': 130,
            'Точность взвешивания (%)': 0.7,
            'Расход материала (м²/час)': 0.55,
            'Энергопотребление (кВт·ч)': 3.2,
            'Диапазон размеров (мм)': 280
        }
        competitor_b = {
            'Скорость упаковки (уп/мин)': 80,
            'Точность взвешивания (%)': 1.2,
            'Расход материала (м²/час)': 0.85,
            'Энергопотребление (кВт·ч)': 4.5,
            'Диапазон размеров (мм)': 220
        }
        evaluator.add_sample(competitor_a, "Конкурент А")
        evaluator.add_sample(competitor_b, "Конкурент Б")

    else:
        competitor_a = {
            'Вместимость (противней/час)': 180,
            'Макс. температура (°C)': 330,
            'Равномерность выпечки (баллы)': 8,
            'Расход энергии (кВт·ч)': 28,
            'Время разогрева (мин)': 18
        }
        competitor_b = {
            'Вместимость (противней/час)': 120,
            'Макс. температура (°C)': 280,
            'Равномерность выпечки (баллы)': 6,
            'Расход энергии (кВт·ч)': 40,
            'Время разогрева (мин)': 30
        }
        evaluator.add_sample(competitor_a, "Конкурент А")
        evaluator.add_sample(competitor_b, "Конкурент Б")

    results = evaluator.evaluate_all()
    radar_fig = evaluator.get_radar_chart(results)
    bar_fig = evaluator.get_bar_chart(results)

    return radar_fig, bar_fig


# Создаём объекты
forecast_model = SalesForecast()
transport_solver = TransportSolver()


# Callback для обучения модели
@app.callback(
    Output('model-metrics', 'children'),
    [Input('train-model', 'n_clicks')]
)
def train_model(n_clicks):
    if n_clicks is None:
        return "Нажмите 'Обучить модель' для начала"

    r2_train, r2_test, mae = forecast_model.train()

    return html.Div([
        html.H5("Результаты обучения модели:"),
        html.P(f"R² на обучающей выборке: {r2_train:.3f}"),
        html.P(f"R² на тестовой выборке: {r2_test:.3f}"),
        html.P(f"Средняя абсолютная ошибка (MAE): {mae:.1f} шт./мес"),
        html.P(f"Уравнение регрессии: y = {forecast_model.coefficients['intercept']:.1f} "
               f"- {abs(forecast_model.coefficients['avg_equipment_price']):.1f}*цена "
               f"+ {forecast_model.coefficients['production_capacity']:.2f}*производительность "
               f"+ {forecast_model.coefficients['is_automation']:.1f}*автоматизация "
               f"- {abs(forecast_model.coefficients['energy_consumption']):.1f}*энергия "
               f"- {abs(forecast_model.coefficients['equipment_weight']):.2f}*вес")
    ])


# Callback для построения графика зависимости
@app.callback(
    Output('dependence-graph', 'figure'),
    [Input('plot-dependence', 'n_clicks')],
    [State('feature-select', 'value'),
     State('range-input', 'value')]
)
def plot_dependence(n_clicks, feature, range_str):
    if n_clicks is None:
        return go.Figure()

    try:
        parts = range_str.split(',')
        min_val = float(parts[0])
        max_val = float(parts[1])

        if feature in ['avg_equipment_price', 'energy_consumption', 'equipment_weight']:
            feature_range = list(range(int(min_val), int(max_val) + 1))
        else:
            step = max(1, (int(max_val) - int(min_val)) // 50)
            feature_range = list(range(int(min_val), int(max_val) + 1, step))

        return forecast_model.get_dependence_plot(feature, feature_range)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Ошибка: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


# Callback для ручного прогноза
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('input-price', 'value'),
     State('input-capacity', 'value'),
     State('input-automation', 'value'),
     State('input-energy', 'value'),
     State('input-weight', 'value')]
)
def manual_predict(n_clicks, price, capacity, automation, energy, weight):
    if n_clicks is None:
        return "Введите значения и нажмите 'Получить прогноз'"

    try:
        pred = forecast_model.predict(price, capacity, automation, energy, weight)
        return html.Div([
            html.H5(f"Прогнозируемый объём продаж: {int(pred)} шт./мес")
        ])
    except:
        return "Ошибка: сначала обучите модель"


# Callback для транспортной задачи
@app.callback(
    [Output('transport-result', 'children'),
     Output('transport-table', 'children')],
    [Input('solve-transport', 'n_clicks')]
)
def solve_transport(n_clicks):
    if n_clicks is None:
        return "Нажмите 'Решить транспортную задачу'", ""

    result = transport_solver.solve()

    if result['success']:
        df, total_cost = transport_solver.get_result_table()

        header = html.Tr([html.Th("Поставщик \\ Потребитель")] + [html.Th(col) for col in df.columns])

        rows = []
        for row in df.index:
            rows.append(html.Tr([html.Th(row)] + [html.Td(df.loc[row, col]) for col in df.columns]))

        table = html.Table([header] + rows)

        result_text = html.Div([
            html.H5(f"Минимальные общие затраты: {total_cost:.0f} млн руб."),
            html.P(result.get('balance_msg', ''))
        ])

        return result_text, table
    else:
        return html.Div([
            html.H5("Ошибка при решении задачи"),
            html.P(result.get('message', 'Неизвестная ошибка'))
        ]), ""


if __name__ == '__main__':
    app.run(debug=True)