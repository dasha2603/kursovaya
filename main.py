# Глобальный флаг, что модель обучена
model_is_trained = False

import dash
from dash import dcc, html, Input, Output, State, ALL
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


# ==================== ВКЛАДКА 1: ОЦЕНКА ТУ ====================
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
                style={'width': '50%', 'marginBottom': '20px'}
            ),

            html.Div(id='etalon-info', style={'marginBottom': '20px', 'padding': '15px',
                                              'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

            html.H4("Введите характеристики образца:"),
            html.Div(id='input-fields-quality', style={'marginBottom': '20px'}),

            html.Div([
                html.Button("➕ Добавить образец", id='add-sample-btn',
                            style={'backgroundColor': '#2ecc71', 'color': 'white',
                                   'padding': '8px 16px', 'marginRight': '10px'}),
                html.Button("🗑 Очистить все образцы", id='clear-samples-btn',
                            style={'backgroundColor': '#e74c3c', 'color': 'white',
                                   'padding': '8px 16px', 'marginRight': '10px'}),
                html.Button("📊 Рассчитать", id='calc-quality',
                            style={'backgroundColor': '#3498db', 'color': 'white',
                                   'padding': '8px 16px'})
            ], style={'marginBottom': '20px'}),

            html.H4("Добавленные образцы:"),
            html.Div(id='samples-list', style={'marginBottom': '20px', 'padding': '10px',
                                               'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),

            # Хранилище для образцов по типам
            dcc.Store(id='samples-storage-bottling', data=[]),
            dcc.Store(id='samples-storage-packing', data=[]),
            dcc.Store(id='samples-storage-oven', data=[]),

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
            html.Div(id='model-metrics', style={'marginBottom': '30px', 'padding': '10px',
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

            html.H4("Мощности площадок (через запятую):"),
            dcc.Input(id='supplies-input', type='text', value='30,25',
                      style={'width': '300px', 'marginBottom': '10px'}),
            html.P("Пример: 30,25", style={'fontSize': '12px', 'color': 'gray', 'marginTop': '0px'}),

            html.H4("Спрос комбинатов (через запятую):"),
            dcc.Input(id='demands-input', type='text', value='15,12,20,10',
                      style={'width': '300px', 'marginBottom': '10px'}),
            html.P("Пример: 15,12,20,10", style={'fontSize': '12px', 'color': 'gray', 'marginTop': '0px'}),

            html.H4("Матрица затрат (строки через точку с запятой, числа через запятую):"),
            dcc.Textarea(id='costs-input',
                         value='15,17,19,21\n18,16,18,20',
                         style={'width': '400px', 'height': '80px', 'marginBottom': '10px'}),
            html.P("Пример:\n15,17,19,21\n18,16,18,20",
                   style={'fontSize': '12px', 'color': 'gray', 'whiteSpace': 'pre-line'}),

            html.Button("Решить транспортную задачу", id='solve-transport',
                        style={'backgroundColor': '#9b59b6', 'color': 'white',
                               'padding': '10px 20px', 'marginBottom': '20px'}),

            html.Div(id='transport-result', style={'marginBottom': '20px', 'padding': '10px',
                                                   'backgroundColor': '#ecf0f1'}),
            html.H4("Оптимальный план поставок:"),
            html.Div(id='transport-table')
        ])


# ==================== ЭТАЛОННЫЕ ЗНАЧЕНИЯ ====================
@app.callback(
    Output('etalon-info', 'children'),
    [Input('product-type', 'value')]
)
def show_etalon(product_type):
    evaluator = QualityEvaluator(product_type)
    etalon = evaluator._get_etalon()

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


# ==================== ПОЛЯ ДЛЯ ВВОДА ХАРАКТЕРИСТИК ====================
@app.callback(
    Output('input-fields-quality', 'children'),
    [Input('product-type', 'value')]
)
def update_input_fields(product_type):
    if product_type == 'bottling':
        return html.Div([
            html.Label("Производительность (бут/час):"),
            dcc.Input(id={'type': 'indicator', 'index': 'bottling_1'}, type='number', value=12000,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Точность дозирования (%):"),
            dcc.Input(id={'type': 'indicator', 'index': 'bottling_2'}, type='number', value=98.5, step=0.1,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Энергопотребление (кВт·ч):"),
            dcc.Input(id={'type': 'indicator', 'index': 'bottling_3'}, type='number', value=50,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Ресурс (тыс. часов):"),
            dcc.Input(id={'type': 'indicator', 'index': 'bottling_4'}, type='number', value=40000,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Автоматизация (баллы 1-10):"),
            dcc.Input(id={'type': 'indicator', 'index': 'bottling_5'}, type='number', value=7, min=1, max=10,
                      style={'width': '200px', 'marginBottom': '10px'}),
        ])
    elif product_type == 'packing':
        return html.Div([
            html.Label("Скорость упаковки (уп/мин):"),
            dcc.Input(id={'type': 'indicator', 'index': 'packing_1'}, type='number', value=100,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Точность взвешивания (%):"),
            dcc.Input(id={'type': 'indicator', 'index': 'packing_2'}, type='number', value=1.0, step=0.1,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Расход материала (м²/час):"),
            dcc.Input(id={'type': 'indicator', 'index': 'packing_3'}, type='number', value=0.7, step=0.1,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Энергопотребление (кВт·ч):"),
            dcc.Input(id={'type': 'indicator', 'index': 'packing_4'}, type='number', value=4.0, step=0.1,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Диапазон размеров (мм):"),
            dcc.Input(id={'type': 'indicator', 'index': 'packing_5'}, type='number', value=250,
                      style={'width': '200px', 'marginBottom': '10px'}),
        ])
    else:
        return html.Div([
            html.Label("Вместимость (противней/час):"),
            dcc.Input(id={'type': 'indicator', 'index': 'oven_1'}, type='number', value=150,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Макс. температура (°C):"),
            dcc.Input(id={'type': 'indicator', 'index': 'oven_2'}, type='number', value=300,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Равномерность выпечки (баллы 1-10):"),
            dcc.Input(id={'type': 'indicator', 'index': 'oven_3'}, type='number', value=7, min=1, max=10,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Расход энергии (кВт·ч):"),
            dcc.Input(id={'type': 'indicator', 'index': 'oven_4'}, type='number', value=35,
                      style={'width': '200px', 'marginBottom': '10px'}),
            html.Br(),
            html.Label("Время разогрева (мин):"),
            dcc.Input(id={'type': 'indicator', 'index': 'oven_5'}, type='number', value=25,
                      style={'width': '200px', 'marginBottom': '10px'}),
        ])


# ==================== ДОБАВЛЕНИЕ ОБРАЗЦА ====================
def get_storage_id(product_type):
    if product_type == 'bottling':
        return 'samples-storage-bottling'
    elif product_type == 'packing':
        return 'samples-storage-packing'
    else:
        return 'samples-storage-oven'


@app.callback(
    [Output('samples-storage-bottling', 'data'),
     Output('samples-storage-packing', 'data'),
     Output('samples-storage-oven', 'data'),
     Output('samples-list', 'children')],
    [Input('add-sample-btn', 'n_clicks'),
     Input('clear-samples-btn', 'n_clicks'),
     Input('product-type', 'value')],
    [State('product-type', 'value'),
     State({'type': 'indicator', 'index': ALL}, 'value'),
     State('samples-storage-bottling', 'data'),
     State('samples-storage-packing', 'data'),
     State('samples-storage-oven', 'data')]
)
def manage_samples(add_clicks, clear_clicks, product_type_trigger, current_product_type, input_values,
                   bottling_samples, packing_samples, oven_samples):
    ctx = dash.callback_context
    if not ctx.triggered:
        return bottling_samples or [], packing_samples or [], oven_samples or [], "Нет добавленных образцов"

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Получаем текущее хранилище для выбранного типа
    if current_product_type == 'bottling':
        current_samples = bottling_samples or []
    elif current_product_type == 'packing':
        current_samples = packing_samples or []
    else:
        current_samples = oven_samples or []

    # Обработка очистки
    if trigger_id == 'clear-samples-btn':
        if current_product_type == 'bottling':
            return [], packing_samples or [], oven_samples or [], "Нет добавленных образцов"
        elif current_product_type == 'packing':
            return bottling_samples or [], [], oven_samples or [], "Нет добавленных образцов"
        else:
            return bottling_samples or [], packing_samples or [], [], "Нет добавленных образцов"

    # Обработка добавления образца
    if trigger_id == 'add-sample-btn' and add_clicks and input_values and len(input_values) >= 5:
        if current_product_type == 'bottling':
            sample = {
                'Производительность (бут/час)': input_values[0],
                'Точность дозирования (%)': input_values[1],
                'Энергопотребление (кВт·ч)': input_values[2],
                'Ресурс (тыс. часов)': input_values[3],
                'Автоматизация (баллы)': input_values[4]
            }
            sample_name = f"Образец {len(current_samples) + 1}"
            current_samples.append({'name': sample_name, 'data': sample})

            # Формируем список для отображения
            samples_list = html.Ul([html.Li(f"{s['name']}") for s in current_samples])

            if current_product_type == 'bottling':
                return current_samples, packing_samples or [], oven_samples or [], samples_list
            elif current_product_type == 'packing':
                return bottling_samples or [], current_samples, oven_samples or [], samples_list
            else:
                return bottling_samples or [], packing_samples or [], current_samples, samples_list

        elif current_product_type == 'packing':
            sample = {
                'Скорость упаковки (уп/мин)': input_values[0],
                'Точность взвешивания (%)': input_values[1],
                'Расход материала (м²/час)': input_values[2],
                'Энергопотребление (кВт·ч)': input_values[3],
                'Диапазон размеров (мм)': input_values[4]
            }
            sample_name = f"Образец {len(current_samples) + 1}"
            current_samples.append({'name': sample_name, 'data': sample})

            samples_list = html.Ul([html.Li(f"{s['name']}") for s in current_samples])

            if current_product_type == 'bottling':
                return current_samples, packing_samples or [], oven_samples or [], samples_list
            elif current_product_type == 'packing':
                return bottling_samples or [], current_samples, oven_samples or [], samples_list
            else:
                return bottling_samples or [], packing_samples or [], current_samples, samples_list

        else:  # oven
            sample = {
                'Вместимость (противней/час)': input_values[0],
                'Макс. температура (°C)': input_values[1],
                'Равномерность выпечки (баллы)': input_values[2],
                'Расход энергии (кВт·ч)': input_values[3],
                'Время разогрева (мин)': input_values[4]
            }
            sample_name = f"Образец {len(current_samples) + 1}"
            current_samples.append({'name': sample_name, 'data': sample})

            samples_list = html.Ul([html.Li(f"{s['name']}") for s in current_samples])

            if current_product_type == 'bottling':
                return current_samples, packing_samples or [], oven_samples or [], samples_list
            elif current_product_type == 'packing':
                return bottling_samples or [], current_samples, oven_samples or [], samples_list
            else:
                return bottling_samples or [], packing_samples or [], current_samples, samples_list

    # Если просто переключили тип продукции - показываем список образцов для этого типа
    if trigger_id == 'product-type':
        if current_product_type == 'bottling':
            samples_list = html.Ul([html.Li(f"{s['name']}") for s in (bottling_samples or [])]) if (
                bottling_samples) else "Нет добавленных образцов"
            return bottling_samples or [], packing_samples or [], oven_samples or [], samples_list
        elif current_product_type == 'packing':
            samples_list = html.Ul([html.Li(f"{s['name']}") for s in (packing_samples or [])]) if (
                packing_samples) else "Нет добавленных образцов"
            return bottling_samples or [], packing_samples or [], oven_samples or [], samples_list
        else:
            samples_list = html.Ul([html.Li(f"{s['name']}") for s in (oven_samples or [])]) if (
                oven_samples) else "Нет добавленных образцов"
            return bottling_samples or [], packing_samples or [], oven_samples or [], samples_list

    return bottling_samples or [], packing_samples or [], oven_samples or [], "Нет добавленных образцов"


# ==================== ОБНОВЛЕНИЕ СПИСКА ПРИ СМЕНЕ ТИПА ====================
@app.callback(
    Output('samples-list', 'children', allow_duplicate=True),
    [Input('product-type', 'value')],
    [State('samples-storage-bottling', 'data'),
     State('samples-storage-packing', 'data'),
     State('samples-storage-oven', 'data')],
    prevent_initial_call=True
)
def update_samples_list_on_type_change(product_type, bottling_samples, packing_samples, oven_samples):
    if product_type == 'bottling':
        if bottling_samples:
            return html.Ul([html.Li(f"{s['name']}") for s in bottling_samples])
        return "Нет добавленных образцов"
    elif product_type == 'packing':
        if packing_samples:
            return html.Ul([html.Li(f"{s['name']}") for s in packing_samples])
        return "Нет добавленных образцов"
    else:
        if oven_samples:
            return html.Ul([html.Li(f"{s['name']}") for s in oven_samples])
        return "Нет добавленных образцов"


# ==================== РАСЧЁТ ОЦЕНКИ ТУ ====================
@app.callback(
    [Output('radar-chart', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('calc-quality', 'n_clicks')],
    [State('product-type', 'value'),
     State('samples-storage-bottling', 'data'),
     State('samples-storage-packing', 'data'),
     State('samples-storage-oven', 'data')]
)
def update_quality(n_clicks, product_type, bottling_samples, packing_samples, oven_samples):
    if n_clicks is None:
        return go.Figure(), go.Figure()

    evaluator = QualityEvaluator(product_type)

    # Берём образцы только для выбранного типа
    if product_type == 'bottling':
        samples = bottling_samples or []
    elif product_type == 'packing':
        samples = packing_samples or []
    else:
        samples = oven_samples or []

    if not samples:
        return go.Figure(), go.Figure()

    for sample in samples:
        evaluator.add_sample(sample['data'], sample['name'])

    results = evaluator.evaluate_all()
    radar_fig = evaluator.get_radar_chart(results)
    bar_fig = evaluator.get_bar_chart(results)

    return radar_fig, bar_fig


# ==================== ПРОГНОЗИРОВАНИЕ ====================
forecast_model = SalesForecast()


@app.callback(
    Output('model-metrics', 'children'),
    [Input('train-model', 'n_clicks')]
)
def train_model(n_clicks):
    global model_is_trained
    if n_clicks is None:
        return "Нажмите 'Обучить модель' для начала"

    r2_train, r2_test, mae = forecast_model.train()
    model_is_trained = True  # ← вот здесь флаг становится True

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


@app.callback(
    Output('dependence-graph', 'figure'),
    [Input('plot-dependence', 'n_clicks')],
    [State('feature-select', 'value'),
     State('range-input', 'value')]
)
def plot_dependence(n_clicks, feature, range_str):
    # Проверяем, что кнопка нажата
    if n_clicks is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Нажмите 'Построить график'",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Проверяем, что модель обучена
    global model_is_trained
    if not model_is_trained:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Сначала обучите модель (кнопка 'Обучить модель')",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=16)
        )
        return fig

    try:
        parts = range_str.split(',')
        min_val = float(parts[0])
        max_val = float(parts[1])

        if min_val >= max_val:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ Минимальное значение должно быть меньше максимального",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(color="red", size=16)
            )
            return fig

        if feature in ['avg_equipment_price', 'energy_consumption', 'equipment_weight']:
            feature_range = list(range(int(min_val), int(max_val) + 1))
        else:
            step = max(1, (int(max_val) - int(min_val)) // 50)
            feature_range = list(range(int(min_val), int(max_val) + 1, step))

        return forecast_model.get_dependence_plot(feature, feature_range)

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Ошибка: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        return fig


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
    global model_is_trained

    if n_clicks is None:
        return "Введите значения и нажмите 'Получить прогноз'"

    if not model_is_trained:
        return html.Div(
            "⚠️ Сначала обучите модель (кнопка 'Обучить модель')",
            style={'color': 'red', 'fontSize': '16px', 'padding': '10px'}
        )

    try:
        pred = forecast_model.predict(price, capacity, automation, energy, weight)
        return html.Div([
            html.H5(f"📊 Прогнозируемый объём продаж: {int(pred)} шт./мес",
                    style={'color': '#2c3e50'})
        ])
    except Exception as e:
        return html.Div(
            f"Ошибка: {str(e)}",
            style={'color': 'red', 'padding': '10px'}
        )

# ==================== ТРАНСПОРТНАЯ ЗАДАЧА ====================
@app.callback(
    [Output('transport-result', 'children'),
     Output('transport-table', 'children')],
    [Input('solve-transport', 'n_clicks')],
    [State('supplies-input', 'value'),
     State('demands-input', 'value'),
     State('costs-input', 'value')]
)
def solve_transport_with_data(n_clicks, supplies_str, demands_str, costs_str):
    if n_clicks is None:
        return "Нажмите 'Решить транспортную задачу'", ""

    try:
        supplies = [float(x.strip()) for x in supplies_str.split(',')]
        demands = [float(x.strip()) for x in demands_str.split(',')]

        lines = costs_str.strip().split('\n')
        costs = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(',')]
            costs.append(row)

        solver = TransportSolver()
        solver.set_data(supplies, demands, costs)

        result = solver.solve()

        if result['success']:
            df, total_cost = solver.get_result_table()

            header = html.Tr([html.Th("Поставщик \\ Потребитель")] + [html.Th(col) for col in df.columns])
            rows = []
            for row in df.index:
                rows.append(html.Tr([html.Th(row)] + [html.Td(df.loc[row, col]) for col in df.columns]))

            table = html.Table([header] + rows, style={'border': '1px solid black', 'borderCollapse': 'collapse'})

            result_text = html.Div([
                html.H5(f"Минимальные общие затраты: {total_cost:.0f} млн руб."),
                html.P(result.get('balance_msg', ''))
            ])

            return result_text, table
        else:
            return html.Div(
                [html.H5("Ошибка при решении задачи"), html.P(result.get('message', 'Неизвестная ошибка'))]), ""

    except Exception as e:
        return html.Div([html.H5("Ошибка ввода данных"), html.P(str(e))]), ""


if __name__ == '__main__':
    app.run(debug=True)