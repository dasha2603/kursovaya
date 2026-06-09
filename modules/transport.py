import numpy as np
import pandas as pd
from scipy.optimize import linprog


class TransportSolver:
    """Класс для решения транспортной задачи"""

    def __init__(self):
        """Инициализация данных транспортной задачи"""
        # Мощности поставщиков (производственные площадки)
        self.supplies = [30, 25]  # P1, P2
        self.supply_names = ['P1', 'P2']

        # Спрос потребителей (пищевые комбинаты)
        self.demands = [15, 12, 20, 10]  # FC1, FC2, FC3, FC4
        self.demand_names = ['FC1', 'FC2', 'FC3', 'FC4']

        # Матрица затрат (млн руб./ед.)
        self.costs = np.array([
            [15, 17, 19, 21],  # P1 -> FC1, FC2, FC3, FC4
            [18, 16, 18, 20]  # P2 -> FC1, FC2, FC3, FC4
        ])

        self.result = None
        self.total_cost = None
        self.is_solved = False

    def check_balance(self):
        """Проверка сбалансированности задачи"""
        total_supply = sum(self.supplies)
        total_demand = sum(self.demands)

        if total_supply == total_demand:
            return True, total_supply, total_demand
        else:
            return False, total_supply, total_demand

    def add_dummy(self):
        """Добавление фиктивного поставщика при несбалансированности"""
        total_supply = sum(self.supplies)
        total_demand = sum(self.demands)

        if total_supply < total_demand:
            # Добавляем фиктивного поставщика
            dummy_supply = total_demand - total_supply
            self.supplies.append(dummy_supply)
            self.supply_names.append('Фиктивный')
            # Добавляем строку с нулевыми затратами
            dummy_row = np.zeros(len(self.demands))
            self.costs = np.vstack([self.costs, dummy_row])
            return f"Добавлен фиктивный поставщик с мощностью {dummy_supply} ед."

        elif total_supply > total_demand:
            # Добавляем фиктивного потребителя
            dummy_demand = total_supply - total_demand
            self.demands.append(dummy_demand)
            self.demand_names.append('Фиктивный')
            # Добавляем столбец с нулевыми затратами
            dummy_col = np.zeros(len(self.supplies))
            self.costs = np.column_stack([self.costs, dummy_col])
            return f"Добавлен фиктивный потребитель со спросом {dummy_demand} ед."

        return "Задача сбалансирована"

    def solve(self):
        """Решение транспортной задачи методом линейного программирования"""

        # Проверяем баланс и добавляем фиктивного при необходимости
        balanced, total_supply, total_demand = self.check_balance()
        balance_msg = ""

        if not balanced:
            balance_msg = self.add_dummy()

        # Количество переменных = количество поставщиков * количество потребителей
        n_suppliers = len(self.supplies)
        n_consumers = len(self.demands)
        n_vars = n_suppliers * n_consumers

        # Целевая функция (коэффициенты затрат в виде вектора)
        c = self.costs.flatten()

        # Ограничения: для каждого поставщика сумма поставок = его мощность
        A_eq_supply = []
        b_eq_supply = []

        for i in range(n_suppliers):
            row = np.zeros(n_vars)
            for j in range(n_consumers):
                row[i * n_consumers + j] = 1
            A_eq_supply.append(row)
            b_eq_supply.append(self.supplies[i])

        # Ограничения: для каждого потребителя сумма поставок = его спрос
        A_eq_demand = []
        b_eq_demand = []

        for j in range(n_consumers):
            row = np.zeros(n_vars)
            for i in range(n_suppliers):
                row[i * n_consumers + j] = 1
            A_eq_demand.append(row)
            b_eq_demand.append(self.demands[j])

        # Объединяем все ограничения-равенства
        A_eq = np.vstack([A_eq_supply, A_eq_demand])
        b_eq = np.array(b_eq_supply + b_eq_demand)

        # Границы переменных (неотрицательность)
        bounds = [(0, None) for _ in range(n_vars)]

        # Решаем задачу линейного программирования
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            self.is_solved = True
            self.total_cost = result.fun

            # Преобразуем результат в матрицу для удобного отображения
            self.result = result.x.reshape(n_suppliers, n_consumers)

            return {
                'success': True,
                'total_cost': self.total_cost,
                'plan': self.result,
                'supply_names': self.supply_names.copy(),
                'demand_names': self.demand_names.copy(),
                'balance_msg': balance_msg
            }
        else:
            return {
                'success': False,
                'message': 'Не удалось найти оптимальное решение'
            }

    def get_result_table(self):
        """Получение таблицы результатов для отображения"""
        if not self.is_solved:
            return None, None

        # Создаём DataFrame для отображения
        df = pd.DataFrame(
            self.result,
            index=self.supply_names,
            columns=self.demand_names
        )

        # Округляем значения (могут быть небольшие погрешности из-за вычислений)
        df = df.round(0).astype(int)

        return df, self.total_cost

    def get_solution_summary(self):
        """Получение текстового описания решения"""
        if not self.is_solved:
            return "Задача ещё не решена"

        df, total_cost = self.get_result_table()

        summary = f"**Минимальные общие затраты: {total_cost:.0f} млн руб.**\n\n"
        summary += "**Оптимальный план поставок:**\n\n"

        for i in range(len(self.supply_names)):
            for j in range(len(self.demand_names)):
                val = self.result[i][j]
                if val > 0.01:  # Игнорируем нулевые поставки
                    summary += f"- {self.supply_names[i]} → {self.demand_names[j]}: {int(round(val))} ед.\n"

        return summary