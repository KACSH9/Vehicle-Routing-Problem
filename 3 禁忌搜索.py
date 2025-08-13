# 1 导入库
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from itertools import pairwise
from collections import deque
from haversine import haversine

font_path = '/Library/Fonts/Arial Unicode.ttf'  # 替换为实际的字体路径
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


# 2 读取文件数据
def read_tsp_excel(file_path):
    df = pd.read_excel(file_path)

    cities = {}

    # 配送中心
    depot_id = int(df.iloc[0, 0])
    depot_x = float(df.iloc[0, 1])
    depot_y = float(df.iloc[0, 2])
    cities[depot_id] = (depot_x, depot_y)

    # 客户点
    customer_ids = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        cus_id = int(row.iloc[0])
        lat = float(row.iloc[1])
        lon = float(row.iloc[2])
        cities[cus_id] = (lat, lon)
        customer_ids.append(cus_id)

    return cities, depot_id, customer_ids


# 3 haversine经纬度距离矩阵
def build_distance_matrix(cities):
    n = len(cities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(cities[i], cities[j])
            D[i, j] = D[j, i] = d

    return D


# 4 生成初始解
def generate_initial_solution(depot_id, customer_ids):
    customer_seq = customer_ids.copy()
    random.shuffle(customer_seq)
    path = [depot_id] + customer_seq + [depot_id]

    return path


# 5 路径长度
def calculate_path_length(path, D):
    total_distance = 0.0
    for a, b in pairwise(path):
        total_distance += D[a, b]

    return total_distance


# 6 邻域操作(2-opt/2-exchange)
# 6.1 2-opt 固定位置转换
def two_opt(path, i, j):
    new_path = path[:i] + path[i:j][::-1] + path[j:]

    return new_path

# 6.2 2-exchange 固定位置转换
def two_exchange(path, i, j):
    new_path = path[:]
    new_path[i], new_path[j] = new_path[j], new_path[i]

    return new_path

# 7 禁忌表操作
# 7.1 特征编码
def ts_2opt_key(path_before, i, j):
    ci = path_before[i]
    cj = path_before[j]

    return ('2opt', tuple(sorted((ci, cj))))

def ts_2exchange_key(path_before, i, j):
    ci = path_before[i]
    cj = path_before[j]

    return ('2exchange', tuple(sorted((ci, cj))))

# 7.2 生成候选邻域
def cand_neighbors(path, k_2opt, k_2exchange):
    n = len(path)
    candidates = set()

    # 2-opt候选
    for _ in range(k_2opt * 3):
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, n - 2)
        if j - i > 1:
            candidates.add(('2opt', i, j))
        if len([c for c in candidates if c[0] == '2opt']) >= k_2opt:
            break

    # 2-exchange候选
    for _ in range(k_2exchange * 3):
        i = random.randint(1, n - 2)
        j = random.randint(1, n - 2)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        candidates.add(('2exchange', i, j))
        if len([c for c in candidates if c[0] == '2exchange']) >= k_2exchange:
            break

    return list(candidates)

# 7.3 禁忌搜索主过程
def ts_tsp(initial_path,
           distance_matrix,
           ts_tenure=15,
           max_iter=2000,
           k_2opt=60,
           k_2ex=60,
           max_no_improve=300,):
    # 初始解
    cur_path = initial_path.copy()
    cur_cost = calculate_path_length(cur_path, distance_matrix)

    best_path = cur_path.copy()
    best_cost = cur_cost

    # 禁忌表
    tabu_list = deque(maxlen=ts_tenure)

    history_best = [best_cost]
    no_improve = 0

    for it in range(1, max_iter + 1):
        # 候选邻域
        moves = cand_neighbors(cur_path, k_2opt=k_2opt, k_2exchange=k_2ex)

        best_move = None
        best_move_cost = float('inf')
        best_move_path = None
        best_move_key = None

        # 评估候选 + 禁忌 + 特赦准则
        for (mv, i, j) in moves:
            if mv == '2opt':
                new_path = two_opt(cur_path, i, j)
                new_cost = calculate_path_length(new_path, distance_matrix)
                key = ts_2opt_key(cur_path, i, j)
            else:  # '2ex'
                new_path = two_exchange(cur_path, i, j)
                new_cost = calculate_path_length(new_path, distance_matrix)
                key = ts_2exchange_key(cur_path, i, j)

            is_tabu = key in tabu_list

            # 特赦准则：若能打破历史最优，破禁
            if is_tabu and new_cost < best_cost:
                is_tabu = False

            if not is_tabu:
                if new_cost < best_move_cost:
                    best_move_cost = new_cost
                    best_move = (mv, i, j)
                    best_move_path = new_path
                    best_move_key = key

        # 若全被禁且无法破禁，保底选禁忌中的最优
        if best_move is None:
            for (mv, i, j) in moves:
                if mv == '2opt':
                    new_path = two_opt(cur_path, i, j)
                    new_cost = calculate_path_length(new_path, distance_matrix)
                    key = ts_2opt_key(cur_path, i, j)
                else:
                    new_path = two_exchange(cur_path, i, j)
                    new_cost = calculate_path_length(new_path, distance_matrix)
                    key = ts_2exchange_key(cur_path, i, j)

                if new_cost < best_move_cost:
                    best_move_cost = new_cost
                    best_move = (mv, i, j)
                    best_move_path = new_path
                    best_move_key = key

        cur_path = best_move_path
        cur_cost = best_move_cost

        tabu_list.append(best_move_key)

        improved = False
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_path = cur_path.copy()
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        history_best.append(best_cost)

        mv, i, j = best_move
        print(f"[迭代 {it:3d}] | 当前距离={cur_cost:.6f} | 最优距离={best_cost:.6f} ")
        print("当前路线:", " -> ".join(map(str, cur_path)))
        print("历史最优路线:", " -> ".join(map(str, best_path)))

        if no_improve >= max_no_improve:
            print(f"连续 {max_no_improve} 代未改进，提前停止。")
            break

    return best_path, best_cost, history_best, it


# 8 画图
def plot_path(cities, path, depot_id=None):
    x_coords = [cities[city][0] for city in path]
    y_coords = [cities[city][1] for city in path]

    plt.figure(figsize=(6, 5))
    plt.plot(x_coords, y_coords, linestyle='-', color='gray', label='路径')

    for city in path:
        x, y = cities[city]
        plt.scatter(x, y, c='blue', s=40)
        plt.text(x, y, str(city), fontsize=8, ha='right', va='bottom')

    if depot_id is not None:
        depot_x, depot_y = cities[depot_id]
        plt.scatter(depot_x, depot_y, c='red', s=100, label='起始点（depot）', zorder=5)

    plt.title("禁忌搜索最优路径")
    plt.xlabel("经度 / X")
    plt.ylabel("纬度 / Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_convergence(history_distance):
    plt.figure(figsize=(6, 5))
    plt.plot(history_distance, marker='o', label='best so far')
    plt.title("禁忌搜索收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最优路径长度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 9 主程序
if __name__ == "__main__":
    cities, depot_id, customer_ids = read_tsp_excel(
        '/Users/sunhaoqing/Desktop/pythonProject/智能优化算法/ulysses16_TSP.xlsx'
    )
    distance_matrix = build_distance_matrix(cities)
    initial_path = generate_initial_solution(depot_id, customer_ids)

    best_path, best_distance, history_best, iters = ts_tsp(
        initial_path,
        distance_matrix,
        ts_tenure=15,
        max_iter=4000,
        k_2opt=80,
        k_2ex=80,
        max_no_improve=400
    )

    print("\n======== 结果 ========")
    print("最短距离: ", best_distance)
    print("最优路径: ", " -> ".join(map(str, best_path)))
    print("总迭代次数: ", iters)

    plot_path(cities, best_path, depot_id=depot_id)
    plot_convergence(history_best)













