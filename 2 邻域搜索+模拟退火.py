'''
TSP问题 
2-opt + 2-exchange 
SA接受准则
'''

# 1 导入库
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from geopy.distance import geodesic

font_path = '/Library/Fonts/Arial Unicode.ttf'  # 替换为实际的字体路径
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 2 读取文件数据
def read_tsp_excel(file_path):
    df = pd.read_excel(file_path)

    cities = {}

    # 配送中心（第0行）
    depot_id = int(df.iloc[0, 0])
    depot_x = float(df.iloc[0, 1])
    depot_y = float(df.iloc[0, 2])
    cities[depot_id] = (depot_x, depot_y)

    customer_ids = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        id = int(row.iloc[0])
        x = float(row.iloc[1])
        y = float(row.iloc[2])
        cities[id] = (x, y)
        customer_ids.append(id)

    return cities, depot_id, customer_ids

# 3 构建距离矩阵
def geo_distance(p1, p2):
    return geodesic(p1, p2).kilometers

def build_distance_matrix(cities):
    D = {}
    for i, coord_i in cities.items():
        D[i] = {}
        for j, coord_j in cities.items():
            if i == j:
                D[i][j] = 0.0
            else:
                D[i][j] = geo_distance(coord_i, coord_j)
    return D

# 4 初始解
def generate_initial_solution(depot_id, customer_ids):
    customer_seq = customer_ids.copy()
    random.shuffle(customer_seq)
    path = [depot_id] + customer_seq + [depot_id]
    return path

# 5 路径长度
def calculate_path_length(path, distance_matrix):
    dist_mat = np.array([[distance_matrix[i][j] for j in path] for i in path])
    indices = np.arange(len(path) - 1)
    return np.sum(dist_mat[indices, indices + 1])

# 6 邻域搜索
# 6.1 2-opt
def gen_neighbor_2opt(path):
    n = len(path)
    if n <= 4:
        return path[:]
    while True:
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, n - 2)
        if j - i > 1:
            break
    new_path = path[:i] + path[i:j][::-1] + path[j:]
    return new_path

# 6.2 2-exchange
def gen_neighbor_2exchange(path):
    n = len(path)
    if n <= 4:
        return path[:]
    i = random.randint(1, n - 2)
    j = random.randint(1, n - 2)
    while j == i:
        j = random.randint(1, n - 2)
    new_path = path[:]
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# 7 模拟退火接受准则
def simulated_annealing_tsp(initial_path, distance_matrix, T0, Tmin, alpha, L, max_outer, p):
    # 初始化
    x_cur = initial_path.copy()
    f_cur = calculate_path_length(x_cur, distance_matrix)

    x_best = x_cur.copy()
    f_best = f_cur

    T = T0

    history_best = [f_best]
    history_temp = [T]
    iter_counter = 0

    # 外循环
    for k in range(1, max_outer+1):
        if T < Tmin:
            print(f"达到最低温度 Tmin={Tmin}，停止。")
            break

        accepted_cnt = 0
        improved_cnt = 0

        # 内循环
        for _ in range(L):
            if random.random() < p:
                x_new = gen_neighbor_2opt(x_cur)
            else:
                x_new = gen_neighbor_2exchange(x_cur)

            f_new = calculate_path_length(x_new, distance_matrix)
            delta = f_new - f_cur

            if delta <= 0:
                x_cur, f_cur = x_new, f_new
                accepted_cnt += 1
            else:
                u = random.random()
                if u < math.exp(-delta / T):
                    x_cur, f_cur = x_new, f_new
                    accepted_cnt += 1

            if f_cur < f_best:
                x_best, f_best = x_cur.copy(), f_cur
                improved_cnt += 1

        history_best.append(f_best)
        history_temp.append(T)
        iter_counter += 1

        print(
            f"[outer {k:4d}] T={T:.6f} | "
            f"accepted={accepted_cnt:4d}/{L} | "
            f"improved_best={improved_cnt:3d} | "
            f"f_best={f_best:.6f} | "
            f"cur_path = {x_cur} | "
            f"best_path = {x_best}"
        )

        T = alpha * T

    return x_best, f_best, history_best, history_temp, iter_counter

# 8 画图
def plot_path(cities, path, depot_id=None):
    x_coords = [cities[city][0] for city in path] + [cities[path[0]][0]]
    y_coords = [cities[city][1] for city in path] + [cities[path[0]][1]]

    plt.figure(figsize=(6, 5))
    plt.plot(x_coords, y_coords, linestyle='-', color='gray', label='路径')

    for city in path:
        x, y = cities[city]
        plt.scatter(x, y, c='blue', s=40)
        plt.text(x, y, str(city), fontsize=8, ha='right', va='bottom')

    if depot_id is not None:
        depot_x, depot_y = cities[depot_id]
        plt.scatter(depot_x, depot_y, c='red', s=100, label='起始点（depot）', zorder=5)

    plt.title("模拟退火最优路径")
    plt.xlabel("经度 / X")
    plt.ylabel("纬度 / Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_convergence(history_distance):
    plt.figure(figsize=(6, 5))
    plt.plot(history_distance, marker='o', label='best so far')
    plt.title("模拟退火收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最优路径长度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_temperature(history_temp):
    plt.figure(figsize=(6, 5))
    plt.plot(history_temp)
    plt.title("温度变化曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("温度 T")
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

    best_path, best_distance, history_best, history_temp, iters = simulated_annealing_tsp(
        initial_path,
        distance_matrix,
        T0=1000,
        Tmin=1e-4,
        alpha=0.98,
        L=200,
        max_outer=1000,
        p=0.7
    )

    print("\n======== 结果 ========")
    print("Best distance: ", best_distance)
    print("Best path: ", " -> ".join(map(str, best_path)))
    print("Total iterations: ", iters)

    plot_path(cities, best_path, depot_id=depot_id)
    plot_convergence(history_best)
    plot_temperature(history_temp)
