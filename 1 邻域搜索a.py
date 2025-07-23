'''
TSP问题 有起始点
数据采用十进制经纬度
2-opt + 最优改进策略
'''

# 1 导入库
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from geopy.distance import geodesic   # 十进制格式（即：纬度/经度小数表示，例如 38.24°）

font_path = '/Library/Fonts/Arial Unicode.ttf' # 替换为实际的字体路径
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False # 防止负号显示为方块


# 2 读取文件数据
def read_tsp_excel(file_path):
    df = pd.read_excel(file_path)

    cities = {}

    # 提取配送中心信息（第0行）
    depot_id = int(df.iloc[0, 0])
    depot_x = float(df.iloc[0, 1])
    depot_y = float(df.iloc[0, 2])
    cities[depot_id] = (depot_x, depot_y)  # 添加到城市字典中

    customer_ids = []  # 保存所有客户城市编号（不含 depot）

    # 提取客户城市信息（从第1行开始）
    for i in range(1, len(df)):
        row = df.iloc[i]
        id = int(row.iloc[0])
        x = float(row.iloc[1])
        y = float(row.iloc[2])
        cities[id] = (x, y)
        customer_ids.append(id)

    return cities, depot_id, customer_ids


# 3 构建距离矩阵
# 3.1 计算两点间GEO距离
def geo_distance(p1, p2):
    return geodesic(p1, p2).kilometers

# 3.2 构建矩阵
def build_distance_matrix(cities):
    D = {}
    for i, coord_i in cities.items():
        D[i] = {}
        for j, coord_j in cities.items():
            if i == j:
                D[i][j] = 0
            else:
                D[i][j] = geo_distance(coord_i, coord_j)
    return D


# 4 生成初始解
# 4.1 随机生成
def generate_initial_solution(depot_id, customer_ids):
    customer_seq = customer_ids.copy()
    random.shuffle(customer_seq)
    path = [depot_id] + customer_seq + [depot_id]

    return path


# 5 路径长度计算函数
def calculate_path_length(path, distance_matrix):
    dist_mat = np.array([[distance_matrix[i][j] for j in path] for i in path])
    indices = np.arange(len(path) - 1)
    return np.sum(dist_mat[indices, indices + 1])


# 6 2-opt+最优改进策略
def two_opt(path, distance_matrix, max_iter):
    best_path = path.copy()
    best_distance = calculate_path_length(path, distance_matrix)

    # 初始化记录结构
    history_paths = [best_path.copy()]
    history_best_distance = [best_distance]
    iteration_list = [0]

    iteration = 1
    improved = True

    while improved and iteration <= max_iter:
        improved = False

        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                if j - i == 1:
                    continue

                # 生成新路径
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                new_distance = calculate_path_length(new_path, distance_matrix)

                # 记录信息
                iteration_list.append(iteration)
                history_paths.append(new_path.copy())

                # 更新最优路径
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True
                    history_best_distance.append(best_distance)
                else:
                    history_best_distance.append(history_best_distance[-1])

                # 实时输出
                print(f"Iteration: {iteration}")
                print(f"Path: {' -> '.join(map(str, new_path))}")
                print(f"Best distance: {history_best_distance[-1]:.2f}")
                print("-" * 40)

                iteration += 1

                # 提前终止判断
                if iteration > max_iter:
                    print(f"⚠️ 达到最大迭代次数限制（max_iter={max_iter}），提前终止。")
                    break
            if iteration > max_iter:
                break

    return best_path, best_distance, history_best_distance, history_paths, iteration_list


# 7 画图
# 7.1 路径图
def plot_path(cities, path, depot_id=None):
    # 提取路径坐标（闭合路径）
    x_coords = [cities[city][0] for city in path] + [cities[path[0]][0]]
    y_coords = [cities[city][1] for city in path] + [cities[path[0]][1]]

    plt.figure(figsize=(6, 5))
    plt.plot(x_coords, y_coords, linestyle='-', color='gray', label='路径')

    # 绘制城市节点
    for city in path:
        x, y = cities[city]
        plt.scatter(x, y, c='blue', s=40)
        plt.text(x, y, str(city), fontsize=8, ha='right', va='bottom')

    # 绘制起始点
    if depot_id is not None:
        depot_x, depot_y = cities[depot_id]
        plt.scatter(depot_x, depot_y, c='red', s=100, label='起始点（depot）', zorder=5)

    plt.title("路径图")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 7.2 迭代图
def plot_convergence(history_distance):
    plt.figure(figsize=(6, 5))
    plt.plot(history_distance, marker='o', color='green', label='路径长度')
    plt.title("迭代图")
    plt.xlabel("迭代次数")
    plt.ylabel("路径总距离")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 8 主程序调用
cities, depot_id, customer_ids = read_tsp_excel('/Users/sunhaoqing/Desktop/pythonProject/智能优化算法/ulysses16_TSP.xlsx')
initial_path = generate_initial_solution(depot_id, customer_ids)
distance_matrix =build_distance_matrix(cities)
best_path, best_distance, history_best_distance, history_paths, iteration_list = two_opt(
    initial_path, distance_matrix, max_iter=500
)


# 9 输出结果
plot_path(cities, best_path, depot_id=depot_id)
plot_convergence(history_best_distance)

