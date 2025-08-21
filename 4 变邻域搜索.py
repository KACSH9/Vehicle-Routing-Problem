# 1 导入库
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from itertools import pairwise
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


# 3 haversine经纬度距离矩阵（要求城市编号为 0..n-1 连续）
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


# 6 邻域操作
# 6.1 2-opt
def two_opt(path, i, j):
    return path[:i] + path[i:j][::-1] + path[j:]

# 6.2 2-exchange
def two_exchange(path, i, j):
    new_path = path[:]
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# 6.3 insertion
def insertion(path, i, j):
    if i == j:
        return path[:]
    new_path = path[:]
    node = new_path.pop(i)
    if j > i:
        j -= 1
    new_path.insert(j, node)
    return new_path

# 6.4 or-opt
def or_opt(path, i, j, k=2):
    if i == j or i + k > len(path) - 1:
        return path[:]
    new_path = path[:]
    segment = new_path[i:i+k]
    del new_path[i:i+k]
    if j > i:
        j -= k
    for idx, node in enumerate(segment):
        new_path.insert(j + idx, node)
    return new_path

# 6.5 3-opt
def three_opt(path, i, j, k):
    if not (1 <= i < j < k <= len(path) - 2):
        return path[:]
    A, B, C, D = path[:i], path[i:j], path[j:k], path[k:]
    return A + B[::-1] + C[::-1] + D


# 7 生成邻域内候选
def gen_moves_two_opt(n, sample_k):
    seen, moves = set(), []
    trials = sample_k * 3 if sample_k is not None else (n * (n - 1)) // 2   # 尝试次数
    while len(moves) < (sample_k or 10**9) and trials > 0:
        trials -= 1
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, n - 2)
        if j - i <= 1:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        moves.append(('2opt', i, j))
    return moves

def gen_moves_two_exchange(n, sample_k):
    seen, moves = set(), []
    trials = sample_k * 5 if sample_k is not None else (n * (n - 1)) // 2
    while len(moves) < (sample_k or 10**9) and trials > 0:
        trials -= 1
        i = random.randint(1, n - 2)
        j = random.randint(1, n - 2)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        moves.append(('2ex', i, j))
    return moves

def gen_moves_insertion(n, sample_k):
    seen, moves = set(), []
    trials = sample_k * 5 if sample_k is not None else n * (n - 1)
    while len(moves) < (sample_k or 10**9) and trials > 0:
        trials -= 1
        i = random.randint(1, n - 2)
        j = random.randint(1, n - 2)
        if i == j:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        moves.append(('ins', i, j))
    return moves

def gen_moves_or_opt(n, sample_k, ks=(2, 3)):
    # 索引范围：1..n-2（不动两端 depot）
    seen, moves = set(), []
    trials = (sample_k or 0) * 6 if sample_k is not None else n * n
    while (sample_k is None or len(moves) < sample_k) and trials > 0:
        trials -= 1
        k = random.choice(ks)
        i = random.randint(1, n - 1 - k)  # 段起点
        j = random.randint(1, n - 2)      # 插入位置（会在 or_opt 内部修正）
        if j >= i and j < i + k:          # 不能插进自身
            continue
        key = (i, j, k)
        if key in seen:
            continue
        seen.add(key)
        moves.append(('or_opt', i, j, k))
    return moves

def gen_moves_three_opt(n, sample_k):
    seen, moves = set(), []
    trials = (sample_k or 0) * 10 if sample_k is not None else n * n
    while (sample_k is None or len(moves) < sample_k) and trials > 0:
        trials -= 1
        i = random.randint(1, n - 4)      # 留出 j,k 空间
        j = random.randint(i + 1, n - 3)
        k = random.randint(j + 1, n - 2)  # <= n-2，不碰尾端 depot
        key = (i, j, k)
        if key in seen:
            continue
        seen.add(key)
        moves.append(('three_opt', i, j, k))
    return moves




# 8 在指定邻域做一次“最好改进”（Best-Improvement）局部搜索
def best_improvement_in_neighborhood(path, D, neighborhood, sample_k=None):
    n = len(path)
    cur_cost = calculate_path_length(path, D)

    if neighborhood == '2opt':
        moves = gen_moves_two_opt(n, sample_k)
    elif neighborhood == '2ex':
        moves = gen_moves_two_exchange(n, sample_k)
    elif neighborhood == 'ins':
        moves = gen_moves_insertion(n, sample_k)
    elif neighborhood == 'or_opt':
        moves = gen_moves_or_opt(n, sample_k, ks=(2, 3))
    elif neighborhood == 'three_opt':
        moves = gen_moves_three_opt(n, sample_k)
    else:
        raise ValueError(f"Unknown neighborhood: {neighborhood}")

    best_path, best_cost = None, float('inf')

    for mv, *params in moves:
        if mv == '2opt':
            i, j = params
            new_path = two_opt(path, i, j)
        elif mv == '2ex':
            i, j = params
            new_path = two_exchange(path, i, j)
        elif mv == 'ins':
            i, j = params
            new_path = insertion(path, i, j)
        elif mv == 'or_opt':
            i, j, k = params
            new_path = or_opt(path, i, j, k=k)
        elif mv == 'three_opt':
            i, j, k = params
            new_path = three_opt(path, i, j, k)
        else:
            continue

        new_cost = calculate_path_length(new_path, D)
        if new_cost < best_cost:
            best_cost, best_path = new_cost, new_path

    if best_path is not None and best_cost + 1e-12 < cur_cost:
        return best_path, best_cost, True
    else:
        return path, cur_cost, False



# 9 VNS 主过程
def vns_tsp(initial_path,
            distance_matrix,
            neighborhoods=('2opt', '2ex', 'ins', 'or_opt', 'three_opt'),
            sample_sizes=(120, 120, 120, 120, 120),
            max_outer_iter=2000,
            max_no_improve_outer=400):

    x = initial_path[:]                              # 当前解
    fx = calculate_path_length(x, distance_matrix)   # 当前距离
    best_x, best_fx = x[:], fx                       # 历史最优

    history_best = [best_fx]
    no_improve_outer = 0
    outer_iters = 0

    for _ in range(max_outer_iter):
        i = 0
        improved_flag = False

        while i < len(neighborhoods):
            nb = neighborhoods[i]
            sample_k = sample_sizes[i] if isinstance(sample_sizes, (list, tuple)) else sample_sizes

            new_x, new_fx, improved = best_improvement_in_neighborhood(
                x, distance_matrix, nb, sample_k=sample_k
            )

            if improved and new_fx + 1e-12 < fx:
                # 有改进：更新当前解，并回到第一个邻域
                x, fx = new_x, new_fx
                if fx + 1e-12 < best_fx:
                    best_x, best_fx = x[:], fx
                improved_flag = True
                i = 0
            else:
                # 无改进：切换到更大的邻域
                i += 1

        outer_iters += 1
        print(f"[迭代 {outer_iters:3d}] | 当前距离={fx:.6f} | 最优距离={best_fx:.6f}")
        print("当前路线:", " -> ".join(map(str, x)))
        print("最优路线:", " -> ".join(map(str, best_x)))

        history_best.append(best_fx)

        if not improved_flag:
            no_improve_outer += 1
            if no_improve_outer >= max_no_improve_outer:
                print(f"连续 {max_no_improve_outer} 轮未改进，提前停止。")
                break
        else:
            no_improve_outer = 0

    return best_x, best_fx, history_best, outer_iters


# 10 画图
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

    plt.title("VNS 最优路径")
    plt.xlabel("经度 / X")
    plt.ylabel("纬度 / Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_convergence(history_distance):
    plt.figure(figsize=(6, 5))
    plt.plot(history_distance, marker='o', label='best so far')
    plt.title("VNS 收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最优路径长度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 11 主程序
if __name__ == "__main__":
    cities, depot_id, customer_ids = read_tsp_excel(
        '/Users/sunhaoqing/Desktop/pythonProject/智能优化算法/ulysses16_TSP.xlsx'
    )
    distance_matrix = build_distance_matrix(cities)
    initial_path = generate_initial_solution(depot_id, customer_ids)

    best_path, best_distance, history_best, iters = vns_tsp(
        initial_path, distance_matrix,
        neighborhoods=('2opt', '2ex', 'ins', 'or_opt', 'three_opt'),
        sample_sizes=200,
        max_outer_iter=4000, max_no_improve_outer=400
    )

    print("\n======== 结果 ========")
    print("最短距离: ", best_distance)
    print("最优路径: ", " -> ".join(map(str, best_path)))
    print("总评估步数: ", iters)

    plot_path(cities, best_path, depot_id=depot_id)
    plot_convergence(history_best)
