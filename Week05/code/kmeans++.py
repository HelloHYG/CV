import math
import random
import sklearn.datasets


def euler_distance(point1: list, point2: list):
    # 计算两点之间的欧拉距离
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无限大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set: list, k: int):
    # 从数据集中返回k个对象可作为质心
    cluster_centers = [random.choice(data_set)]
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):
            # 轮盘法选出下一个聚类中心
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
        return cluster_centers


if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    print(kpp_centers(iris.data, 4))