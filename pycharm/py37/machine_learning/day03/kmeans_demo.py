import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def kmeans():
    # 加载数据
    aisles = pd.read_csv('D:/学习资料/机器学习资料/kaggle数据集/启动市场篮子分析/aisles.csv')
    prior = pd.read_csv('D:/学习资料/机器学习资料/kaggle数据集/启动市场篮子分析/order_products__prior.csv')
    orders = pd.read_csv('D:/学习资料/机器学习资料/kaggle数据集/启动市场篮子分析/orders.csv')
    products = pd.read_csv('D:/学习资料/机器学习资料/kaggle数据集/启动市场篮子分析/products.csv')
    # 合并几张表
    _mg = pd.merge(prior, orders, on=['order_id', 'order_id'])
    _mg = pd.merge(_mg, products, on=['product_id', 'product_id'])
    mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])
    # 数据整理成类似行类的形式（交叉表）
    # 每一行代表用户购买的所有商品
    cross = pd.crosstab(mt['user_id'], mt['aisle'])
    # PCA主成分分析（降维）
    # 一般降到原始数据的90%-95%
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    # 取其中部分数据做KMeans聚类
    data = data[:1000]
    km = KMeans(n_clusters=4)
    km.fit(data)
    predict = km.predict(data)
    print('聚类结果为：', predict)
    # 绘制散点图查看聚类结果
    plt.figure(figsize=(10, 10))
    # 建立四个颜色的列表
    color = ['orange', 'blue', 'red', 'black']
    color = [color[i] for i in predict]
    # 取两个特征作为坐标
    plt.scatter(data[:, 1], data[:, 20], color=color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # 轮廓系数判断KMeans聚类效果 取值范围(-1,1)，越大越好
    score = silhouette_score(data,predict)
    print('轮廓系数为：', score)
    return None


if __name__ == '__main__':
    kmeans()
