"""
使用knn预测签到位置
1.数据量过大，适当缩小数据（x,y）范围
2.时间戳转为日期格式，取出月、周、小时当作新的特征。并删除无关特征
3.排除签到人数较少的位置，减小数据量
4.切分训练集和测试集，并做标准化
5.训练并评估
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def knn_cv():
    """
    使用交叉验证和网格搜索对knn进行优化
    :return: None
    """
    # 加载数据
    data = pd.read_csv(r'D:\学习资料\机器学习资料\kaggle数据集\facebook签到位置分析\train.csv')
    # 缩小范围
    data = data.query('x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75')
    # 处理时间戳
    dt = pd.to_datetime(data['time'])
    dt = pd.DatetimeIndex(dt)
    data['month'] = dt.month
    data['week'] = dt.week
    data['hour'] = dt.hour
    # 删除无用的特征
    data = data.drop(['row_id', 'time'], axis=1)
    # 过滤签到人数较少的位置
    cnt = data.groupby('place_id').count().reset_index()
    cnt = cnt[cnt.x > 3]
    data = data[data['place_id'].isin(cnt['place_id'])]
    # print(data)
    # 取出特征和标签
    y = data['place_id']
    x = data.drop('place_id', axis=1)
    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 实例模型
    kn = KNeighborsClassifier()
    # 网格搜索
    param = {'n_neighbors': [3, 5, 9, 11]}
    gc = GridSearchCV(estimator=kn, param_grid=param, cv=5)
    gc.fit(x_train, y_train)
    y_predict = gc.predict(x_test)
    print('测试集精确率：', gc.score(x_test, y_test))
    print('最优超参为：', gc.best_params_)
    print('最优模型为：', gc.best_estimator_)
    print('最优评估为：', gc.best_score_)
    print('每个超参每次交叉验证的结果：', gc.cv_results_)
    return None


def knn():
    # 加载数据
    data = pd.read_csv(r'D:\学习资料\机器学习资料\kaggle数据集\facebook签到位置分析\train.csv')
    # 1.query 筛选符合条件的值
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")
    # 2.转换时间戳格式，精确到秒
    dt = pd.to_datetime(data['time'], unit='s')
    # 时间格式转为字典格式
    dt = pd.DatetimeIndex(dt)
    data['month'] = dt.month
    data['week'] = dt.week
    data['hour'] = dt.hour
    # 删除无关特征
    data = data.drop(['row_id', 'time'], axis=1)
    # 3.删除签到人数较少的位置
    rs = data.groupby(['place_id']).count().reset_index()
    data = data[data['place_id'].isin(rs['place_id'])]
    # print(data)
    # 4.取出特征和目标，切分数据集
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 5.训练
    kn = KNeighborsClassifier(n_neighbors=5)
    kn.fit(x_train, y_train)
    y_predict = kn.predict(x_train)
    # 评估
    score = kn.score(x_test, y_test)
    print('准确率为：', score)

    return None


if __name__ == '__main__':
    knn_cv()
