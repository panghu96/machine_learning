"""
线性回归预测波士顿房价
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error

# 加载数据
data = load_boston()
# 切分数据集
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)
# 标准化，注意线性回归标准化需要将标签也一起进行标准化，需要两个标准化API
x_std = StandardScaler()
x_train = x_std.fit_transform(x_train)
x_test = x_std.transform(x_test)
y_std = StandardScaler()
# 参数必须是二维数组，使用reshape将一维数组转为二维
y_train = y_std.fit_transform(y_train.reshape(-1, 1))
y_test = y_std.fit_transform(y_test.reshape(-1, 1))


def linear():
    """
    正规方程解（最小二乘法）线性回归模型
    :return:None
    """
    # 创建线性回归模型
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print('最小二乘法模型系数为：', lr.coef_)
    # 预测值转为标准化前的值
    y_predict = y_std.inverse_transform(lr.predict(x_test))
    # print('预测结果为：', y_predict)
    # 线性回归模型使用均方误差来进行评估，不要忘了把目标值转为标准化前的值
    mse = mean_squared_error(y_std.inverse_transform(y_test), y_predict)
    print('最小二乘法模型误差为：', mse)
    return None


def SGD_linear():
    """
    梯度下降法线性回归模型
    :return: None
    """
    # 创建梯度下降算法线性回归模型
    sgd_lr = SGDRegressor()
    sgd_lr.fit(x_train, y_train)
    print('梯度下降法模型系数为：', sgd_lr.coef_)
    # 预测值转为标准化前的值
    y_predict = y_std.inverse_transform(sgd_lr.predict(x_test))
    # print('预测结果为：', y_predict)
    # 线性回归模型使用均方误差来进行评估，不要忘了把目标值转为标准化前的值
    mse = mean_squared_error(y_std.inverse_transform(y_test), y_predict)
    print('梯度下降法模型误差为：', mse)
    return None


def ridge():
    """
    岭回归，具有L2正则化的线性最小二乘法，解决过拟合问题
    :return:None
    """
    # alpha正则化力度
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    y_predict = rd.predict(x_test)
    # 转化为标准化前的值
    mse = mean_squared_error(y_std.inverse_transform(y_test), y_std.inverse_transform(y_predict))
    print('岭回归系数为：', rd.coef_)
    print('岭回归均方误差为：', mse)
    return None


if __name__ == '__main__':
    linear()
    SGD_linear()
    ridge()