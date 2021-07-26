"""
逻辑回归判断乳腺癌分类
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer


def logistic():
    # 加载数据
    columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin'
                       '/breast-cancer-wisconsin.data',
                       names=columns)

    # 取出特征和标签
    x = data[columns[1:-1]]
    y = data[columns[-1]]
    # 填充空缺值
    x.replace(to_replace='?', value=np.nan, inplace=True)
    sim = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = sim.fit_transform(x)
    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 标准化，分类问题不需要对标签做标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 模型训练
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    print('准确率为：', lr.score(x_test, y_test))
    print('精确率和召回率：', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))
    return None


if __name__ == '__main__':
    logistic()
