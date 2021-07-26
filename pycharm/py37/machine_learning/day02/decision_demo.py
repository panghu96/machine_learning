"""
使用决策树对泰坦尼克号生存预测
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def dec():
    # 加载数据
    data = pd.read_csv(r'D:\学习资料\机器学习资料\kaggle数据集\泰坦尼克号生存预测\titanic.csv')
    # 取出需要的数据
    x = data[['pclass', 'sex', 'age', 'boat']]
    y = data['survived']
    # 空值处理
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 字符串类型的数据，空值填充为随机字符串
    x['boat'].fillna('-999', inplace=True)
    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程 one-hot编码
    # orient='records' 将DataFrame数据转为[{key1:value}, {key2:value}]类型
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(x_train.to_dict(orient='records'))
    print(dv.get_feature_names())
    x_test = dv.transform(x_test.to_dict(orient='records'))
    # 创建决策树模型
    dec = DecisionTreeClassifier()
    # 网格搜索与交叉验证
    param = {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 3, 5]}
    gc = GridSearchCV(dec, param, cv=3)
    gc.fit(x_train, y_train)
    y_predict = gc.predict(x_test)
    print('准确率为：', gc.score(x_test, y_test))
    print('精确率和召回率为：', classification_report(y_test, y_predict))
    print('最优超参为：', gc.best_estimator_)

    return None


if __name__ == '__main__':
    dec()
