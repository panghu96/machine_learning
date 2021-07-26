"""
随机森林：集成学习，多组决策树的组合。每组决策树的样本不同
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def rd_forest():
    # 加载数据
    data = pd.read_csv(r'D:\学习资料\机器学习资料\kaggle数据集\泰坦尼克号生存预测\titanic.csv')
    # 取出需要的数据
    x = data[['pclass', 'age', 'sex']]
    y = data['survived']
    # 填补空值
    x['age'].fillna(x['age'].mean(), inplace=True)
    # x['boat'].fillna('None', inplace=True)
    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征值化
    dc = DictVectorizer(sparse=False)
    x_train = dc.fit_transform(x_train.to_dict(orient='records'))
    x_test = dc.transform(x_test.to_dict(orient='records'))
    # 创建模型
    rf = RandomForestClassifier()
    # 网格搜索和交叉验证
    param = {'n_estimators': [120, 150, 200, 300, 500, 800, 1200],
             'max_depth': [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param, cv=3)
    gc.fit(x_train, y_train)
    y_predict = gc.predict(x_test)
    print('准确率为：', gc.score(x_test, y_test))
    print('精确率和召回率：', classification_report(y_test, y_predict))
    print('最优参数为：', gc.best_params_)
    return None


if __name__ == '__main__':
    rd_forest()
