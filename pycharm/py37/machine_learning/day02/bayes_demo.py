"""
使用朴素贝叶斯对新闻文档进行分类
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def bayes():
    # 加载数据,data_home指定数据下载地址
    news = fetch_20newsgroups(data_home=r'D:\学习资料\机器学习资料\fetch数据集', subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 提取重要特征
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    # alpha即拉普拉斯平滑系数
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print('准确率为：', mlt.score(x_test, y_test))
    print('精确率和召回率为：', classification_report(y_test, y_predict))
    return None


if __name__ == '__main__':
    bayes()
