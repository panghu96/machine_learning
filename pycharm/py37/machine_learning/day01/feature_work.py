"""
特征工程：对非数值类型数据进行特征值化
DictVectorizer：对字典类型的数据进行特征值化
CountVectorizer：对文本类型的数据进行特征值化，得到的是每个词出现的次数
TfidfVectorizer：对文本类型的数据进行特征值化，log(总文档数/该词出现的文档数)，表示该词的重要程度
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def tfidfVec():
    """
    使用tfidf提取文本的重要程度
    :return:
    """
    c1, c2, c3 = chinese_cut()
    tf = TfidfVectorizer()
    # 提取特征
    tf_trans = tf.fit_transform([c1, c2, c3])
    # 特征名称
    feature_names = tf.get_feature_names()
    print(feature_names)
    print(tf_trans.toarray())
    return None


def chinese_cut():
    """
    利用jieba对中文进行分词
    :return:
    """
    # 切分单词
    cut1 = jieba.cut("人与人，无信不交往，守信方长久；心与心，互敬才生情，互爱才有真。欺人莫欺心，伤人勿伤情。信任一个人很难，再次相信一个人更难。")
    cut2 = jieba.cut("人生路上几人来，几人走，几多欢喜，几多忧愁，为何会如此不堪。相遇不易，相守很难，珍惜且珍惜。")
    cut3 = jieba.cut("成长必然充斥了生命的创痛，我们还可以肩并肩寻找幸福就已足够。")
    # 分词返回列表
    content1 = list(cut1)
    content2 = list(cut2)
    content3 = list(cut3)
    # 列表转为以空格分隔的字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def chineseVec():
    """
    利用jieba分词实现中文文本特征提取
    :return:
    """
    c1, c2, c3 = chinese_cut()
    cv = CountVectorizer()
    # 特征值化
    cv_trans = cv.fit_transform([c1, c2, c3])
    # 特征名称
    feature_names = cv.get_feature_names()
    # 特征值化稀疏矩阵转为ndarray
    cv_matrix = cv_trans.toarray()
    print(feature_names)
    print(cv_matrix)
    return None


def englishVec():
    """
    文本特征提取：对（英文）文本类型的数据进行特征值化
    :return:
    """
    cv = CountVectorizer()
    cv_trans = cv.fit_transform(['life is is short,i like python', 'life is to long,i dislike python'])
    # 特征
    feature_names = cv.get_feature_names()
    print(feature_names)
    # 稀疏矩阵转为ndarray，矩阵中每个位置的数值代表出现的次数
    print(cv_trans.toarray())
    return None


def dictVec():
    """
    字典特征提取：对字典类型的数据进行特征值化
    :return:None
    """
    # False不产生稀疏矩阵
    dictV = DictVectorizer(sparse=False)
    data = [{'city': '北京', 'salary': 200},
            {'city': '上海', 'salary': 180},
            {'city': '深圳', 'salary': 170}]
    # 对字典数据进行提取
    res = dictV.fit_transform(data)
    # feature名称
    feature_names = dictV.get_feature_names()
    # 特征提取后的数据样式
    trans_data = dictV.inverse_transform(res)
    print(feature_names)
    print(res)
    print(trans_data)
    return None


if __name__ == "__main__":
    # dictVec()
    # englishVec()
    # chineseVec()
    tfidfVec()