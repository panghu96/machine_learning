from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minMax():
    """
    归一化：把原始数据映射到[0,1]之间，这个[0,1]的范围是默认的，可以手动指定。
    公式：x‘ = (x - min) / (max - min)     x'' = x' * (mx - mi) + mi
    缺点：最大值和最小值容易受异常数据的影响，该方法鲁棒性较差。只适合传统精确小数据量的场景
    :return:None
    """
    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


def standard():
    """
    标准化：把原始特征数据映射到均值为0，方差为1的范围内
    公式：(x - mean) / σ
    :return:None
    """
    stand = StandardScaler()
    data = stand.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


if __name__ == "__main__":
    standard()
