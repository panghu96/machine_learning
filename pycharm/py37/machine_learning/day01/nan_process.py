"""
缺失值处理：
    删除：如果某一行或某一列的缺失值所占比例过多，可以放弃此行或此列的数据
    插补：计算当前行或者当前列的平均值、中位数，填充到缺失值位置
"""
import numpy as np
from sklearn.impute import SimpleImputer


def im():
    """
    使用SimpleImputer填充空值，注意空值的类型是float类型的np.nan
    :return:
    """
    # 空值类型是np.nan，填充规则为平均值
    sim = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = sim.fit_transform([[1, 2],
                       [np.nan, 3],
                       [7, 6]]
                      )
    print(data)
    return None


if __name__ == '__main__':
    im()