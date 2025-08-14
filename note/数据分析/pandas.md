# Pandas简介

Pandas 是 Python 数据分析工具链中最核心的库，充当数据读取、清洗、分析统计、辙出的高效工具。

Pandas 提供了易于使用的数据结构和数据分析工具，特别适用于处理结构化数据，如表格型数据(类似于Excel表格)。

Pandas 是数据科学和分析领域中常用的工具之一，它使得用户能够轻松地从各种数据源中导入数据，并对数据进行高效的操作和分析。

Pandas是基于NumPy构建的专门为处理表格和混杂数据设计的Python库，核心设计理念包括：

- 标签化数据结构：提供带标签的轴
- 灵活处理缺失数据：内置NaN处理机制
- 智能数据对齐：自动按标签对齐数据
- 强大I0工具：支持从CSV、Excel、SQL等20+数据源读写
- 时间序列处理：原生支持日期时间处理和频率转换

## 工具对比

|      工具       |       功能特色       |           使用场景           |
| :-------------: | :------------------: | :--------------------------: |
|      Excel      |  图形界面，上手简单  |     人工分析、小规模数据     |
|       SQL       | 高效读写，最终数据源 |       数据库查询和联表       |
| Python + Pandas |  算法和分析部署核心  | 数据清洗，统计分析，可视化等 |

## 数据类型对比

|   特性   |        Series        |           DataFrame           |
| :------: | :------------------: | :---------------------------: |
|   维度   |         一维         |             二维              |
|   索引   |        单索引        |          行索引+列名          |
| 数据存储 |    同质化数据类型    |      各列可不同数据类型       |
|   类比   |      Excel单列       |        整张Excel工作表        |
| 创建方式 | pd.Series([1, 2, 3]) | pd.DataFrame({'col':[1,2,3]}) |

# Series

## Series创建

- 一般创建方式

```python
# Series的创建
import pandas as pd

s = pd.Series([1,2,3,4,5])
print(s)

# 自定义索引
s = pd.Series([1,2,3,4,5], index=['A','B','C','D','E'])
print(s)

# 定义name
s = pd.Series([1,2,3,4,5], index=['A','B','C','D','E'], name='月份')
print(s)


'''
0    1
1    2
2    3
3    4
4    5
dtype: int64
A    1
B    2
C    3
D    4
E    5
dtype: int64
A    1
B    2
C    3
D    4
E    5
Name: 月份, dtype: int64
'''
```

- 通过字典创建

```python
# 通过字典创建
s = pd.Series({'a':1, 'b':2, 'c':3, 'd':4, 'e':5})
print(s)
# 可以截取Series的一部分作为新的Series
s1 = pd.Series(s,index=['a','c'])
print(s1)


'''
a    1
b    2
c    3
d    4
e    5
dtype: int64
a    1
c    3
dtype: int64
'''
```

## Series属性

|  属性  |       说明       |  属性  |            说明            |
| :----: | :--------------: | :----: | :------------------------: |
| index  | Series的索引对象 | loc[]  | 显示索引，按标签索引或切片 |
| values |    Series的值    | iloc[] | 隐式索引，按位置索引或切片 |
| dtype  | Series的元素类型 |  at[]  |    使用标签访问单个元素    |
| shape  |   Series的形状   | iat[]  |    使用位置访问单个元素    |
|  ndim  |   Series的维度   |        |                            |
|  size  | Series的元素个数 |        |                            |
|  name  |   Series的名称   |        |                            |

```python
print(s)
print(s.index)
print(s.values)
print(s.shape, s.ndim, s.size)
s.name = 'test'
print(s.dtype, s.name)
print(s.loc['a':'c'])
print(s.iloc[0:2])
print(s.at['a']) # 不支持切片
print(s.iat[0]) # 不支持切片


'''
a    1
b    2
c    3
d    4
e    5
Name: test, dtype: int64
Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
[1 2 3 4 5]
(5,) 1 5
int64 test
a    1
b    2
c    3
Name: test, dtype: int64
a    1
b    2
Name: test, dtype: int64
1
1
'''
```



## 访问Series数据

```python
# 访问数据
print(s)
# print(s[1]) # 不推荐使用，容易混淆
print(s['a']) # 通过标签
print(s[s<3]) # 布尔索引
s['f'] = 6
print(s.head()) # 返回前五行数据
print(s.tail()) # 返回后五行数据


'''
a    1
b    2
c    3
d    4
e    5
Name: test, dtype: int64
1
a    1
b    2
Name: test, dtype: int64
a    1
b    2
c    3
d    4
e    5
Name: test, dtype: int64
b    2
c    3
d    4
e    5
f    6
Name: test, dtype: int64
'''
```

## Series常用方法

| 方法   | 说明                               | 方法        | 说明                                                      |
| :----- | :--------------------------------- | :---------- | :-------------------------------------------------------- |
| head() | 查看前 n 行数据，默认 5 行         | max()       | 最大值                                                    |
| tail() | 查看后 n 行数据，默认 5 行         | var()       | 方差                                                      |
| isin() | 判断元素是否包含子啊参数集合中     | std()       | 标准差                                                    |
| isna() | 判断是否为缺失值（如 NaN 或 None） | median      | 中位数                                                    |
| sum()  | 求和，自动忽略缺失值               | mode()      | 众数（可返回多个）                                        |
| mean() | 平均值                             | quantile(q) | 分位数，q 去 0~1 之间                                     |
| min()  | 最小值                             | describe()  | 常见统计信息（count、mean、std、min、25%、50%、75%、max） |

| 方法              | 说明                 | 方法          | 说明                   |
| ----------------- | -------------------- | ------------- | ---------------------- |
| value_counts()    | 每个唯一值的出现次数 | sort_values() | 按值排序               |
| count()           | 非缺失值数量         | replace()     | 替换值                 |
| nunique()         | 唯一值个数（去重）   | keys()        | 返回 Series 的索引对象 |
| unique()          | 获取去重后的值数组   |               |                        |
| drop_duplicates() | 去除重复项           |               |                        |
| sample()          | 随机取样             |               |                        |
| sort_index()      | 按索引排序           |               |                        |

