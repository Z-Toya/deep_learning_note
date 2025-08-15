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
| isin() | 判断元素是否包含在参数集合中       | std()       | 标准差                                                    |
| isna() | 判断是否为缺失值（如 NaN 或 None） | median      | 中位数                                                    |
| sum()  | 求和，自动忽略缺失值               | mode()      | 众数（可返回多个）                                        |
| mean() | 平均值                             | quantile(q) | 分位数，q 取 0~1 之间                                     |
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

### 一般函数

```python
# 常见函数
import pandas as pd
import numpy as np

s = pd.Series([10,2,np.nan,None,3,4,5], index=['A','B','C','D','E','F','G'], name='data')
print(s)

print(s.head())
print(s.tail())


'''
A    10.0
B     2.0
C     NaN
D     NaN
E     3.0
F     4.0
G     5.0
Name: data, dtype: float64
A    10.0
B     2.0
C     NaN
D     NaN
E     3.0
Name: data, dtype: float64
C    NaN
D    NaN
E    3.0
F    4.0
G    5.0
Name: data, dtype: float64
'''
```

```python
# 查看所有的描述性信息
print(s.describe())


'''
count     5.000000
mean      4.800000
std       3.114482
min       2.000000
25%       3.000000
50%       4.000000
75%       5.000000
max      10.000000
Name: data, dtype: float64
'''
```

```python
# 获取元素个数（忽略缺失值）
print(s.count())


'''
5
'''
```

```python
# 获取索引
print(s.keys()) # 方法
print(s.index) # 属性


'''
Index(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='object')
Index(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='object')
'''
```

```python
# 判断是否是缺失值
print(s.isna())


'''
A    False
B    False
C     True
D     True
E    False
F    False
G    False
Name: data, dtype: bool
'''
```

```python
# 判断是否存在
print(s.isin([4,5,6]))
print(s > 6) # 注意比较上述两者的区别


'''
A    False
B    False
C    False
D    False
E    False
F     True
G     True
Name: data, dtype: bool
A     True
B    False
C    False
D    False
E    False
F    False
G    False
Name: data, dtype: bool
'''
```

### 统计函数

```python
# 统计函数
print('平均值：', s.mean())
print('方差：', s.var())
print('标准差：', s.std())
print('中位数：', s.median())
s['H'] = 2
print('众数：', s.mode()) # 返回Series
print('分位数（0.5）：', s.quantile(0.5)) # 注意和np.percentile的区别
print('最大值：', s.max())
print('最小值：', s.min())
print('随机取样：\n', s.sample(2))
print('元素数量统计：\n', s.value_counts()) # 里面加参数，可以显示占比


'''
平均值： 4.333333333333333
方差： 9.066666666666666
标准差： 3.011090610836324
中位数： 3.5
众数： 0    2.0
Name: data, dtype: float64
分位数（0.5）： 3.5
最大值： 10.0
最小值： 2.0
随机取样：
 F    4.0
H    2.0
Name: data, dtype: float64
元素数量统计：
 data
2.0     2
10.0    1
3.0     1
4.0     1
5.0     1
Name: count, dtype: int64
'''
```

```python
# 去重：drop_duplicates() 与 unique()
# 两者返回类型不同，unique()返回Numpy类型数组(ndarray)，而drop_duplicates()返回新的Series()
print(s.drop_duplicates())
print(s.unique())


'''
A    10.0
B     2.0
C     NaN
E     3.0
F     4.0
G     5.0
Name: data, dtype: float64
[10.  2. nan  3.  4.  5.]
'''
```

### 排序函数

```python
# 排序 值、索引（默认升序）
# 排序时默认NA值排在最后
s.sort_index()
s.sort_values()

# 降序
s.sort_index(ascending=False)
s.sort_values(ascending=False)

# 将NA值排在前面
s.sort_values(na_position='first')

# 重置索引
s.sort_values().reset_index(drop=True)

# 对于较大的Series，如果只取排序中最大的一部分或最小的一部分结果，可以使用 nlargest(n) 和 nsmallest(n)
top3 = s.nlargest(3) # 等效操作：s.sort_values(ascending=False).head(3)
```

## 综合案例

### 学生成绩统计

创建一个包含10名学生数学成绩的Series，成绩范围在50-100之间。

计算平均分、最高分、最低分，并找出高于平局分的学生人数。

```python
import numpy as np
import pandas as pd

# np.random.seed(42)
# math_scores = pd.Series(np.random.randint(50,101,10), index=['学生'+str(i) for i in range(1,11)])

# 注意：np.random.seed(42) 和 np.random.default_rng(seed=42) 虽然都是设置随机种子为 42，但是生成随机数的结果不同
rng = np.random.default_rng(seed=42)
values = rng.integers(50,101,size=10)
indexes = (f'学生{i}' for i in range(1,11))
math_scores = pd.Series(values, indexes, name='math_scores')
print(math_scores)

avg = math_scores.mean()
print('平均分：', avg)
print('最高分：', math_scores.max())
print('最低分：', math_scores.min())
# greater_avg = math_scores[math_scores > avg]
# print('高于平均分的学生人数：', greater_avg.size)
print('高于平均分的学生人数：', len(math_scores[math_scores > avg]))


'''
学生1     54
学生2     89
学生3     83
学生4     72
学生5     72
学生6     93
学生7     54
学生8     85
学生9     60
学生10    54
Name: math_scores, dtype: int64
平均分： 71.6
最高分： 93
最低分： 54
高于平均分的学生人数： 6
'''
```

### 温度数据分析

给定某城市一周每天的最高温度Series，完成以下任务：

- 找出温度超过30度的天数
- 计算平均温度
- 将温度从高到低排序
- 找出温度变化最大的两天

temperatures = pd.Series([28, 31, 29, 32, 30, 27, 33], index=['周一','周二','周三','周四','周五','周六','周日'])

```python
temps = pd.Series([28, 31, 29, 32, 30, 27, 33], index=['周一','周二','周三','周四','周五','周六','周日'])
higher_temps = temps[temps > 30]
print(f'温度超过30度的天数：{len(higher_temps.keys())}')
print(f'平均温度：{temps.mean()}')
print(f'温度从高到低排序：\n{temps.sort_values(ascending=False)}')
diffs = temps.diff().abs() # 计算Series的变化值
print(f'温度变化最大的两天：{diffs.sort_values(ascending=False).keys()[:2].tolist()}')

'''
温度超过30度的天数：3
平均温度：30.0
温度从高到低排序：
周日    33
周四    32
周二    31
周五    30
周三    29
周一    28
周六    27
dtype: int64
温度变化最大的两天：['周日', '周二']
'''
```

### 股票价格分析

给定某股票连续10个交易日的收盘价Series：

- 计算每日收益率（当日收盘价/前日收盘价 - 1）
- 找出收益率最高和最低的日期
- 计算波动率（收益率的标准差）

prices = pd.Series([102.3, 103.5, 105.1, 104.8, 106.2, 107.0, 106.5, 108.1, 109.3, 110.2], index=pd.date_range('2023-01-01',periods=10))

```python
prices = pd.Series([102.3, 103.5, 105.1, 104.8, 106.2, 107.0, 106.5, 108.1, 109.3, 110.2], 
          index=pd.date_range('2025-8-15',periods=10))
return_rate = prices.pct_change()
print(f'每日收益率：\n{return_rate}')
print(f'收益率最高的日期：{return_rate.idxmax()}')
print(f'收益率最低的日期：{return_rate.idxmin()}')
print(f'波动率：{return_rate.std()}')


'''
每日收益率：
2025-08-15         NaN
2025-08-16    0.011730
2025-08-17    0.015459
2025-08-18   -0.002854
2025-08-19    0.013359
2025-08-20    0.007533
2025-08-21   -0.004673
2025-08-22    0.015023
2025-08-23    0.011101
2025-08-24    0.008234
Freq: D, dtype: float64
收益率最高的日期：2025-08-17 00:00:00
收益率最低的日期：2025-08-21 00:00:00
波动率：0.007373623845361105
'''
```

