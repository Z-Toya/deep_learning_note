# numpy

## 介绍

numpy是Python中科学计算的基础包。

它是一个Python库，提供多维数组对象、各种派生对象(例如掩码数组和矩阵)以及用于对数组进行快速操作的各种方法，包括数学、逻辑、形状操作、排序、选择、1/0、离散傅里叶变换、基本线性代数、基本统计运算、随机模拟等等。

numpy的部分功能如下：

- ndarray是一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。
- 用于对整且数据进行快速运算的标准数学函数(无需编写循环)。
- 用于读写磁盘数据的工具以及用于操作内存映射文件的工具。
- 线性代数、随机数生成以及傅里叶变换功能。
- 用于集成由C、C++、Fortran等语言编写的代码的API。

# ndarray

## 特性

- 多维性：支持0维（标量）、1维（向量）、2维（矩阵）及更高维数组。

```python
arr = np.array([1,2,3,4,5])
print(arr)
print('arr的维度：', arr.ndim)
```

- 同质性：所有元素类型必须一致（通过dtype指定）。

```python
arr = np.array([1, 'hello'])
print(arr)

# ['1' 'hello']
```

> 不同的数据类型会被强制转换成相同的数据类型

- 高效性：基于连续内存块存储，支持向量化运算。

## 属性

- shape：数组的形状
- ndim：维度数量
- size：总元素个数
- dtype：元素类型

- T：转置
- itemsize：单个元素占用的内存字节数
- nbytes：数组总内存占用量
- flags：内存存储方式：是否连续存储（高级优化）

## 创建方式

### 1. 基础构造

```python
arr = np.array([1,2,3]) # 传入的参数是列表
print(arr.ndim)
print(arr)
```

```
# copy:会开辟新内存
arr1 = np.copy(arr)
print(arr1)
```

### 2. 预定义形状

```python
# 全0
arr1 = np.zeros((2,),dtype=int) # 一维
arr2 = np.zeros((2,3),dtype=int) # 二维
print(arr)
```

```python
# 全1
arr = np.ones((5,),dtype=int)
print(arr)
```

 ```python
 # 未初始化
 arr = np.empty((2,3))
 print(arr)
 ```

```python
# 全部填充同一值
arr = np.full((3,4),2025)
print(arr)
```

```python
# 形状和其他多维数组形状相同
arr1 = np.zeros_like(arr)
print(arr1)
```

### 3. 基于数值范围生成

```python
# 等差数列
arr = np.arange(1,10,1) # start,end,step
print(arr)

# [1 2 3 4 5 6 7 8 9]

arr = np.arange(10)
print(arr)

# [0 1 2 3 4 5 6 7 8 9]
```

```python
# 等间隔数列
arr = np.linspace(1,10,5)
print(arr)

# [ 1.    3.25  5.5   7.75 10.  ]
```

```python
# 对数间隔数列
arr = np.logspace(0,4,2,base=2)
print(arr)

arr = np.logspace(0,4,3,base=2)
print(arr)

# [ 1. 16.]
# [ 1.  4. 16.]
```

### 4. 特殊矩阵生成

> 0维：标量
>
> 1维：向量
>
> 2维：矩阵
>
> 3维：张量

```python
# 特殊矩阵
# 单位矩阵
arr = np.eye(3,4,dtype=int) # 不加后面的4就是方阵
print(arr)

# 对角矩阵
arr = np.diag([1,2,3])
print(arr)

'''
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]]
 
[[1 0 0]
 [0 2 0]
 [0 0 3]]
'''
```

### 5. 随机数组生成

```python
# 随机数组的生成
# 生成0到1之间的随机浮点数（均匀分布）
arr = np.random.rand(2,3)
print(arr)

# 生成指定范围区间内的随机浮点数（均匀分布）
arr = np.random.uniform(3,6,(2,3))
print(arr)

'''
[[0.29965968 0.81254775 0.00289498]
 [0.06621712 0.92451074 0.88306852]]
 
[[5.96293233 3.92376248 5.42026115]
 [3.2786077  5.5069818  3.91766106]]
'''
```

```python
# 生成指定范围区间的随机整数
arr = np.random.randint(3,30,(2,3))
print(arr)

'''
[[29 13  4]
 [ 3  7 28]]
'''
```

```python
# 生成随机数列（正态分布）
arr = np.random.randn(2,3)
print(arr)

'''
[[-1.12423669 -0.20021014 -0.72611745]
 [-1.38210366 -0.58444512 -1.6302119 ]]
'''
```

```python
# 设置随机种子
np.random.seed(20)
arr = np.random.randint(1,10,(2,5))
print(arr)

'''
[[4 5 7 8 3]
 [1 7 9 6 4]]
'''
```

### 6. 高级构造方法

- np.array()
- np.loadtxt()
- np.fromfunction()

## 数据类型

| 数据类型                                                     | 说明                                                       |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| bool                                                         | 布尔类型                                                   |
| int8、uint8<br />int16、uint16<br />int32、uint32<br />int64、uint64 |                                                            |
| float16<br />float32<br />float64                            | 半精度浮点型<br />单精度浮点型<br />双精度浮点型           |
| complex64<br />complex128                                    | 用两个32位浮点数表示的复数<br />用两个64位浮点数表示的复数 |

## 索引与切片

- 基本索引

- 行/列切片
- 连续切片
- 使用 slice 函数
- 布尔索引

```python
# 一维数组的索引与切片
arr = np.random.randint(1,100,20)
print(arr)
print(arr[10])
print(arr[:]) # 获取全部的数据
print(arr[2:5]) # start:end+1 作闭右开区间
print(arr[slice(2,15,3)]) # start,end,step
print(arr[ (arr>10) & (arr<70) ]) # 布尔索引

'''
[64 25 78 70 85 90 29 31 33 53 23 77 85 48 33 49 51 35 82 38]
23
[64 25 78 70 85 90 29 31 33 53 23 77 85 48 33 49 51 35 82 38]
[78 70 85]
[78 90 33 77 33]
[64 25 29 31 33 53 23 48 33 49 51 35 38]
'''
```

```python
# 二维数组的索引与切片
arr = np.random.randint(1,100,(4,8))
print(arr)
print(arr[1,3])
print(arr[:,:]) # 获取全部数据
print(arr[1,2:5])
print(arr[arr>50]) # 返回一维的结果
print(arr[2][arr[2]>50]) # 返回第三行大于50的元素，注意嵌套
print(arr[:,3]) # 返回第四列

'''
[[93 51 50 10 26 53 62 38]
 [29 72 50 30  5 69 84 74]
 [94 64 33 27 93 38 43 53]
 [84 11 31 80 61 61 85 51]]
30
[[93 51 50 10 26 53 62 38]
 [29 72 50 30  5 69 84 74]
 [94 64 33 27 93 38 43 53]
 [84 11 31 80 61 61 85 51]]
[50 30  5]
[93 51 53 62 72 69 84 74 94 64 93 53 84 80 61 61 85 51]
[94 64 93 53]
[10 30 27 80]
'''
```

## 运算

### 一维运算

```python
# 算术运算
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)

'''
[5 7 9]
[-3 -3 -3]
[ 4 10 18]
[0.25 0.4  0.5 ]
[  1  32 729]
'''
```

```python
# 数组与标量的算术运算
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a + 3)
print(a * 3)

'''
[[ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
[[ 3  6  9]
 [12 15 18]
 [21 24 27]]
'''
```

### 广播机制

```python
# 广播机制：1.获取形状 2.是否可以广播
# 广播条件：所有维度相同或者有一个维度是1
a = np.array([1,2,3]) # 1*3
b = np.array([[4],[5],[6]]) # 3*1
print(a + b)
'''
分析：
a
1 2 3
1 2 3
1 2 3
b
4 4 4
5 5 5
6 6 6
'''

'''
[[5 6 7]
 [6 7 8]
 [7 8 9]]
'''
```

```python
# 不满足条件示例
a = np.array([1,2,3]) # 1*3
b = np.array([4,5]) # 1*2
print(a + b)

'''
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipython-input-1651296882.py in <cell line: 0>()
      1 a = np.array([1,2,3]) # 1*3
      2 b = np.array([4,5]) # 1*2
----> 3 print(a + b)

ValueError: operands could not be broadcast together with shapes (3,) (2,) 
'''
```

### 矩阵运算

```python
# 矩阵运算
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[4,5,6],[7,8,9],[1,2,3]])
print(a * b) # 对应元素相乘
print(a @ b) # 矩阵乘法

'''
[[ 4 10 18]
 [28 40 54]
 [ 7 16 27]]
[[ 21  27  33]
 [ 57  72  87]
 [ 93 117 141]]
'''
```

## 常用函数

![image-20250808165721007](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20250808165721007.png)

### 基本数学函数

```python
# 基本数学函数
# 计算平方根
print(np.sqrt(9)) # 标量
print(np.sqrt([1,4,9])) # 列表
arr = np.array([1,25,81]) # ndarray
print(np.sqrt(arr))

# 计算指数
print(np.exp(1))

# 计算自然对数
print(np.log(2.71))

# 计算正弦值、余弦值
print(np.sin(np.pi/2))
print(np.cos(np.pi))

# 计算绝对值
arr = np.array([-1, 1, 2, -3])
print(np.abs(arr))

# 计算a的b次幂
print(np.power(arr,2))

# 四舍五入
print(np.round([3.2, 4.5, 4.51, 8.1, 9.6])) # 注意4.5和4.51的结果

# 向上取整，向下取整
arr = np.array([1.6, 25.1, 81.7])
print(np.ceil(arr))
print(np.floor(arr))

# 检测缺失值NaN
np.isnan([1,2,np.nan,3])

'''
3.0
[1. 2. 3.]
[1. 5. 9.]
2.718281828459045
0.9969486348916096
1.0
-1.0
[1 1 2 3]
[1 1 4 9]
[ 3.  4.  5.  8. 10.]
[ 2. 26. 82.]
[ 1. 25. 81.]
array([False, False,  True, False])
'''
```

### 统计函数

求和、计算平均值、计算中位数、标准差、方差

查找最大值、最小值

计算分位数、累计和、累计差

```python
# 统计函数
arr = np.random.randint(1,20,8)
print(arr)

# 求和
print(np.sum(arr))

# 计算平均值
print(np.mean(arr))

# 计算中位数
# 奇数：排序后中间的数值
# 偶数：中间的两个数的平均值
print(np.median(np.median([4,1,2])))
print(np.median([1,2,4,8]))

# 计算标准差、方差
# 1,2,3 的平均值 2
# ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 0.666
print(np.var([1,2,3]))
print(np.std([1,2,3]))
print(np.var(arr))
print(np.std(arr))


'''
[19  2 12  1 11 15  4 18]
82
10.25
2.0
3.0
0.6666666666666666
0.816496580927726
44.4375
6.6661458129866915
'''
```

```python
# 均值与方差对比
# 衡量数据稳定性（好坏）---方差
arr1 = np.array([1,2,1,2,1,1,1,2])
arr2 = np.array([1,0,3,0,0,0,4,3])
print(np.mean(arr1))
print(np.mean(arr2))
print(np.var(arr1))
print(np.var(arr2))


'''
1.375
1.375
0.234375
2.484375
'''
```



```python
# 计算最大值、最小值
print(arr)
print(np.max(arr), np.argmax(arr))
print(np.min(arr), np.argmin(arr))


'''
[19  2 12  1 11 15  4 18]
19 0
1 3
'''
```

```python
# 分位数
# 中位数
np.random.seed(0)
arr = np.random.randint(0,100,4)
print(arr)
# ---------------------
# 44  47  64  67
print(np.median(arr))
print(np.percentile(arr,50)) # 分位数，等效于median
print(np.percentile(arr,25))
# 计算策略：
# 数轴上一共有3段
# 0.25*3 = 0.75 (在第一段)，所以加上第一段起始部分
# (47-44)*0.75 + 44 = 46.25
print(np.percentile(arr,80))
# 3*0.8 = 2.4 (在第三段)
# (67-64)*0.4 + 64 = 65.2  


'''
[44 47 64 67]
55.5
55.5
46.25
65.2
'''
```

```python
# 累积和、累积积
arr = np.array([1,2,3])
print(np.sum(arr))
print(np.cumsum(arr)) # 可以理解为前缀和
print(np.cumprod(arr))


'''
6
[1 3 6]
[1 2 6]
'''
```

### 比较函数

比较是否大于、小于、等于

逻辑与、或、非

检查数组中是否有一个True，是否所有都为True，自定义条件

```python
# 比较函数
# 是否大于
print(np.greater([3,4,5,6,7],4))
# 是否小于
print(np.less([3,4,5,6,7,8],4))
# 是否等于
print(np.equal([3,4,5,6,7,8],4))
print(np.equal([3,4,5],[4,4,4]))


'''
[False False  True  True  True]
[ True False False False False False]
[False  True False False False False]
[False  True False]
'''
```

```python
# 逻辑运算
# 逻辑与
print(np.logical_and([1,0],[1,1]))
print(np.logical_or([0,0],[1,0]))
print(np.logical_not([1,0]))


'''
[ True False]
[ True False]
[False  True]
'''
```

```python
# 检查元素是否至少有一个元素为True
print(np.any([0,0,0,0,0,0]))
# 检查元素是否全部为True
print(np.all([1,1,1,0,0,0]))


'''
False
False
'''
```

```python
# 自定义条件
# np.where(条件，符合条件，不符合条件的)
arr = np.array([1,2,3,4,5])
print(np.where(arr<3,arr,0)) # 做二分类、三分类

# eg：
arr = np.array([1,2,3,4,5])
print(np.where(arr<3,1,0)) 

score = np.random.randint(50,100,20)
print(score)
print(np.where(score>=60,'及格','不及格'))

# 三分类（嵌套）
print(np.where(
    score<60,'不及格',np.where(
      score<70,'及格',np.where(
          score<80,'良好','优秀'
      )
  )
))


'''
[1 2 0 0 0]
[1 1 0 0 0]
[54 97 53 62 86 90 64 65 70 85 73 65 63 71 98 99 55 91 85 50]
['不及格' '及格' '不及格' '及格' '及格' '及格' '及格' '及格' '及格' '及格' '及格' '及格' '及格' '及格'
 '及格' '及格' '不及格' '及格' '及格' '不及格']
['不及格' '优秀' '不及格' '及格' '优秀' '优秀' '及格' '及格' '良好' '优秀' '良好' '及格' '及格' '良好'
 '优秀' '优秀' '不及格' '优秀' '优秀' '不及格']
'''
```

```python
# np.select(条件，返回的结果)，和where做对比，使用select的方式更方便简洁
print(np.select([score>=80,score>=70,score>=60,score<60],
        ['优秀','良好','及格','不及格'],
        default='未知'))


'''
['不及格' '优秀' '不及格' '及格' '优秀' '优秀' '及格' '及格' '良好' '优秀' '良好' '及格' '及格' '良好'
 '优秀' '优秀' '不及格' '优秀' '优秀' '不及格']
'''
```

### 排序函数

```python
# 排序函数
np.random.seed(0)
arr = np.random.randint(1,100,20)
print(arr)
# arr.sort()
# print(arr)
print(np.sort(np.sort(arr))) # np.sort()不修改arr
print(np.argsort(arr)) # 获取排序前的元素索引
print(arr)


'''
[45 48 65 68 68 10 84 22 37 88 71 89 89 13 59 66 40 88 47 89]
[10 13 22 37 40 45 47 48 59 65 66 68 68 71 84 88 88 89 89 89]
[ 5 13  7  8 16  0 18  1 14  2 15  3  4 10  6  9 17 11 12 19]
[45 48 65 68 68 10 84 22 37 88 71 89 89 13 59 66 40 88 47 89]
'''
```

```python
# 去重函数
print(np.unique(arr)) # 去重的同时还会排序

'''
[10 13 22 37 40 45 47 48 59 65 66 68 71 84 88 89]
[45 48 65 68 68 10 84 22 37 88 71 89 89 13 59 66 40 88 47 89]
'''
```

```python
# 数组的拼接
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print(arr1+arr2)
print(np.concatenate((arr1,arr2)))


'''
[5 7 9]
[1 2 3 4 5 6]
'''
```

```python
# 数组的分割
print(np.split(arr,5)) # 必须刚好等分
print(np.split(arr,[6,12,18])) # 不能等分的情况必须分段


'''
[array([45, 48, 65, 68]), array([68, 10, 84, 22]), array([37, 88, 71, 89]), array([89, 13, 59, 66]), array([40, 88, 47, 89])]
[array([45, 48, 65, 68, 68, 10]), array([84, 22, 37, 88, 71, 89]), array([89, 13, 59, 66, 40, 88]), array([47, 89])]
'''
```

```python
# 调整数组的形状
print(np.reshape(arr,[5,4]))


'''
[[45 48 65 68]
 [68 10 84 22]
 [37 88 71 89]
 [89 13 59 66]
 [40 88 47 89]]
'''
```

## 补充

### axis参数



# 案例练习

## 温度数据分析

某城市一周的最高气温（℃）为 [28, 30, 29, 31, 32, 30, 29]

- 计算平均气温、最高气温和最低气温
- 找出气温超过 30 ℃ 的天数

```python
temps = np.array([28, 30, 29, 31, 32, 30, 29])
print(temps)
print('平均气温：', '%.3f'%np.mean(temps))
print('最高气温：', np.max(temps))
print('最低气温：', np.min(temps))
print('气温超过30的天数：', len(temps[temps>30]))
print('气温超过30的天数：', np.sum(np.where(temps>30,1,0)))
print('气温超过30的天数：', np.count_nonzero(temps>30))


'''
[28 30 29 31 32 30 29]
平均气温： 29.857
最高气温： 32
最低气温： 28
气温超过30的天数： 2
气温超过30的天数： 2
气温超过30的天数： 2
'''
```

## 学生成绩统计

某班级 5 名学生的数学成绩为 [85, 90, 78, 92, 88]

- 计算成绩的平均分、中位数和标准差
- 将成绩转化为百分制（假设满分为100）

```python
score = np.array([85, 90, 78, 92, 88])
print('平均分：', np.mean(score))
print('中位数：', np.median(score))
print('中位数：', np.percentile(score,50))
print('标准差：%.3f'% np.std(score))
print(score/10)


'''
平均分： 86.6
中位数： 88.0
中位数： 88.0
标准差：4.883
[8.5 9.  7.8 9.2 8.8]
'''
```

## 矩阵运算

给定矩阵 A = [[1, 2], [3, 4]] 和 B = [[5, 6], [7, 8]]

- 计算 A + B 和 A * B （逐元素乘法）
- 计算 A 和 B 的矩阵乘法（点积）

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print('A + B：')
print(A + B)
print('A 和 B 逐元素相乘：')
print(A * B)
print('A 和 B 矩阵相乘：')
print(A @ B)


'''
A + B：
[[ 6  8]
 [10 12]]
A 和 B 逐元素相乘：
[[ 5 12]
 [21 32]]
A 和 B 矩阵相乘：
[[19 22]
 [43 50]]
'''
```

## 随机数据生成

生成一个 （3,4）的随机整数数组，范围 [0, 10]

- 计算每列的最大值和每行的最小值
- 将数组的所有奇数替换为 -1

```python
np.random.seed(0)
arr = np.random.randint(0,10,[3,4])
# arr = np.random.randint(0,10,(3,4)) # 上述两种括号都可以
# arr = np.random.randint(0,10,12).reshape([3,4])
print(arr)
print('每列的最大值：', np.max(arr,axis=0)) # axis=0 列 axis=1 行
print('每列的最小值', np.min(arr, axis=1))
# 第一种
print(np.where(arr%2==1,-1,arr))
# 第二种
print(np.select([arr%2==1],[-1],default=arr))
# 第三种(会修改arr的值)
arr[arr%2==1] = -1
print(arr)


'''
[[5 0 3 3]
 [7 9 3 5]
 [2 4 7 6]]
每列的最大值： [7 9 7 6]
每列的最小值 [0 3 2]
[[-1  0 -1 -1]
 [-1 -1 -1 -1]
 [ 2  4 -1  6]]
'''
```



## 数组变型

创建一个 1 到 12 的一维数组，并转换为（3, 4）的二维数组

- 计算每行的和每列的平均值
- 将数组展平为一维数组

```python
arr = np.arange(1,13).reshape(3,4)
print(arr)
print('每列的平均值：', np.mean(arr, axis=0))
print('每行的和：', np.sum(arr, axis=1))
# print(arr.reshape(12))
print(np.reshape(arr,(12)))


'''
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
每列的平均值： [5. 6. 7. 8.]
每行的和： [10 26 42]
[ 1  2  3  4  5  6  7  8  9 10 11 12]
'''
```

## 布尔索引

生成一个 （5, 5）的随机数组，范围 [0, 20)

- 找出数组中大于 10 的元素
- 将所有大于 10 的元素替换为0

```python
np.random.seed(1)
arr = np.random.randint(0,20,(5,5))
print(arr)
print(arr[arr>10])
print(np.where(arr>10,0,arr))
# arr[arr>10] = 0
# print(arr)

'''
[[ 5 11 12  8  9]
 [11  5 15  0 16]
 [ 1 12  7 13  6]
 [18  5 18 11 10]
 [14 18  4  9 17]]
[11 12 11 15 16 12 13 18 18 11 14 18 17]
[[ 5  0  0  8  9]
 [ 0  5  0  0  0]
 [ 1  0  7  0  6]
 [ 0  5  0  0 10]
 [ 0  0  4  9  0]]
'''
```

## 统计函数应用

某公司 6 个月的销售额 （万元） 为 [120, 135, 110, 125, 130, 140]

- 计算销售额的总和、均值和方差
- 找出销售额最高的月份和最低的月份

```python
sales = np.array([120, 135, 110, 125, 130, 140])
print('销售额总和：', np.sum(sales))
print('销售额均值：%.3f'% np.mean(sales))
print('销售额方差：%.3f'% np.var(sales))
print('销售额最高月份：', np.argmax(sales) + 1)
print('销售额最低月份：', np.argmin(sales) + 1)


'''
销售额总和： 760
销售额均值：126.667
销售额方差：97.222
销售额最高月份： 6
销售额最低月份： 3
'''
```

## 数组拼接

给定 A = [1, 2, 3] 和 B = [4, 5, 6]

- 将 A 和 B 水平拼接为一个新数组
- 将 A 和 B 垂直拼接为一个新数组

```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
# 水平方向拼接
print(np.hstack((A,B)))
print(np.concatenate((A,B)))

# 垂直方向拼接
# print(np.reshape(C,(2,3)))
print(np.vstack((A,B)))
# 另一种方式是将 A 和 B 升维，从一维变成二维，再通过修改axis参数拼接
# 升维操作通过np.newaxis实现
A1 = A[np.newaxis, :]
B1= B[np.newaxis, :]
print(np.concatenate((A1,B1),axis=0))


'''
[1 2 3 4 5 6]
[1 2 3 4 5 6]
[[1 2 3]
 [4 5 6]]
[[1 2 3]
 [4 5 6]]
'''
```

## 唯一值与排序

给定数组 [2, 1, 2, 3, 1, 4, 3]

- 找出数组中的唯一值并排序
- 计算每个唯一值出现的次数

```python
arr = np.array([2, 1, 2, 3, 1, 4, 3])
# 第一种，可以返回额外参数
# u_arr,counts = np.unique(arr,return_counts=True)
# print(u_arr)
# print(counts)

# 第二种，for循环实现
'''
if 'len' in globals():
  del len
d = []
arr_len = len(u_arr)
for i in range(arr_len):
  d = d+[len(arr[arr==u_arr[i]])]
print(d)
'''
# 第三种
print(np.unique(arr))
print(np.unique_counts(arr))


'''
[1 2 3 4]
UniqueCountsResult(values=array([1, 2, 3, 4]), counts=array([2, 2, 2, 1]))
'''
```

