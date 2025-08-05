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

### 1. 基本索引

### 2. 行/列切片

### 3. 连续切片

### 4. 使用 slice 函数

### 5. 布尔索引