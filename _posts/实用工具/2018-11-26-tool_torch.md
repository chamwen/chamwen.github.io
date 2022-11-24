---
layout: post
title: Pytorch 学习笔记
date: 2018-11-26 9:26
tags: [torch]
categories: 实用工具
description: "关于pytorch基础语法的学习笔记"
mathjax: false
---

* content
{:toc}
关于 pytorch 基础语法的学习笔记，时间 Nov. 26 -- Nov. 27.  <!--more-->

参考：[PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)

### 张量创建和随机采样

```python
x = tr.FloatTensor([[1, 2, 3], [4, 5, 6]]) # 从list构建张量
print(x[1][2])  # tensor的索引
b = tr.ones(5)  # tensor中这种形式是行向量，没有转置
c = tr.arange(1, 5)
tr.randn(4)  # 返回4个从标准正态分布中随机选取的张量
tr.randn(3,2)  # 返回3*2个从标准正态分布中随机选取的张量
tr.randperm(n)  # 返回一个从 0 to n-1 的整数的随机排列
y.copy_(x)  # 将x中的元素复制到y中并返回y; 两个tensor应该有相同shape
```

### 张量的拼接和select

```python
tr.cat((x, y), 0)  # x，y是tensor，0表示行方向叠加类似于[x;y]
indices = tr.LongTensor([0, 2])
tr.index_select(x, 1, indices)  # 列方向上选第0和2列，返回的张量和原张量不共享内存空间
src = tr.Tensor([[4, 3, 5],[6, 7, 8]])
tr.take(src, torch.LongTensor([0, 2, 5]))  # 选出指定位置元素4,5,8
tr.t(src)  # 张量转置
tr.unbind(src, 1)  # 移除张量的第1维
src.view(1,-1)  # 将一个多行的tensor拼接成一行
```

### Pointwise Ops (逐点操作)

```python
tr.add(a,10)  # 张量a逐元素加上10
b.add_(a)  # b+a --> b，结果覆盖b
a = tr.randn(4)
tr.ceil(a)  # 向上取整
tr.mul(a, b)  # a和b逐元素相乘，不要求size格式一样，和a*b一样
tr.mm(a,b)  # 真正的矩阵叉乘
```

### Reduction Ops (归约操作)

```python
tr.cumrod(a,dim=0)  # 累积求积
tr.cumsum(a,dim=0)  # 累积求和
tr.dist(x,y,p)  # x和y的p范数
tr.median()  # 计算中位数
tr.var()  # 计算方差，tr.std()标准差
```

### Comparison Ops (比较操作)

```python
tr.eq(a, b)  # 整体判断a=b
tr.ge(a, b)  # 逐元素判断a=b，反之tr.ne(a, b)
tr.gt(a, b)  # 逐元素判断a>b，反之tr.le(a, b)，tr.lt(a,b)
sorted, indices = torch.sort(x)  # 沿着x的最后一维的方向（2维则是沿着列变化的方向）
```

### Other Operations (其它操作)

```python
tr.tril(a)  # 获得a的下三角（上三角置0），反之tr.triu()
tr.addmm(M, mat1, mat2)  # mat1 和 mat2 的相乘，再加上M
tr.addmv(M, mat, vec)  # 矩阵 mat 和向量 vec 的相乘，再加上M
tr.addr(M, vec1, vec2)  # 向量 vec11 和向量 vec2 的相乘，再加上M
tr.bmm(batch1, batch2)  # 执行保存在 batch1 和 batch2 中的矩阵的批量点乘
tr.dot(tensor1, tensor2)  # 向量之间的点积求和
tr.eig(a)  # 特征值分解
tr.inverse(a)  # 求逆
tr.mm()  # 矩阵和矩阵相乘，对应tr.mv()矩阵和向量
```