---
layout: post
title: 矩阵基础和求导
date: 2018-08-11 14:01
tags: [matrix]
categories: Math
description: "矩阵的基础和求导相关知识笔记，包括矩阵常规的性质、不同类型的矩阵、矩阵以及向量的求导"
mathjax: true
---

* content
{:toc}
矩阵的基础和求导相关知识笔记，包括矩阵常规的性质、不同类型的矩阵、矩阵以及向量的求导。<!--more-->

**Cham's Blog 首发原创**



### 矩阵转置

$\left(A^{\mathrm {T} }\right)^{\mathrm {T} }=A$

${(cA)^{\mathrm {T} }=cA^{\mathrm {T} }}$

${\det(A^{\mathrm {T} })=\det(A)} $

${\left(AB\right)^{\mathrm {T} }=B^{\mathrm {T} }A^{\mathrm {T} } }$

${(A+B)^{\mathrm {T} }=A^{\mathrm {T} }+B^{\mathrm {T} }} $

补充：满足结合律 $(AB)C=A(BC)$、分配率 $(A+B)C=AC+BC$，不满足交换律 

### 矩阵的秩

矩阵的最高阶非零子式的阶数，可逆矩阵的秩等于其阶数，可逆矩阵又称满秩矩阵，不可逆的方阵成为奇异矩阵
计算方法：`rank(A)`

### 矩阵的行列式

矩阵的行列式等于其特征值的乘积，$\| A\|=\prod\limits_i\lambda_i$， matlab 计算方法 `det(A)`

### 矩阵的逆

$AA^{-1}=A^{-1}A=I$, 要求 $A$ 是方阵，且 $\text{det}(A)\neq 0$

$A(^{-1})^{-1}=A$

$(AB)^{-1}=B^{-1}A^{-1}$

$(A^{-1})^T=(A^T)^{-1}$

非奇异矩阵(nonsingular) = 可逆矩阵(reversible) = 满秩矩阵(full rank)
判断矩阵 A (m阶) 可逆 MATLAB: `rank(A)=m` 或者 `det(A)！=0`，对称矩阵不一定可逆

### 对称矩阵

$A=A^T$，定义知对称矩阵一定为方阵，而且位于主对角线对称位置上的元素必对应相等

### 正交矩阵

$A^{-1}=A^T, AA^T=I$

具有以下性质：
1) $A^T$是正交矩阵
2) $A$ 的各行是单位向量且两两正交
3) $A$ 的各列是单位向量且两两正交

### 正定矩阵

对称而且特征值大于0的矩阵，所以正定矩阵一定是对称的，正定矩阵的特征值分解得到的特征向量是无关的。

判断矩阵 A 是否正定 MATLAB: `A=A' && eig(A)>0`

### 矩阵对角化

对于可逆方阵 $R$，$\exists\ U$，使得 $U^{-1}RU=\Lambda$，则 $R$ 相似于 $\Lambda$，两者具有相同的特征值。

注意 MATLAB 里的 `diag(R)` 函数，是取方阵 $R$ 的对角元素组成的向量。当 $R=AA^T\in\mathbb{R}^{d\times d}$ 时，向量 $\text{diag}(R)$ 中的元素代表某个特征在所有样本上的方差，$\text{diag}(R)/\text{tr}(R)$ 代表归一化后的方差。

### 广义特征值分解

```matlab
[v,d]=eig(A,B)
```

注意：即便广义特征值分解之后的特征值全大于 0，特征向量之间是相关的，因为 $A/B$ 不一定正定，主要原因在于 $A$ 和 $B$ 正定不能保证 $AB$ 或者 $A/B$ 是对称的，更不用说正定，即便特征值之和全大于 0，也不是正定，也就是说特征向量之间还是相关的。

### F范数及其展开

$$
\| A \|_F =\| A^T \|_F= \sqrt { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \left| a _ { i j } \right| ^ { 2 } } = \sqrt { \operatorname { tr } \left( A ^ { T } A \right) } = \sqrt { \sum _ { i = 1 } ^n \sigma_i^2}= \sqrt { \sum _ { i = 1 } ^n a _ { i,i } ^ { 2 } }
$$

类似的，F范数可按二范数展开，再进行取迹操作，同时也易化成二次型的形式

$$
\begin{align}
\| A-B \|_F^2&=\operatorname{tr}[(A-B)^T(A-B)]=\operatorname{tr}(A^TA-B^TA-A^TB+B^TB)\\&=\operatorname{tr}(\left[ \begin{array} {c c} {A^T}&{B^T} \end{array} \right] \left[ \begin{array} {c c} I&{-I} \\ {-I}&I \end{array} \right] \left[ \begin{array} {l} A \\ B \end{array} \right])
\end{align}
$$

### 二次型矩阵

$$
\mathrm{x}^TA\mathrm{x}=\Sigma_{i=1}^{n} \Sigma_{j=1}^{n}A_{ij}x_ix_j=\mathrm{x}^TA^T\mathrm{x}\\
l = (y-Hx)^2=\|y-Hx\|_F^2= (y-Hx)^T(y-Hx) \\
= y^Ty - y^THx - x^TH^Ty + x^TH^THx
$$

向量不说明都是列向量，范数的计算结果是标量。对于 $m\times n$ 的样本 $X$，$m$ 为样本数，$n$ 为特征数，$X^TX/(m-1)$ 是协方差矩阵，$X^TX$ 是总体散度矩阵。二范数平方对 $X$ 求导：

$$
\frac {\partial l} {\partial x}=2H^T(Hx-y)
$$

类似求导：

$$
\frac {\partial AX} {\partial X}=A^T，\ \frac {\partial X^TA} {\partial X}=A，\frac {\partial AX^T} {\partial X}=A ,\ \ \frac {\partial XA^T} {\partial X}=A^T\\
\frac {\partial XX^T} {\partial X}=2X, \quad \frac {\partial X^TAX} {\partial X}=(A+A^T)X, \quad \frac {\partial A^TX^TXA} {\partial X}=2XAA^T\\
\frac {\partial X^TA^TAX+b^TAX } {\partial X}=2 A^TAX+A^Tb\\
\frac {\partial \sigma^2(X-\mu)^T \Sigma^{-1} (X-\mu)}{\partial X}=2\sigma^2\Sigma^{-1}(X-\mu)\\
$$

###  概率的链式法则

$$
\begin{align}
P(a,b)&=P(a\mid b)P(b)\\
P(a,b,c)&=P(a\mid b,c)P(b,c)=P(a\mid b,c)P(b\mid c)P(c)
\end{align}
$$


### 矩阵的迹

方阵对角线元素之和，也是特征值之和，满足性质：对尺寸相同的矩阵$A$ , $B$，$\text{tr}(A^TB) = \sum_{i,j}A_{ij}B_{ij}$, 即 $\text{tr}(A^TB)$ 是矩阵$A$, $B$ 的内积。$trace(A)=\sum A_{ii}$

1. **标量套上迹**：$a = \text{tr}(a)$
2. **转置**：$\mathrm{tr}(A^T) = \mathrm{tr}(A)$
3. **线性**：$\text{tr}(A\pm B) = \text{tr}(A)\pm \text{tr}(B)$
4. **矩阵乘法交换**：$\text{tr}(AB) = \text{tr}(BA)$，其中$A$与$B^T$尺寸相同，两侧都等于$\sum_{i,j}A_{ij}B_{ji}$。
5. **矩阵乘法/逐元素乘法交换**：$\text{tr}(A^T(B\odot C)) = \text{tr}((A\odot B)^TC)$，其中$A, B, C$尺寸相同，两侧都等于$\sum_{i,j}A_{ij}B_{ij}C_{ij}$

### 标量对矩阵向量求导（从元素角度）

1. **标量对向量的微分和导数关联**：$df = \sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i = \frac{\partial f}{\partial \boldsymbol{x}}^T d\boldsymbol{x}$
2. **标量对矩阵的微分和导数关联**： $df = \text{tr}\left(\frac{\partial f}{\partial X}^T dX\right)$, 这里的 $\frac{\partial f}{\partial X}^T$ 不同于 $\frac{\partial f^T}{\partial X}$ 
3. **加减法**：$d(X\pm Y) = dX \pm dY$
4. **矩阵乘法**：$d(XY) = dX Y + X dY$
5. **转置**：$d(X^T) = (dX)^T$
6. **迹**：$d\text{tr}(X) = \text{tr}(dX)$
7. **逆**：$dX^{-1} = -X^{-1}dX X^{-1}$。此式可在 $XX^{-1}=I$ 两侧求微分来证明。
8. **行列式**：$d\|X\| = \text{tr}(X^{\ast}dX)$ ，其中$X^{\ast}$表示 $X$ 的伴随矩阵，在X可逆时又可以写作 $d\|X\|= \|X\|\text{tr}(X^{-1}dX)$
9. **逐元素乘法**：$d(X\odot Y) = dX\odot Y + X\odot dY$，$\odot$表示尺寸相同的矩阵$X,Y$逐元素相乘。
10. **逐元素函数**：$d\sigma(X) = \sigma'(X)\odot dX$ ，举个例子，$d \sin(X) = [\cos x_1 dx_1, \cos x_2 dx_2] = \cos(X)\odot dX$，$X=[x_1, x_2]$



**例：**$f = \boldsymbol{a}^T X\boldsymbol{b}$，求$\frac{\partial f}{\partial X}$。其中 $\boldsymbol{a}$ 是 $m×1$ 列向量，$X$是 $m\times n$ 矩阵，$\boldsymbol{b}$ 是 $n×1$ 是标量。
> 解：先使用矩阵乘法法则求微分：$df = \boldsymbol{a}^T dX\boldsymbol{b}$ （其中和 $X$ 无关的微分项为0，
> 再套上迹做矩阵乘法交换：$df = \text{tr}(\boldsymbol{a}^TdX\boldsymbol{b}) = \text{tr}(\boldsymbol{b}\boldsymbol{a}^TdX)$，这里根据$\text{tr}(AB) = \text{tr}(BA)$交换了 $\boldsymbol{a}^TdX$ 与 $\boldsymbol{b}$。这里 $\text{tr}(\boldsymbol{a}^TdX\boldsymbol{b})=\text{tr}\left(\frac{\partial f}{\partial X}^T dX\right)$ ，但前一项内部乘积为标量，无法约去$dX$，所以需要变成 $ \boldsymbol{b}\boldsymbol{a}^TdX $。对照导数与微分的联系 $df = \text{tr}\left(\frac{\partial f}{\partial X}^T dX\right)$，得到 $\frac{\partial f}{\partial X} = (\boldsymbol{b}\boldsymbol{a}^T)^T= \boldsymbol{a}\boldsymbol{b}^T$



**关于矩阵求导的一些tricks:**

公式 $df = \text{tr}\left(\frac{\partial f}{\partial X}^T dX\right)$不是一成不变的，如果 $f$ 求导的对象是一个行向量，那么可以去掉trace，同时很多时候对于一个复杂的矩阵组合求全微分，注意分析矩阵的维度，可以实现最大化简化问题。



### 矩阵向量标量之间求导（从整体角度）

如果计算标量和矩阵、向量之间的导数直接用上面一种方法更简单，但是从整体出发的角度可以概括向量和向量，矩阵和矩阵、标量和向量，标量和矩阵等多种情况，采用统一的定义，可解释性更强。结论：

1. **标量对向量**的导数与微分的联系是 $df = \nabla_{\boldsymbol{x}}^T f d\boldsymbol{x}$
2. **标量对矩阵**的导数与微分的联系是 $df = \mathrm{tr}(\nabla_X^T f dX)$，先对 $f$ 求微分，再使用迹技巧求导数
3. **向量对向量**的导数与微分的联系是 $d\boldsymbol{f} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}^Td\boldsymbol{x}$
4. **矩阵对矩阵**的导数与微分的联系是 $\mathrm{vec}(dF) = \frac{\partial F}{\partial X}^T \mathrm{vec}(dX)$，先对 $F$ 求微分，再使用向量化技巧求导数


**结论中的两个 tricks**
第一将 $\frac{\partial f}{\partial X}^T$简记为 $\nabla_X^T f$，为了计算矩阵之间的导数，在当前定义下，标量对矩阵（$m\times n$）求导的结果是 $mn\times 1$，即 $\frac{\partial f}{\partial X}=\mathrm{vec}(\nabla_X f)$；第二向量化技巧是把矩阵转换成向量（按列拼接成行向量），然后利用类似向量之间的微分导数关系求导。


**具体展开：**
向量之间的导数，$\boldsymbol{f}$ (p×1) 对向量 $\boldsymbol{x}$ (m×1) 的导数 $\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}$ (m×p)，注意这边相当于定义了矩阵除法的维度变换过程，可以迁移到矩阵的偏导上去， 矩阵 $F$ ($p\times q$) 对矩阵 $X$ ($m\times n$) 的导数 $\frac{\partial F}{\partial X} = \frac{\partial \mathrm{vec}(F)}{\partial \mathrm{vec}(X)}$ (mn×pq) 


**关于向量化的结论**

1. **线性**： $\mathrm{vec}(A+B) = \mathrm{vec}(A) + \mathrm{vec}(B)$ 。
2. $(A\otimes B)^T = A^T \otimes B^T$ 
3. $\mathrm{vec}(\boldsymbol{ab}^T) = \boldsymbol{b}\otimes\boldsymbol{a}$， $\otimes$ 表示 Kronecker 积，假设$A(m×n)$与$B(p×q)$，$A\otimes B = [A_{ij}B] (mp×nq)$
4. **矩阵乘法**： $\mathrm{vec}(AXB) = (B^T \otimes A) \mathrm{vec}(X)$
5. **转置**： $\mathrm{vec}(A^T) = K_{mn}\mathrm{vec}(A)$ ，$A$是$m×n$矩阵，其中 $K_{mn} (mn×mn)$是交换矩阵(commutation matrix)。
6. **逐元素乘法**： $\mathrm{vec}(A\odot X) = \mathrm{diag}(A)\mathrm{vec}(X)$ ，其中 $\mathrm{diag}(A)$ (mn×mn)是用A的元素（按列优先）排成的对角阵。
7. **标量对矩阵的二阶导**数，定义为 $\nabla^2_X f = \frac{\partial^2 f}{\partial X^2} = \frac{\partial \nabla_X f}{\partial X} (mn×mn)$ 



**例：** $F = AX$，$X$ 是 $m×n$ 矩阵，求 $\frac{\partial F}{\partial X}$ 。
> 先求微分 $dF=AdX$，再做向量化，使用矩阵乘法的技巧，注意在$dX$右侧添加单位阵：$\mathrm{vec}(dF) = \mathrm{vec}(AdX) = (I_n\otimes A)\mathrm{vec}(dX)$，对照导数与微分的联系得到 $\frac{\partial F}{\partial X} = I_n\otimes A^T$。




矩阵求导部分是下面两篇文章的笔记：
1. [矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)
2. [矩阵求导术（下）](https://zhuanlan.zhihu.com/p/24863977)