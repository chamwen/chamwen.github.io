---
layout: post
title: 协方差矩阵的计算以及实际意义
date: 2018-08-12 21:11
tags: [Covariance,PCA]
categories: Machine-Learning
description: "协方差矩阵的理解，定义、计算、和散度矩阵、相关系数矩阵的关系，协方差矩阵的特征值分解、奇异值分解，以及分析了协方差矩阵和数据结构的联系"
mathjax: true
---

* content
{:toc}
协方差矩阵的理解，定义、计算、和散度矩阵、相关系数矩阵的关系，协方差矩阵的特征值分解、奇异值分解，以及分析了协方差矩阵和数据结构的联系。<!--more-->

**Cham's Blog 首发原创**



## 协方差矩阵的理解

### 协方差以及协方差矩阵的定义和计算

$$
cov(X,Y)=\frac{∑_{i=1}^n(X_i−\bar{X})(Y_i−\bar{Y})}{n−1}
$$

协方差是用来度量变量之间的相关性，协方差矩阵是多个变量组的协方差的组合，在机器学习中，$X$和$Y$是样本的不同的特征维度。

对于数据集 $D\in \mathbb{R^{m\times n}}$，设其共有$m$个样本，每个样本包含$\{X,Y,Z\}$共3个特征，则其协方差矩阵为

$$
\begin{bmatrix}
cov(x,x)&cov(x,y)&cov(x,z)\\
cov(y,x)&cov(y,y)&cov(y,z)\\
cov(z,x)&cov(z,y)&cov(z,z)
\end{bmatrix}
$$

协方差矩阵，可视作**方差**和**协方差**两部分组成，即方差（各个通道的方差）构成了对角线上的元素，协方差（不同通道信号的协方差）构成了非对角线上的元素，matlab计算源码：

```matlab
Sample = fix(rand(10,4)*50);
X = Sample-ones(10,1)*mean(Sample); % 中心化样本矩阵zero-centered，使各维度均值为0
% X = bsxfun(@minus, X, mean(Sample)); % 另外一种方法，原理是for循环，速度更快
C = (X'*X)./(size(X,1)-1);
```

### 协方差矩阵和散度矩阵

协方差是样本还是特征之间的关系？散度矩阵是样本还是特征之间的关系？用到 LDA 时经常忘了其中的散度矩阵和协方差阵有什么区别。其实两者都体现的是特征之间的关系，比如 $X\in \mathbb{R}^{n\times d}$（注意样本和特征维度顺序），则协方差阵为 $X^{\top}X\in \mathbb{R}^{d\times d}$，散度矩阵为协方差矩阵乘上 $(n-1)$，也即 $(n-1)X^{\top}X\in \mathbb{R}^{d\times d}$。

```matlab
C1=cov(X); % 原始：X, n(samples) x d(features)
C2=cov([X(y==1,:);X(y==2,:)]); % 改变顺序
[n,d]=size(X); 
mu=mean(X); St=zeros(d);
for i=1:n; St=St+(X(i,:)-mu)'*(X(i,:)-mu); end
C3 = St/(n-1); % 散度矩阵除以 n-1
```

上面是三种计算协方差矩阵的方法：C1, C2, C3 的结果一样。也就是说对于矩阵 X，改变其样本的顺序不会改变其计算结果，因为其计算原理是总体散度矩阵除以 n-1。

### 协方差矩阵和相关系数矩阵

相关系数矩阵指的是由皮尔逊相关系数( Pearson correlation coefficient）构成的矩阵，Pearson 系数用于计算两个向量之间的相关程度，matlab 计算方式 `corrcoef `。如果计算协方差矩阵之前将数据正态化，即将原始数据按列处理成均值为 0，方差为 1 的标准数据，那么协方差矩阵等同于相关矩阵。

```matlab
Sample = fix(rand(11,4)*50);
r1 = corrcoef(Sample);
r2 = cov(zscore(Sample));
```

## 协方差矩阵和特征分解

特征值分解的一般目的是为了降维，以解决在高维情况下出现的数据样本稀疏、距离计算困难等问题，也即解决维度灾难（curse of dimensionality）

### 方阵的特征值分解 (eigenvalue decomposition, EVD )

对于方阵$A$，首先根据公式求其特征值和特征向量

$$
A\nu=\lambda\nu
$$

这时候 $λ$ 就被称为特征向量 $\nu$ 对应的特征值，一个矩阵的一组特征向量是一组正交向量。特征值分解是将一个方阵分解成下面的形式：

$$
A=Q\Sigma Q^{-1}
$$

其中 $Q$ 是 $A$ 的特征向量组成的矩阵，$\Sigma$ 是一个对角阵，对角线上的元素就是特征值，从大到小，描述这个矩阵的主要到次要变化方向 。特征值分解可以得到特征值与特征向量，特征向量表示旋转矩阵，特征值对应于每个维度上缩放因子的平方。 那么选取前 $n$ 个特征值对应的特征向量就是方阵A的前 $n$ 个主要线性子空间。注意如果 $A$ 不是对称的话，那么这 n 个方向不是正交的。

### 协方差矩阵的特征值分解

注意这边是针对 $\Sigma$ 进行分解，但是并不代表原始数据 $A$ 等价于 $\Sigma$ ，当 $A$ 是一个方阵时，不需要计算 $\Sigma$ ，当 $A$ 不是方阵时，下面的分解可以作为 SVD 的一个步骤，但是如果直接用来特征值分解，效果没有 SVD 好。对于任意**正定对称**矩阵 $\Sigma$，存在一个特征值分解 (EVD)：

$$
\Sigma=U\Lambda U^{\top}
$$

其中，$U$ 的每一列都是相互正交的特征向量，且是单位向量，满足 $U^{\top}U=I$, $\Lambda=\mathrm{diag}(\lambda_1, \lambda_2, ..., \lambda_d )$，对角线上的元素是从大到小排列的特征值，非对角线上的元素均为 0。

$$
\Sigma=\left(U\Lambda^{1/2}\right)\left(U\Lambda^{1/2}\right)^{\top}=AA^{\top}
$$

协方差矩阵的最大特征向量总是指向数据最大方差的方向，并且该向量的幅度等于相应的特征值。第二大特征向量总是正交于最大特征向量，并指向第二大数据的传播方向。

### 奇异值分解（singular value decomposition, SVD）

奇异值分解是一个能适用于任意的矩阵的一种分解的方法，奇异值有类似于特征值的性质，当矩阵为共轭对称矩阵时，特征值=奇异值。 

$$
A=U\Sigma V^{\top}
$$

假设 $A$ 是一个 $m\times n$ 的矩阵，那么得到的 $U$ 是一个 $m\times m$ 的方阵（里面的向量是正交的，$U$ 里面的向量称为左奇异向量），$\Sigma$ 是一个 $m\times n$  的矩阵（除了对角线的元素都是 0，对角线上的元素称为奇异值），$V^{\top}$ 是一个 $n\times n$ 的矩阵，里面的向量也是正交的，$V$ 里面的向量称为右奇异向量）

计算 $U$ 和 $V$ 也是利用了协方差矩阵特征值分解原理，由于 $A^{\top}A$ 是一个对称方阵，

$$
(A^{\top}A)\nu_i=\lambda_i\nu_i
$$

这里得到的 $\nu$，就是上面的右奇异向量。此外还可以得到：

$$
\sigma_i=\sqrt{\lambda_i}
$$

$$
u_i=\frac{1}{\sigma_i}A\nu_i
$$

这里的 $\sigma$ 就是上面说的奇异值，$u$ 就是上面说的左奇异向量。奇异值 $\sigma$ 跟特征值类似，在矩阵 $\Sigma$ 中也是从大到小排列，而且 $\sigma$ 的减少特别的快，因此取 $\Sigma$ 的前 $r$ 个特征值就可以估计出 $A$。这里提一下，虽然奇异向量有左右，但是可以用的只有左奇异向量，由公式（10）可以知 $\nu_i$ 的计算方式和协方差矩阵特征分解相似，但维度不对。

$$
A_{m\times n}\approx U_{m\times r}\Sigma_{r\times r}V_{r\times n}^{\top}
$$

### 协方差矩阵和PCA

参考了很多blog，最后还是觉得西瓜书上讲的清楚一些。首先区分一下矩阵内积和协方差矩阵。矩阵内积形式为 $XX^{\top}$，是样本之间的相关性，而协方差矩阵是 $X$ 经过中心化处理之后，$kX_c^{\top}X_c$ ，其中 $k$ 是缩放系数，$k^{-1}=size(X,1)-1$，所以可以用 $X^{\top}X$ 指代协方差矩阵。对于样本特征和样本数不一致的非方阵数据可以采用协方差矩阵特征值分解以及奇异值分解。无论哪种PCA，原则是

$$
\frac{\Sigma^{d^\prime}_{i=1}\lambda_i}{\Sigma^{d}_{i=1}\lambda_i}\ge t
$$

其中的 $t$ 是降维之后保留的特征值数值占的比例。使用奇异值分解SVD进行PCA流程如下：

```matlab
%% 完整版本实现svd
function [S]=svd_wen(X, rate)
t=size(X,1);
X = X-repmat(mean(X),t,1);
C = (X'*X)./(t-1);

[V,Lambda] = eig(C);
ldig=diag(Lambda);
for i=1:length(ldig)
    if sum(ldig(end-i+1:end))>rate*sum(ldig); break; end
end
V=V(:,(end-i+1:end));
Lambda=ldig(end-i+1:end);

nPc=length(Lambda);
S=zeros(t,nPc);
for k=1:nPc
    S(:,k)=(1/sqrt(Lambda(k)))*X*V(:,k); % 按照公式(12)来计算的
end
end

%% 简化版本
MySample = fix(rand(11,15)*50);
X=MySample; t=size(X,1);
X = X-repmat(mean(X),t,1);
[U,S,V]=svd(X);
ldig=diag(S); rate=0.95;
for i=1:length(ldig)
    if sum(ldig(1:i))>rate*sum(ldig); break; end
end
sample=U(:,1:i);
```

比较来看 SVD 版本的 PCA 是对 SVD 的一个简单包装，而 SVD 又是基于数据的协方差矩阵，并计算了特征值和特征向量，但是这里的计算出的特征向量只是原始数据协方差矩阵的，真正的特征向量是左奇异向量的列向量。

## 协方差矩阵和数据结构

尽管协方差矩阵很简单，可它却是很多领域里的非常有力的工具。它能导出一个变换矩阵，参考白化 PCA 以及 ZCA，这个矩阵能使数据完全去相关 (decorrelation)。从不同的角度看，也就是说能够找出一组最佳的基以紧凑的方式来表达数据。在脑机接口领域，协方差矩阵可以体现数据的很多信息，不展开。

转载请注明出处，谢谢！