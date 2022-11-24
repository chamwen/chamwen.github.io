---
layout: post
title: 损失函数和正则化项
date: 2018-12-06 19:12
tags: [loss,regularization]
categories: Machine-Learning
description: "目标函数中常用的损失函数，如0-1损失、平方损失、对数损失，常见的正则化项 L1 和 L2，以及常见的回归模型损失函数"
mathjax: true
---

* content
{:toc}
目标函数中常用的损失函数，如0-1损失、平方损失、对数损失，常见的正则化项 L1 和 L2，以及常见的回归模型损失函数。 <!--more-->

### 目标函数

模型的目标函数通常定义为如下形式：$Obj(\Theta)=L(\Theta)+\Omega(\Theta)$

其中，$L(\Theta)$ 是损失函数，用来衡量模型拟合训练数据的好坏程度；$\Omega(\Theta)$ 称之为正则项，用来衡量学习到的模型的复杂度。目标函数之所以定义为损失函数和正则项两部分，是为了尽可能平衡模型的偏差和方差（Bias Variance Trade-off）。最小化目标函数意味着同时最小化损失函数和正则项，损失函数最小化表明模型能够较好的拟合训练数据，一般也预示着模型能够较好地拟合真实数据；另一方面，对正则项的优化鼓励算法学习到较简单的模型，简单模型一般在测试样本上的预测结果比较稳定、方差较小（奥卡姆剃刀原则）。也就是说，优化损失函数尽量使模型走出欠拟合的状态，优化正则项尽量使模型避免过拟合。 <!--more-->



### 常用的损失函数

训练集上的损失定义为：$L=\sum_{i=1}^n l(y_i, \hat{y}_i)$

1.0-1损失函数 (0-1 loss function): 
$$
L(Y, f(X)) = \left\{ \begin{array} { l } { 1 , \quad Y \neq \mathrm{f}(\mathrm {X}) } \\ {0, \quad Y = \mathrm { f } ( \mathrm { X } ) } \end{array} \right.
$$

2.平方损失函数 (quadratic loss function) : $L(Y,f(X))=(Y−f(x))^2$

3.绝对值损失函数 (absolute loss function) : $L(Y,f(x))=\|Y−f(X)\|$

4.对数损失函数 (logarithmic loss function) : $L(Y,P(Y\mid X))=−logP(Y\mid X)$

5.Logistic 损失：$l(y_i, \hat{y}_i)=y_i ln(1+e^{y_i}) + (1-y_i)ln(1+e^{\hat{y}_i})$

6.Hinge 损失：$\text{hinge}(x_i) = \max (0,1-y_i(\mathrm{w}^{\top}x_i+b))$ ，SVM 损失函数，如果点正确分类且在间隔外，则损失为 0；如果点正确分类且在间隔内，则损失在 $(0,1)$；如果点错误分类，则损失在 $(1,+\infty)$

7.负对数损失 (negative log-likelihood, NLL)：$L_i = -log(p_{y_{i}})$，某一类的正确预测的失望程度 ($>0$)，其值越小，说明正确预测的概率越大，表示预测输出与 $y$ 的差距越小

8.交叉熵损失 (cross entropy)：首先是 softmax 定义为 $p_k = {e^{f_k}}/{\sum_{j} e^{f_j}}$，其中 $f_k=Wx+b$ 表示某一类的预测输出值，则某类的交叉熵损失为该类的输出指数值除所有类之和。基于交叉熵和 softmax 归一化的 loss

$$
L=-\frac 1 N\sum_{i=1}^Ny_i\ \text{log}\ \frac {e^{f(x_i)}} {\sum e^{f(x_i)}}
$$

关于softmax和 NLL 可以参考：[Understanding softmax and the negative log-likelihood](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)



### 常用的正则化项

常用的正则项有 L1 范数 $\Omega(w)=\lambda \Vert w \Vert_1$和 L2 范数 $\Omega(w)=\lambda \Vert w \Vert_2$
- L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
- L2正则化可以防止模型过拟合（overfitting），一定程度上，L1也可以防止过拟合

$J=J_0+L$，此时任务变成**在 $L$ 约束下求出 $J_0$ 取最小值的解**，对于L1正则化，约束就是一个菱形，对于L2正则化，约束就是一个圆形。$J_0$ 是一个和 $w$ 有关的椭圆，在二维空间中 $w$ 越小，其半径越大。

{:refdef: style="text-align: center"}
![default](\images\cost_function\cost_l1.jpg){:height="30%" width="30%"}
{:refdef}

L2正则化如图所示，最优解取值为零的概率要小于L1正则化

{:refdef: style="text-align: center"}
![default](\images\cost_function\cost_l2.jpg){:height="30%" width="30%"}
{:refdef}

**Q1：为什么L1 正则化可以获得稀疏特征？**
不同的维度系数一般都是不一样的，因此常见的损失函数图像是一个椭圆形，调整参数 $λ$ 的值，椭圆形和菱形的交接处很大可能性会落在坐标轴附近；实际使用的特征维度是高维的，正常情况下就是在某些维度上不会交于坐标轴上，在某些维度上交于坐标轴或坐标轴附近，所以才有稀疏解；与L2正则化相比，L1正则化更容易得到稀疏解，而不是一定会得到稀疏解，毕竟还是有特例的（比如恰好椭圆与坐标原点相交）。

**Q2：$λ$ 变大，菱形和圆形怎么变化？**
$λ$ 越大，菱形和圆形越小，求得的模型参数越小。

**Q3：为什么 L2 正则化比 L1 正则化应用更加广泛？**
因为 L2 正则化的约束边界光滑且可导，便于采用梯度下降法，而L1正则化不可导，只能采用坐标轴下降法或最小角回归法，计算量大。而且，L1 正则化的效果并不会比 L2 正则化好。

**Q4：L1 和 L2 正则先验分别服从什么分布 ？**
L1 是拉普拉斯分布，L2 是高斯分布。




### L1 和 L2 梯度下降速度对比

根据L1和L2的函数图像可以看出, L1是按照线性函数进行梯度下降的, 而L2则是按照二次函数, 因此, L1在下降时的速度是恒定的, 在接近于0的时候会很快就将参数更新成0 , 而L2在接近于0 时, 权重的更新速度放缓, 使得不那么容易更新为0 :

{:refdef: style="text-align: center"}
![default](\images\cost_function\cost_regu.png){:height="50%" width="50%"}
{:refdef}


### 常用的回归模型

**Linear回归模型**：$\min_{w}\sum_{i=1}^N(y_{i}-w^Tx_{i})=(X^TX)^{-1}X^Ty$

**Lasso 回归模型**：使用平方损失和 L1 范数正则项的线性回归模型，用于估计稀疏参数的线性模型，特别适用于参数数目缩减，获得稀疏特征，其模型：$\min_{w}\sum_{i=1}^N(y_{i}-w^Tx_{i})+\gamma{\|w\|}_1$

**Ridge 回归模型**：使用平方损失和 L2 范数正则项的线性回归模型，适用于特征之间完全共线性或者相关程度异常高的时候，其模型：$\min_{w}\sum_{i=1}^N(y_{i}-w^Tx_{i})+\gamma{\|w\|}^2=(X^TX+\gamma I)^{-1}X^Ty$

**Logistic 回归模型**：使用logistic损失和 L2 范数或 L1 范数正则项的线性分类模型。

注意前面三个都是回归模型，最后一个是分类模型，因为标签的可能取值固定，所以损失函数一般选择对数损失函数。线性回归要求变量服从正态分布，logistic 回归对变量分布没有要求；logistic 回归采用 sigmoid 函数将连续值映射在 $(0,1)$ 之间，将归属各类别概率中最大的类作为预测。

参考：
1. [机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975)
2. [直观理解正则化](https://2018august.github.io/2.%20lr%E6%AD%A3%E5%88%99%E5%8C%96%E7%9A%84%E7%9B%B4%E8%A7%82%E7%90%86%E8%A7%A3/)

