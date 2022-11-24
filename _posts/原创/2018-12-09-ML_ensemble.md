---
layout: post
title: 集成学习算法总结
date: 2018-12-09 16:22
tags: [ensemble,boosting]
categories: Machine-Learning
description: "模式识别课上和平时组会上了解了不少集成学习，做个小总结"
mathjax: true
---

* content
{:toc}
模式识别课上和平时组会上了解了不少集成学习，做个小总结。 <!--more-->

### 1.Bagging, Boosting, Stacking对比

集成学习可以是不同算法的集成，也可以是同一算法在不同设置下的集成，还可以是数据集不同部分分配给不同分类器之后的集成。集成学习可分为两大类，以Adaboost, GBDT为代表的 Boosting（提高） 和以 RandomForest 为代表的 Bagging（装袋），它们在集成学习中属于同源集成（homogenous ensembles）；另一种是 Stacking（堆叠），属于异源集成（heterogenous ensembles）。

Bagging 各个弱学习器之间**没有依赖关系，可以并行拟合**；对于boosting，各个弱学习器之间**有依赖关系**。Stacking 是通过一个元学习器来整合多个基础学习模型，基础模型通常包含不同的学习算法。

Bagging 的方法采用多个分类器集体表决，集体表决意味着模型泛化能力比较强，其分类能力相对于其中的每一个单一分类器是稳定的，相当于降低了方差。Boosting 的方法通过构造一系列的弱分类器对相同的训练数据集的重要性区别对待达到对训练数据的精确拟合，因此降低了偏差。

Bagging典型的是随机森林，而boosting 包括 Adaboost、Gradient Boosting（包括Boosting Tree、Gradient Boosting Decision Tree、xgboost），Stacking 常用方法是堆栈泛化（Stacked Generalization）

### 2.决策树与随机森林

决策树用于分类和回归，包含特征选择(Gini)、决策树的生成和决策树的剪枝三个步骤。可处理缺失数据，运算速度快，但是容易过拟合，随机森林采用多个决策树的投票机制来改善决策树。
假设随机森林使用了 m 棵决策树，那么就需要产生 m 个一定数量的样本集（n个）来训练每一棵树，通过Bootstraping法，这是一种有放回的抽样方法，产生 n 个样本而最终结果采用Bagging的策略来获得，即多数投票机制

随机森林的生成方法：
1.从样本集中通过重采样的方式产生 n 个样本，构成训练样本集
2.假设样本特征数目为 a，对 n 个样本选择 a 中的 k 个特征，用建立决策树的方式获得最佳分割点
3.重复 m 次，产生 m 棵决策树
4.多数投票机制来进行预测

### 3.Adaboost

**重点一**在于弱分类器集成时的权重

$$
\alpha_m=  1/2 ln \frac {(1-e_m)} {e_m}
$$

可以看出对于某个迭代子阶段的分类器，其权重和其误差负相关。**重点二**在于迭代过程中的样本的权重更新规则

$$
\begin{align}
w_{m+1,i}&= \frac {w_{mi} } {Z_m}exp(-\alpha_m y_i G_m (x_i)),i=1,2,⋯,N，其中Z_m 为归一化因子，该式可简化为\\
&=\left\{ {\begin{matrix}\frac {w_{mi} }{2(1-e_m)}, &{G_m(x_i)=y_i}\\\frac {w_{mi} }{2e_m},\ &{G_m(x_i)\neq y_i}\\\end{matrix} }\right.
\end{align}
$$

可以看出样本的权重和错误率正相关，即如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它的权值就被降低；相反，如果某个样本点没有被准确地分类，那么它的权值就得到提高。

### 4.随机森林与Adaboost

两者的区别主要是 bagging 和 boosting 的区别。boosting 是一种与 bagging 很相似的技术。但前者不同的分类器是通过串行训练而得到的，每个新分类器都根据已训练出的分类器的性能来进行训练。boosting 中的分类的结果是基于所有分类器的加权求和结果的，使得 loss function 尽量考虑那些分错类的样本（i.e.分错类的样本 weight 大），而 bagging 中的分类器权值是相等的。boosting 重采样的不是样本，而是样本的分布。

随机森林（ Random Forest, RF）是决策树的组合，每棵决策树都是通过对原始数据集中随机生成新的数据集来训练生成，随机森林决策的结果是多数决策树的决策结果。

### 5.GBDT

具体参考，[GBDT算法原理深入解析](https://www.zybuluo.com/yxd/note/611571)

Boosting Tree模型是决策树的加法模型，表现为 

$$
f_M(x)=\sum_{m=1}^MT(x;\theta_m)
$$

其中，$T(x;\theta_m)$ 表示决策树，$\theta_m$ 为树的参数, $M$ 为树的个数。Boosting Tree 提升树利用加法模型实现优化过程时，当损失函数是平方损失函数时，每一步的优化很简单。但对于一般损失函数而言，往往每一步的优化没那么简单，所以引入了梯度提升（Gradient Boosting）算法。GBDT目标函数如下：

$$
Obj = -\frac12 \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda} + \gamma T
$$

其学习步骤为：
1. 算法每次迭代生成一颗新的决策树 
2. 在每次迭代开始之前，计算损失函数在每个训练样本点的一阶导数 $g_i$ 和二阶导数 $h_i$
3. 通过贪心策略生成新的决策树，计算每个叶节点对应的预测值 
4. 把新生成的决策树 $f_t(x)$ 添加到模型中：$\hat{y}_i^t = \hat{y}_i^{t-1} + f_t(x_i)$

### 6.Xgboost

**6.1 Xgboost过程**

Xgboost目标函数可以定义为如下：

$$
\begin{align}
obj&:\Sigma_{i=1}^nl(y_i,\hat y_i)+∑_{k=1}^K\Omega(f_k)\\
&where\ \Omega(f_t)=\gamma T+\frac 1 2 \lambda\Sigma_{j=1}^Tw_j^2
\end{align}
$$

其中 $n$ 代表有 $n$ 个样本。前面一项是 loss 函数，$T$ 是叶子节点数目，$w$ 是leaf score的 L2 模的平方，对leaf scroe做了平滑。

**6.2 Xgboost和GBDT对比**

参考 [机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？](https://www.zhihu.com/question/41354392/answer/98658997)

- GBDT以CART作为基分类器，xgboost也支持线性分类器，此时xgboost相当于带L1和L2正则化项的logistics回归（分类问题）或者线性回归（回归问题）。
- GBDT优化时只用到一阶导信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。
- xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。
- xgboost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。
- 决策树的学习最耗时的步骤是对特征的值进行排序，xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量，支持特征增益的并行计算。
- GBDT采用的是数值优化的思维， 用的最速下降法去求解Loss Function的最优解，其中用CART决策树去拟合负梯度， 用牛顿法求步长。Xgboost用的解析的思维，对Loss Function展开到二阶近似，求得解析解，用解析解作为Gain来建立决策树，使得Loss Function最优。

### 7.Stacking算法

参考 [集成方法（Ensemble）之Stacking](https://blog.csdn.net/g11d111/article/details/80215381 )

Stacking 常用方法是堆栈泛化（Stacked Generalization），其过程

- 将训练集分为3部分，分别用于让3个基分类器（Base-leaner）进行学习和拟合
- 将3个基分类器预测得到的结果作为下一层分类器（Meta-learner）的输入
- 将下一层分类器得到的结果作为最终的预测结果

其特点是通过使用第一阶段（level 0）的预测作为下一层预测的特征，比起相互独立的预测模型能够有更强的非线性表述能力，降低泛化误差。它的目标是同时降低机器学习模型的 Bias-Variance。堆栈泛化就是集成学习中 Aggregation 方法进一步泛化的结果， 是通过 Meta-Learner 来取代 Bagging 和 Boosting 的 Voting/Averaging 来综合降低 Bias 和 Variance 的方法。譬如： Voting可以通过 kNN 来实现， weighted voting 可以通过softmax（Logistic Regression）， 而 Averaging 可以通过线性回归来实现。
