---
layout: post
title: Sklearn 机器学习全过程
date: 2020-10-25 15:33
tags: [ml,python]
categories: Machine-Learning
description: "基于sklearn的机器学习全过程（全流程），从问题定位，数据分析，数据预处理，特征工程，模型训练与验证"
mathjax: true
---

* content
{:toc}
Matlab 做 ML 用的不少，特整理下基于 sklearn 的机器学习全过程，从问题分析，数据分析，数据预处理，到特征工程到模型训练与验证。 <!--more-->

**Cham's Blog 首发原创**



## ML 解决问题全过程

在选择具体的算法之前，最好对数据中每一个特征的模式和产生原理有一定的了解：

- 特征是连续的（real-valued）还是离散的（discrete）？
- 如果特征是连续的，它的直方图（histogram）长什么样？它的 mean 和 variance 是如何分布的？
- 如果特征是离散的，不同的特征值之间是否存在某种顺序关系？例如，1 到 5 的打分，虽然是离散数据，但有一个从低到高的顺序。如果某个特征是“地址”，则不太可能存在一个明确的顺序。
- 特征数据是如何被采集的？

**具体参考以下资料：**

- [机器学习模型训练全流程！](https://mp.weixin.qq.com/s/uQD7j0KCjzxWOshIdccAPg)
- [Sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [在使用sklearn时如何选择合适的分类器](https://osswangxining.github.io/sklearn-classifier/)
- [UCI Datasets](https://archive.ics.uci.edu/ml/datasets.php)



## Sklearn中常用模型

### 分类模型

> **1）最开始**
>
> 建立模型时，选择 high bias, low variance 的线性模型作为 baseline。线性模型的优点包括计算量小、速度快、不太占内存、不容易过拟合。
>
> 常用线性回归器的有 Ridge（含有 L2 正则化的线性回归）和 Lasso（含有 L1 正则化的线性回归，自带特征选择，可以获得 sparse coefficients）。同时，如果对于超参数没有什么特别细致的要求，那么可以使用 sklearn 提供的 RidgeCV 和 LassoCV，自动通过高效的交叉验证来确定超参数的值。
>
> 假如针对同一个数据集 $X$（m samples $\times$ *n features），需要预测的 $y$ 值不止一个（m samples $\times$ n targets），则可以使用 multi-task 的模型。
>

> **2）试试集成**
>
> Ensemble 能够极大提升各种算法，尤其是决策树的表现。在实际应用中，单独决策树几乎不会被使用。Bagging（如 RandomForest）通过在数据的不同部分训练一群 high variance 算法来降低算法们整体的 variance；boosting 通过依次建立 high bias 算法来提升整体的 variance。
>
> BaggingClassifier 和 VotingClassifier 可以作为第二层的 meta classifier/regressor，将第一层的算法（如 XGBoost）作为 base estimator，进一步做成 bagging 或者 stacking。
>

> **3）最后是**
>
> 支持向量机（SVM）和神经网络（Neural Network）

```python
# 基本回归：线性、决策树、KNN
# 集成方法：随机森林、Adaboost、Random Forest、Bagging、Extra Trees等
# 其他方法：SVM、NN
classifiers = [
    ('Logistic Regression', LogisticRegression()),  # 逻辑回归
    ('Nearest Neighbors', KNeighborsClassifier(3)),  # K最近邻
    ('Linear SVM', SVC(kernel='linear', C=0.025)),  # 线性的支持向量机
    ('RBF SVM', SVC(gamma=2, C=1)),  # 径向基函数的支持向量机
    ('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))),  # 基于拉普拉斯近似的高斯过程
    ('Decision Tree', DecisionTreeClassifier(max_depth=5)),  # 决策树
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),  # 随机森林
    ('AdaBoost', AdaBoostClassifier()),  # 通过迭代弱分类器而产生最终的强分类器的算法
    ('Extra Trees', ExtraTreesClassifier()),
    ('GradientBoosting', GradientBoostingClassifier()),  # 梯度提升树
    ('Bagging', BaggingClassifier()),
    ('Naive Bayes', GaussianNB()),  # 朴素贝叶斯
    ('QDA', QuadraticDiscriminantAnalysis()),  # 二次判别分析
    ('LDA', LinearDiscriminantAnalysis()),  # 线性判别分析
    ('MLP', MLPClassifier(alpha=1)),  # 多层感知机
    ('XGB', XGBClassifier()),  # 极端梯度提升
]

for name, clf in classifiers:
    clf.fit(X_train, y_train)  # 训练
    score = clf.score(X_test, y_test)  # 模型评分
```

### 回归模型

```python
# 基本回归：线性、决策树、SVM、KNN
# 集成方法：随机森林、Adaboost、Gradient Boosting、Bagging、Extra Trees
regressors = [
    ('Decision Tree', DecisionTreeRegressor()),  # 逻辑回归
    ('Linear Regression', LinearRegression()),  # K最近邻
    ('SVR', SVR()),  # 线性的支持向量机
    ('KNN',KNeighborsRegressor()),  # 径向基函数的支持向量机
    ('Random Forest', RndomForestRegressor(n_estimators=20))  # 使用20个决策树
    ('AdaBoost',AdaBoostRegressor(n_estimators=50))  # 使用50个决策树
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100))  # 使用100个决策树
    ('Bagging',BaggingRegressor())
    ('Extra Trees', ExtraTreeRegressor())
]

for name, reg in regressors:
    reg.fit(X_train, y_train)  # 训练
    score = reg.score(X_test, y_test)  # 模型评分
```

### 聚类模型

| Method name | Parameters | Scalability | Usecase | Geometry (metric used) |
| -- | -- | -- | -- | -- |
| [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means) | number of clusters                                           | Very large `n_samples`, medium `n_clusters` with [MiniBatch code](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans) | General-purpose, even cluster size, flat geometry, not too many clusters | Distances between points                     |
| [Affinity propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation) | damping, sample preference                                   | Not scalable with n_samples                                  | Many clusters, uneven cluster size, non-flat geometry        | Graph distance (e.g. nearest-neighbor graph) |
| [Mean-shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift) | bandwidth                                                    | Not scalable with `n_samples`                                | Many clusters, uneven cluster size, non-flat geometry        | Distances between points                     |
| [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) | number of clusters                                           | Medium `n_samples`, small `n_clusters`                       | Few clusters, even cluster size, non-flat geometry           | Graph distance (e.g. nearest-neighbor graph) |
| [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold                     | Large `n_samples` and `n_clusters`                           | Many clusters, possibly connectivity constraints             | Distances between points                     |
| [Agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold, linkage type, distance | Large `n_samples` and `n_clusters`                           | Many clusters, possibly connectivity constraints, non Euclidean distances | Any pairwise distance                        |
| [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) | neighborhood size                                            | Very large `n_samples`, medium `n_clusters`                  | Non-flat geometry, uneven cluster sizes                      | Distances between nearest points             |
| [OPTICS](https://scikit-learn.org/stable/modules/clustering.html#optics) | minimum cluster membership                                   | Very large `n_samples`, large `n_clusters`                   | Non-flat geometry, uneven cluster sizes, variable cluster density | Distances between points                     |
| [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html#mixture) | many                                                         | Not scalable                                                 | Flat geometry, good for density estimation                   | Mahalanobis distances to  centers            |
| [Birch](https://scikit-learn.org/stable/modules/clustering.html#birch) | branching factor, threshold, optional global clusterer.      | Large `n_clusters` and `n_samples`                           | Large dataset, outlier removal, data reduction.              | Euclidean distance between points            |



## 模型训练与测试

### 独立划分训练集和测试集

```python
# 独立固定划分训练集和测试集
scores = np.zeros([10, 2])
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(label_X, label_Y, test_size=0.3)

    # SVR regression
    svr_reg = SVR(kernel='linear')
    svr_reg.fit(x_train, y_train)
    y_pred = svr_reg.predict(x_test)
    scores[i, 0] = mean_squared_error(y_test, y_pred)
    scores[i, 1] = mean_absolute_error(y_test, y_pred)

print(np.mean(scores, axis=0), np.std(scores, axis=0))
```

### 常见的交叉验证训练方式

```python
# 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
kf = model_selection.RepeatedStratifiedKFold(n_splits=5)

# 普通10-fold
kf1 = model_selection.KFold(n_splits=10, random_state=7)

i = 0
scores = np.zeros([kf*10, 2])
for train_idx, test_idx in kf.split(label_X, label_Y):
    y_train, y_test = label_Y[train_idx], label_Y[test_idx]

    # select features
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(label_X.iloc[train_idx, :], y_train)
    fim = model.feature_importances_
    b = np.array(sorted(enumerate(fim), key=lambda x: x[1]))
    idx = b[:80, 0].astype(int)

    x_train, x_test = label_X.iloc[train_idx, idx], label_X.iloc[test_idx, idx]

    # KNN classification
    neigh = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    y_pred = neigh.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    scores[i, 0] = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    scores[i, 1] = f1_score(y_test, y_pred, average='weighted')

    i = i + 1

print(np.mean(scores, axis=0), np.std(scores, axis=0))
```



## 评价指标 (分类、回归、聚类)

### 分类指标

对于二类分类问题，通常以关注的类为正类（positive），其他类为负类（negative），分类器在测试数据集上的预测或正确（true）或不正确（false），4 种情况出现的总数分别记作：

|                  |      正类（预测）      | 负类（预测） |
| :---: | :---: | :---: |
| **正类（真实）** | TP——将正类预测为正类数 | FP——将负类预测为正类数 |
| **负类（真实）** | FN——将正类预测为负类数 | TN——将负类预测为负类数 |

**由此定义：**

| 真阳率：      TPR = TP / (TP + FN)  | 特异性：  TNR = TN / (FP + TN) |
| :---: | :---: |
| 召回率 recall： R = TP / (TP + FN)       | 精准率 precision：P = TP / (TP + FP)          |
| F1 值：			  F1 = 2 PR / (P + R) | 准确率 accuracy： ACC = (TP+TN) / (P+N)  |

**注意：**

- **真阳率又称召回率、Sensitivity（敏感性）、查全率；Specificity（特异性）**，表示的是预测正确的负样本个数占所有预测为负样本的比例
- 精准率和召回率和 F1 取值都在 0 和 1 之间，精准率和召回率高，F1 值也会高
- 关于ROC曲线和AUC。假设采用逻辑回归分类器，其给出针对每个实例为正类的概率，那么通过设定一个阈值如 0.6，概率大于等于 0.6 的为正类，小于 0.6 的为负类。对应的就可以算出一组 (FPR,TPR)，在平面中得到对应坐标点。随着阈值的逐渐减小，越来越多的实例被划分为正类，但是这些正类中同样也掺杂着真正的负实例，即 TPR 和 FPR 会同时增大。阈值最大时，对应坐标点为 (0,0)，阈值最小时，对应坐标点(1,1)。曲线下面积为 AUC。python 计算 AUC 过程，参考 [Receiver Operating Characteristic (ROC)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)

```python
# 计算AUC，需要提供类别的预测概率
y_score = model.predict_proba(x_test)

# 1、调用函数计算micro类型的AUC
print('调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro'))

# 2、手动计算micro类型的AUC
#首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel())
auc = metrics.auc(fpr, tpr)
print('手动计算auc：', auc)
```



### 其他指标

参考 [sklearn metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)