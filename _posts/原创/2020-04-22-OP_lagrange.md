---
layout: post
title: 优化问题中的拉格朗日乘子法
date: 2020-04-22 15:16
tags: [Lagrange,ADMM]
categories: Optimization-Theory
description: "拉格朗日法的基础理论，不同限制条件下的拉格朗日法，对偶问题，交替方向乘子法 (ADMM) ，具体优化例子。"
mathjax: true
---
* content
{:toc}
之前看的一些传统优化问题，用的基本都是拉格朗日乘子法 (Lagrange Multiplier)，但最近遇到一些非平常的优化问题，特此系统整理一下这一块的理论，包括：拉格朗日法的基础理论，不同限制条件下的拉格朗日法，对偶问题，交替方向乘子法 (ADMM) ，具体优化例子。 <!--more-->

**Cham's Blog 首发原创**



### 拉格朗日法的基础理论

**1）怎么判断凸函数？**

定义：满足 Jensen 不等式， $f(\theta x+(1-\theta)y)\leq \theta f(x)+(1-\theta)f(y)$，或者记为 $E[f(x)]\ge f(E(X))$
判断：对于一元函数 $f(x)$，我们可以通过其二阶导数 $\nabla f(x)$ 的符号来判断。如果函数的二阶导数总是非负，即 $\nabla^2 f(x)\ge0$ ，则 f(x) 是凸函数；对于多元函数 $f(x)$，其 Hessian 矩阵（Hessian 矩阵是由多元函数的二阶导数组成的方阵）为半正定矩阵时，f(X) 是凸函数。

**2）怎么判断优化问题为凸优化？**

凸优化问题的一般形式

$$
\begin{align}
\min &\ f(x)\\
\text{s.t.} &\ h_i(x)=0, \forall i=1,...,m\\
&g_j(x)\le0,\forall j=1,..,n
\end{align}
$$

$f$，$g_j$ 为凸函数，$h_i$ 是仿射函数 (带有平移的线性变换 $Ax+b$)，一般是最小化损失函数。



### 不同限制条件下的拉格朗日法

**1）无约束**
直接梯度为 0，计算结果，一般有解析解；如果没有的话，需要梯度下降法。

**2）等式约束**
等式约束一般使用拉格朗日乘子法（Lagrange Multiplier）

$$
\begin{align}
\min &\ f(x)\\
\text{s.t.} &\ h_i(x)=0, \forall i=1,...,m
\end{align}
$$

优化方法，

$$
\nabla_x f(x)+\sum_i{\lambda_i\nabla_x h_i(x)}=0
$$

**3）不等式约束**
函数有不等约束一般使用 KKT（Karush Kuhn Tucker）条件

$$
\begin{align}
\min &\ f(x)\\
\text{s.t.}& \ g_j(x)\le0,\forall j=1,..,n
\end{align}
$$

最优解满足需要满足以下条件

$$
\begin{cases}
\nabla_x f(x)+\sum_j{\mu_j\nabla_x g_j(x)}=0\ \\
\mu_i \ge 0\ \\
g_i(x) \le0\ \\
\mu_ig_i(x)=0
\end{cases}
$$

当不等式约束为大于等于 0 时，变换成标准形式即可。SVM 中的优化过程用到了这里的 KKT 条件，对 $x$ 求偏导，然后带入，将原始问题转换成单变量的优化问题。

**4）等式与不等式混合**
将一个有 $n$ 个变量与 $k$ 个约束条件的最优化问题转换为一个解有 $n + k$ 个变量的方程组的解的问题

$$
\begin{align}
\min &\ f(x)\\
\text{s.t.} &\ h_i(x)=0, \forall i=1,...,m\\
&g_j(x)\le0,\forall j=1,..,n
\end{align}
$$

最优解成立的规范性条件，注意这里

$$
\begin{cases}
\nabla_x f(x)  + \sum_{i = 1}^{p}  \lambda_i\nabla_x h_i(x) + \sum_{j=1}^{q} \mu_j \nabla_x g_i(x)=0\\\
h_i(x) = 0\\\
\mu_j \ge 0 \\\
g_j(x) \le 0 \ \ \text{不等式约束，法线方向相反}\\\
\mu_jg_j(x) = 0 \ \ \text{互补松弛性条件，}\mu_j=0,\  g_j(x) \le 0;\ \mu_j\ge0,\  g_j(x)=0
\end{cases}
$$



### 对偶问题

对于含有不等式的约束问题，将其转换为对偶问题很有必要

$$
\begin{align*}
\min &\ f(\mathrm{w})\\
&\text{s.t.} \ h_i(\mathrm{w})=0,\ i=1,\dots,l\\
& \qquad g_i(\mathrm{w})\le 0,\ i=1,\dots,k
\end{align*}
$$

定义增广拉格朗日函数

$$
\mathcal{L}(\mathrm{w},\alpha,\beta)=f(\mathrm{w} )+\Sigma\alpha_ig_i(\mathrm{w})+\Sigma\beta_ih_i(\mathrm{w})
$$

为了解决这一优化问题，定义

$$
\Theta_p(\mathrm{w}) \triangleq \max_{\alpha,\beta,\alpha_i\ge 0}\mathcal{L}(\mathrm{w},\alpha,\beta)=
\begin{cases}
f(\mathrm{w}) \ & 若 \mathrm{w} 满足约束，在可行区域\\\
+\infty \ & 其它情况，\mathrm{w} 不在可行区域
\end{cases}
$$

$p^{\ast}$ 表示这个这个目标函数的最优值

$$
p^\ast=\min_\mathrm{w}\max_{\beta,\alpha_i\ge 0}\mathcal{L}(\mathrm{w},\alpha,\beta)
$$

看一下我们的新目标函数，先求最大值，再求最小值。以 SVM 为例，这样做之后，我们首先就要面对带有需要求解的参数 $\mathrm{w}$ 和 $b$ 的方程，而 $\alpha_i$ 又是不等式约束，这个求解过程不好做。所以，我们需要使用拉格朗日函数对偶性，将最小和最大的位置交换一下，这样就变成了：

$$
d^\ast=\max_{\beta,\alpha_i\ge 0}\min_\mathrm{w}\mathcal{L}(\mathrm{w},\alpha,\beta)
$$

可知 $d^{\ast}\le p^{\ast}$，即对偶问题最优值 $d^{\ast}$ 是原始问题最优值 $p^{\ast}$ 的一个下界，在满足 KKT 条件下，二者是等价的。Slater 定理和 KKT 条件，如果 $f$ 和 $g_i$ 是凸的， $h_i$ 是仿射，假设 $g_i$ 是严格可解的，即 $\exists \ \mathrm{w}$ 使得 $g_i(\mathrm{w})<0\quad \forall  i$，则存在 $\mathrm{w}^{\ast}, \alpha^{\ast}, \beta^{\ast}$ 使得 $\mathrm{w}^{\ast}$ 是原问题的解，$\alpha^{\ast}, \beta^{\ast}$ 是对偶问题的解。此外 $p^{\ast} = d^{\ast}=\mathcal{L}(\mathrm{w}^{\ast}, \alpha^{\ast}, \beta^{\ast})$，且 $\mathrm{w}^{\ast}, \alpha^{\ast}, \beta^{\ast}$ 满足 KKT 条件：

$$
\begin{cases}
\frac {\partial} {\partial \mathrm{w}_i} \mathcal{L} （\mathrm{w^{\ast}, \alpha^{\ast},\beta^{\ast} }）=0, \ i=1,\dots,n  \\\
 h_i(\mathrm{w})=0,\ i=1,\dots,l \\\
 g_i(\mathrm{w}^{\ast})\le0 \quad i=1,\dots,k\\\
\alpha^{\ast}g_i(\mathrm{w}^{\ast})=0,\quad i=1,\dots,k \\\
\alpha_i\ge0 \quad i=1,\dots,k
\end{cases}
$$

满足上述几条约束，就可以用对偶问题替代原始问题，并且对偶问题的解是原问题最优值得最好下界。




### 拉格朗日法和交替方向乘子法

#### 交替方向乘子法 ADMM 

传统解决带约束凸问题时，往往会利用拉格朗日乘子法作转换，将其转化为无约束问题然后进行求解，但传统方法对数据及函数假设较为严格，所以一般情况很难达到收敛，假设要求宽松的又难以做大规模分布式计算，所以为了解决这个问题，Boyd 提出了 ADMM。ADMM 在理论上是综合了对偶上升、对偶分解、增广拉格朗日法。增广拉格朗日 (ALM) 是在拉格朗日函数中增加了关于约束条件的二次项 (或者说在拉格朗日函数上加约束条件的 “L2 正则项” )，目的是增加对偶上升法的鲁棒性和放松函数 $f$ 的强凸约束。先看 ALM 方法的一般形式，对于原始优化问题

$$
\begin{aligned}
\min&\ f(x)\\
\text{s.t.}&\ Ax=b
\end{aligned}
$$

其增广拉格朗日函数为

$$
\mathcal{L}_{\rho}=f(x)+y^T(Ax-b)+\frac{\rho}{2}\|Ax-b\|^2_2
$$

ADMM 针对上述优化问题，通过增加变量，把约束条件转化到原问题中，达到可以函数可分 (separable) 的目的，然后迭代求解时就可以进行分布式并行了。

$$
\begin{aligned}
\min&\ \sum^n_{i=1}f_i(x_i)+g(z)\\
\text{s.t.}&\ x_i-z=0
\end{aligned}
$$

其中 $g(z)=Ax-b$，这样变换的优势会体现在优化过程中。其迭代过程为

$$
\begin{aligned}
x_i^{k+1}=&\underset{x}{\arg\min}\left(f_i(x_i)+y_i^{k\top}(x_i-z^k)+\frac{\rho}{2}\|x_i-z\|^2_2\right)\\
z^{k+1}=&\underset{z}{\arg\min}\left(g(z)+\sum^N_{i=1}(-y_i^{k\top}z+\frac{\rho}{2}\|x_i-z\|^2_2)\right)\\
y^{k+1}=&y^k+\rho(x_i^{k+1}-z^{k+1})
\end{aligned}
$$

其中 $y^k$ 是拉格朗日乘子，可以对比之前的增广拉格朗日乘子法，ADMM 的核心就是加和拆，通过加变量来让原问题的函数可分，然后拆成并行迭代计算的形式。




#### 拉格朗日法 v.s. 期望最大化 EM 算法

拉格朗日法解决一个多变量优化问题时，需要将原问题拆解成多个子凸优化问题，然后用拉格朗日法去解决，这里多个变量交替优化的方法并不是 EM 算法。EM 算法是为了解决模型中存在隐变量的情况，求期望和期望最大化两步交替迭代。而优化问题中的交替是优化不同的变量。



### 具体优化例子

#### 1）偏导带有常数项

有约束的情况下偏导中含有常数项的优化问题，比如

$$
\begin{align}
\min_{\mathrm{w}} \ & (\mathrm{w}-\mu)^{\top}A(\mathrm{w}-\mu)\\
&\text{s.t.} \ \mathrm{w}^{\top}\mathrm{w}=1
\end{align}
$$

其中 $A$ 为协方差矩阵，$\mu$ 为固定向量，其拉格朗日函数偏导为

$$
\nabla_{\mathrm{w}}=2A(\mathrm{w}-\mu)-2\lambda \mathrm{w}=0
$$

该类型问题用拉格朗日乘子法暂时无解，这也展示了其有一定局限性。但是将该问题转换成

$$
\begin{align}
\min_{\mathrm{w},\mathrm{z}} \ & (\mathrm{w}-\mu)^{\top}A(\mathrm{w}-\mu)+\alpha(\mathrm{z}^{\top}\mathrm{z}-1)\\
&\text{s.t.} \ \mathrm{w}=\mathrm{z}
\end{align}
$$

其增广拉格朗日函数为

$$
l=(\mathrm{w}-\mu)^{\top}A(\mathrm{w}-\mu)+\alpha(\mathrm{z}^{\top}\mathrm{z}-1)+\text{tr}(Y_1^{\top}(\mathrm{w}-\mathrm{z}))+\frac {\theta} 2\|\mathrm{w}-\mathrm{z}\|_F^2
$$

然后用 ADMM 法分别优化 $\mathrm{w},\mathrm{z},Y_1,\theta$ 即可。

#### 2）多变量优化 [3]

一种降维方法的损失函数，$Y$ 为 onehot 标签，$E$ 为随机矩阵

$$
\begin{align}
\min_{W,E,G} &\ \|X-WG^{\top}\|_F^2+\alpha\|Y-EG^{\top}\|_F^2+\beta\|G\|_{2,1}\\
&\text{s.t.}\ G^{\top}G=I
\end{align}
$$

令其拉格朗日函数对 $W$ 和 $E$ 的偏导为 0，易得

$$
\begin{align}
W&=XG,\\
E&=YG\\
\end{align}
$$

在对 $W$ 的偏导为 0 计算简化过程中，需要将上述两个结果以及 $G^{\top}G=I$ 带入到原始问题中，可以得到仅含有 $G$ 的凸优化问题。

#### 3）非负矩阵分解 [4]

非负矩阵分解问题中，对于损失函数

$$
J=\|X-FG^{\top}\|^2=\text{tr}(X^{\top}X-2X^{\top}FG^{\top}+GF^{\top}FG^{\top})
$$

优化问题  $\min J\quad \text{s.t.}\ G\ge0$，拉格朗日函数为

$$
L(G)=\text{tr}(-2X^{\top}FG^{\top}+GF^{\top}F G^{\top}-\beta G^{\top})
$$

其中拉格朗日乘子 $\beta_{ij}$ 约束 $G_{ij}$ 非负。易得其偏导

$$
\nabla_G=-2X^{\top}F+2GF^{\top}F-\beta=0
$$

根据 KKT 条件知最优解满足互补松弛条件

$$
(-2X^{\top}F+2GF^{\top}F)_{ik}G_{ik}=\beta_{ik}G_{ik}=0
$$

可以得到收敛时 G 的更新公式

$$
G \gets G \cdot \sqrt{\frac {(X^{\top}F)^{+}+G(F^{\top}F)^{-}} {(X^{\top}F)^{-}+G(F^{\top}F)^{+}} }
$$

#### 4）逐行求偏导 [5]

动态图学习，对于优化问题

$$
\begin{align}
\begin{split}
\min_S &\ \alpha \mathrm{tr}\left(F^{\top} L F\right) + \lambda \left\|S\right\|_F^2 \\
&\text{s.t. } S^{\top} \mathbf{1}=\mathbf{1}, S \geq 0,
\end{split}
\end{align}
$$

优于约束条件中包括了对行和的约束，直接优化 $S$ 较为困难，因此这里逐行求偏导，原问题转换为

$$
\begin{align}
\begin{split}
\min _{\mathbf{s}_{i}} & \sum_{i}\left(\lambda \mathbf{s}_{i} \mathbf{s}_{i}^{\top}+\alpha \left\|f_{i}-f_{j}\right\|_{2}^{2} \mathbf{s}_{i}^{\top}\right) \\
&\text{s.t.} \ \mathbf{s}_{i} \mathbf{1}=1, \mathbf{s}_{i} \geq 0, 
\end{split}
\end{align}
$$

上述优化问题等效为以下问题的求解：

$$
\begin{align}
\min _{\mathbf{s}_{i} \geq 0}\ \|\mathbf{s}_{i}-\frac {d_i} {2 \lambda}\|_{2}^{2}, \quad \text { s.t. } \quad \mathbf{s}_{i} \mathbf{1}=1
\end{align}
$$

其中 $d_i=\alpha \left\|f_{i}-f_{j}\right\|_{2}^{2}$，可以用 KKT 条件逐行优化。

#### 5）多变量 ADMM 法 [6]

SPDA 是一个迁移学习领域的模型，其优化问题很有趣。其原始优化问题为

$$
\begin{align}
\min_{P,Z,M}&\ \frac 1 2 \|P^{\top}X_s-(Y_s+B\odot M)\|_F^2 + \alpha \|P\|_F^2 + \|Z\|_{\ast}+\lambda\|Z\|_1+\sigma\text{tr}(P^{\top}XLX^{\top}P),\\
&\text{s.t.}\ P^{\top}X_t=P^{\top}X_sZ,
\end{align}
$$

其中 $L$ 包括了拉普拉斯正则化以及 MMD 距离约束。转换后的优化问题

$$
\begin{align}
\min_{P,Z,E,Z_1,Z_2,M}&\ \frac 1 2 \|P^{\top}X_s-(Y_s+B\odot M)\|_F^2 + \alpha \|P\|_F^2 + \|Z_1\|_{\ast} + \beta \|E\|_1 + \lambda\|Z_2\|_1+\sigma\text{tr}(P^{\top}XLX^{\top}P),\\
&\text{s.t.}\ P^{\top}X_t=P^{\top}X_sZ+E,\ Z_1=Z,\ Z_2=Z
\end{align}
$$

表面上增加了优化变量，使得问题变复杂了，但是优化过程更好解了。其增广拉格朗日函数为

$$
\begin{align}
l=&\frac 1 2 \|P^{\top}X_s-(Y_s+B\odot M)\|_F^2 + \alpha \|P\|_F^2 + \|Z_1\|_{\ast} + \beta \|E\|_1 + \lambda\|Z_2\|_1+\sigma\text{tr}(P^{\top}XLX^{\top}P)\\
&+ \text{tr}(Y_1^{\top}(P^{\top}X_t-P^{\top}X_sZ-E)) + \text{tr}(Y_2^{\top}(Z-Z_1)) + \text{tr}(Y_3^{\top}(Z-Z_2))\\ 
&+ \frac {\mu} 2 \|P^{\top}X_t-P^{\top}X_sZ-E\|_F^2 + \frac {\mu} 2 (\|Z-Z_1\|_F^2+\|Z-Z_2\|_F^2)
\end{align}
$$

交替对 $P,Z,E,Z_1,Z_2,M$ 进行优化，可以得到各个子问题中变量的更新公式，最后是拉格朗日乘子的更新

$$
\begin{cases}
Y_1=Y_1+\mu (P^{\top}X_t-P^{\top}X_sZ-E)\ \\
Y_2=Y_2+\mu (Z-Z_1)\ \\
Y_3=Y_3+\mu (Z-Z_2)\ \\
\mu=\min (\rho\mu,\mu_{\max})
\end{cases}
$$

其过程是按照标准的 ADMM 流程来的。当原始问题较复杂时，利用 ADMM 可分布优化的性质，通过增加变量，把原问题中的项转为约束条件，或者把约束条件转化到原问题中。然后写出其 ALM 函数，然后分别优化其待优化变量以及拉格朗日系数。



**参考：**

\[1\] [ADMM note](<http://www.arvinzyy.cn/2017/11/01/ADMM-note/>)

\[2\] [广义拉格朗日与乘数法，以及ADMM之间有着怎样的关联？](https://www.zhihu.com/question/362843991/answer/951070094)

[3] Guo, Chenfeng, and Dongrui Wu. "Discriminative sparse generalized canonical correlation analysis (DSGCCA)." *2019 Chinese Automation Congress (CAC)*. IEEE, 2019.

[4] Ding, Chris HQ, Tao Li, and Michael I. Jordan. "Convex and semi-nonnegative matrix factorizations." *TPAMI* 32.1 (2008): 45-55.

[5] Wang, Lichen, Zhengming Ding, and Yun Fu. "Adaptive graph guided embedding for multi-label annotation." *IJCAI*. 2018.

[6] Xiao, Ting, et al. "Structure preservation and distribution alignment in discriminative transfer subspace learning." *Neurocomputing* 337 (2019): 218-234.

