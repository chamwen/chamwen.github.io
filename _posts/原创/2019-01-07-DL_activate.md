---
layout: post
title: 常用激活函数整理
date: 2019-01-07 16:18
tags: [sigmoid,ReLU]
categories: Deep-Learning
description: "深度学习各种激活函数总结"
mathjax: true
---

* content
{:toc}
其他博客上看到的，当做记个笔记。摘自 [各种激活函数整理总结](https://hellozhaozheng.github.io/z_post/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-%E7%9F%A5%E8%AF%86%E7%82%B9%E6%A2%B3%E7%90%86-%E5%90%84%E7%A7%8D%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90/) 。 <!--more-->

## 常用激活函数及其导数

| 激活函数 | 形式 | 导数形式 |
| --- | --- | --- |
| Sigmoid   |  $f(x) =\frac{1}{1+e^{-x}}$ | $f'(x)(1-f(x))$  |
| Tanh  | $f(x) = tanh(x)= \frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $f'(x) = 1-(f(z))^2$  |
| ReLU  | $f(x)=max(0,x)=\begin{cases} 0 & x \leq 0 \\ x & x>0 \end{cases}$  | $f'(x)=\begin{cases} 0 & x\leq 0 \\ 1 & x>0 \end{cases}$  |



## 常用激活函数及其导数的图像

Sigmoid

{:refdef: style="text-align: center"}
![default](\images\activate\ac_sigmoid.jpg){:height="60%" width="60%"}
{:refdef}

Tanh

{:refdef: style="text-align: center"}
![default](\images\activate\ac_tanh.jpg){:height="60%" width="60%"}
{:refdef}

ReLU

{:refdef: style="text-align: center"}
![default](\images\activate\ac_relu.jpg){:height="60%" width="60%"}
{:refdef}



## 为什么需要激活函数

### 标准说法

这是由激活函数的性质所决定来，一般来说，激活函数都具有以下性质:

- **非线性：** 首先，线性函数可以高效可靠对数据进行拟合，但是现实生活中往往存在一些非线性的问题 (如 XOR)，这个时候，我们就需要借助激活函数的非线性来对数据的分布进行重新映射，从而获得更强大的拟合能力 (这个是最主要的原因，其他还有下面这些性质也使得我们选择激活函数作为网络常用层)。
- **可微性：** 这一点有助于我们使用梯度下降发来对网络进行优化
- **单调性：** 激活函数的单调性在可以使单层网络保证网络是凸的
- **$f(x) \approx x：$** 当激活满足这个性质的时候，如果参数初值是很小的值, 那么神经网络的训练将会很高效 (参考 ResNet 训练残差模块的恒等映射)；如果不满足这个性质，那么就需要用心的设值初始值( **这一条有待商榷** )

如果不使用激活函数，多层线性网络的叠加就会退化成单层网络，因为经过多层神经网络的加权计算，都可以展开成一次的加权计算

### 更形象的解释

对于一些线性不可分的情况，比如 XOR，没有办法直接画出一条直线来将数据区分开，这个时候，一般有两个选择：如果已知数据分布规律，那么可以对数据做线性变换，将其投影到合适的坐标轴上，然后在新的坐标轴上进行线性分类；而另一种更常用的办法，就是使用激活函数，以 XOR 问题为例，XOR 问题本身不是线性可分的，https://www.zhihu.com/question/22334626

### 用ReLU解决XOR问题.

首先，XOR 问题如下所示：

| $x_1$ | $x_2$ | y |
| --- | --- | --- |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |
| 0 | 0 | 0 |

首先构造一个简单的神经网络来尝试解决XOR问题，网络结构如下图所示：

{:refdef: style="text-align: center"}
![default](\images\activate\ac_xor.jpg){:height="30%" width="30%"}
{:refdef}

先来看看不使用激活函数时的情况，当不使用激活函数时，整个网络的函数表达式如下所示：

$$
y = f(x_1, x_2; W, c, w ,b) = [w_1, w_2] \bigg( \bigg[\begin{matrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{matrix} \bigg] \Big[\begin{matrix} x_1 \\ x_2 \end{matrix} \Big]+ \Big[\begin{matrix} c_1 \\ c_2 \end{matrix} \Big] \bigg) + b \\= (w^TW^T)x + (w^Tc+b) = w'^Tx+b'
$$

可以看到，多层无激活函数的网络叠加，首先是会退化成单层网络，而对于单层网络，求解出来的参数 $w'$ 和 $b'$ 无法对非线性的数据进行分类。
再来看看进入 ReLU 以后，是如何解决 XOR 问题的，首先，引入后的公式如下所示：
$$
y = f(x_1, x_2; W, c, w ,b) = [w_1, w_2] max \bigg(0 , \bigg[\begin{matrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{matrix} \bigg] \Big[\begin{matrix} x_1 \\ x_2 \end{matrix} \Big]+ \Big[\begin{matrix} c_1 \\ c_2 \end{matrix} \Big] \bigg) + b
$$

可以看到，此时函数是无法化简，因为此时引入了非线性的 ReLU 函数，于是，就可以求得一个参数组合 ${w,W,c,b}$ 使得对于特定的输入$x_1, x_2$，能够得到正确的分类结果 $y$。至于这个参数组合具体是什么，这是需要通过梯度下降来不断学习的，假如我们现在找到了一组参数如下 (当然不一定是最优的)，来看看这组参数具体是如何解决XOR问题的：

$$
W=\bigg[ \begin{matrix} 1 & 1 \\ 1 & 1 \end{matrix} \bigg],\ c =\Big[ \begin{matrix} 0 \\ -1 \end{matrix}  \Big],\ w =\Big[ \begin{matrix} 1 \\ -1 \end{matrix} \Big],\ b = 0
$$

带入计算易知其可以实现 XOR 逻辑。

## 关于各个激活函数的比较和适用场景

**神经元饱和问题：** 当输入值很大或者很小时，其梯度值接近于 0，此时，不管从深层网络中传来何种梯度值，它向浅层网络中传过去的，都是趋近于0的数，进而引发梯度消失问题

**zero-centered：** 如果数据分布不是 zero-centered 的话就会导致后一层的神经元接受的输入永远为正或者永远为负，因为 $\frac{\partial f}{\partial w} = x$，所以如果 x 的符号固定，那么 $\frac{\partial f}{\partial w}$ 的符号也就固定了，这样在训练时，weight 的更新只会沿着一个方向更新，但是我们希望的是类似于 zig-zag 形式的更新路径 (关于非 0 均值问题，由于通常训练时是按 batch 训练的， 所以每个 batch 会得到不同的信号，这在一定程度上可以缓解非 0 均值问题带来的影响，这也是 ReLU 虽然不是非 0 均值，但是却称为主流激活函数的原因之一)

| 激活函数 | 优势 | 劣势 | 适用场景 |
| --- | --- | --- | --- |
| Sigmoid  | 可以将数据值压缩到 [0,1] 区间内 |  1. 神经元饱和问题  <br>2.sigmoid 的输出值域不是 zero-centered 的  <br>3. 指数计算在计算机中相对来说比较复杂 | 在 logistic 回归中有重要地位  |
| Tanh  | 1. zero-centered：可以将 $(-\infty, +\infty)$ 的数据压缩到 $[-1,1]$ 区间内 <br> 2.完全可微分的，反对称，对称中心在原点  | 1. 神经元饱和问题 <br>2. 计算复杂 | 在分类任务中，双曲正切函数（Tanh）逐渐取代 Sigmoid 函数作为标准的激活函数  |
| ReLU  | 1. 在 $(0,+\infty)$，梯度始终为1，没有神经元饱和问题 <br>2. 不论是函数形式本身，还是其导数，计算起来都十分高效 <br />3. 可以让训练过程更快收敛 (实验结果表明比 sigmoid 收敛速度快 6倍) <br>4. 从生物神经理论角度来看，比 sigmoid 更加合理 | 1. 非 zero-centered   <br>2. 如果输入值为负值, ReLU 由于导数为 0，权重无法更新，其学习速度可能会变的很慢，很容易就会"死"掉 (为了克服这个问题，在实际中，人们常常在初始化 ReLU 神经元时，会倾向于给它附加一个正数偏好，如0.01) | 在卷积神经网络中比较主流  |



## 其他要点

### sigmoid 和 softmax 区别

sigmoid 是将一个正负无穷区间的值映射到 (0,1) 区间，通常用作二分类问题，而 softmax 把一个 k 维的实值向量映射成一个 $(b_1,b_2,...,b_k)$，其中 $b_i$ 为一个 0~1 的常数，且它们的和为 1，可以看作是属于每一类的概览，通常用作多分类问题。在二分类问题中，sigmoid 和 softmax 是差不多的，都是求交叉熵损失函数，softmax 可以看作是 sigmoid 的扩展，当类别 k 为 2 时，根据 softmax 回归参冗余的特点，可以将 softmax 函数推导成 sigmoid 函数。


