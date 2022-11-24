---
layout: post
title: 机器学习科研方法论
date: 2020-10-03 14:53
tags: [ML, learning]
categories: 思考阅读
description: "机器学习科研方法论"
mathjax: true
---

* content
{:toc}
机器学习炼丹方法论。 <!--more-->

**Cham's Blog 首发原创**



许久未逛知乎，前几天看到天奇大佬的知乎文章 [机器学习科研的十年](https://zhuanlan.zhihu.com/p/74249758)，瞬间感觉我等渣渣的工作于理论于工程都不美，于是顺便找了些相关素材给自己提提醒。


1. [未来 3~5 年内，哪个方向的机器学习人才最紧缺？](https://www.zhihu.com/question/63883507)

2. [如果我没有那么优秀，我研究生阶段选择机器学习方向还有出路吗？](https://www.zhihu.com/question/63883507)

3. [How to do good research, get it published in SIGKDD and get it cited!](http://www.cs.ucr.edu/~eamonn/Keogh_SIGKDD09_tutorial.pdf)

摘录一些很有意思的观点，以明研究方向。

------



### 天奇大佬的经验

从一开始hack显卡代码的兴奋，到一年之后的焦虑，再到时不时在树下踱步想如何加旋转不变的模型的尝试，在这个方向上，我花费了本科四年级到硕士一年半的所有时间，直到最后还是一无所获。现在看来，当时的我犯了一个非常明显的错误 -- 常见的科学研究要么是问题驱动，比如“如何解决ImageNet分类问题”；要么是方法驱动，如 “RBM可以用来干什么”。**当时的我同时锁死了要解决的问题和用来解决问题的方案，成功的可能性自然不高**。

实验室的不少学长们曾经去香港和杨强老师工作，他们回来之后都仿佛开了光似地在科研上面突飞猛进。去香港之后，我开始明白其中的原因 -- 研究视野。经过几年的磨练，那时候的我或许已经知道如何去解决一个已有的问题，但是却缺乏其他一些必要的技能 -- **如何选择一个新颖的研究问题，如何在结果不尽人意的时候转变方向寻找新的突破点，如何知道整个领域的问题之间的关系等等**

在 CMU visit 的时候我见到了传说中的大神学长李沐，他和我感叹，现在正是大数据大火的时候，但是**等到我们毕业的时候，不知道时代会是如何，不过又反过来说总可以去做更重要的东西**。现在想起这段对话依然依然唏嘘不已。

这个方向有一个经典的方案GK-sketch的论文，但是只能够解决数据点没有权重的情况。**经过一两天的推导，我在一次去爬山的路上终于把结论推广到了有权重的情况**。(PS: ....)

到最后我选择了自己提出的一个课题，在这个曲线里面风险最高，回报也最高。我一直有一个理想，希望可以构建一个终身学习的机器学习系统，并且解决其中可能出现的问题。**这个理想过于模糊，但是我们想办法拿出其中的一个可能小的目标 -- 知识迁移**。

选择做什么眼光和做出好结果的能力一样重要，**眼界决定了工作影响力的上界，能力决定了到底是否到达那个上界**。交大时敖平老师曾经和我说过，一个人做一件简单的事情和困难的事情其实是要花费一样多的时间。因为即使再简单的问题也有很多琐碎的地方。要想拿到一些东西，就必然意味着要放弃一些其他东西，既然如此，为什么不一直选择跳出舒适区，选一个最让自己兴奋的问题呢。

但是总是觉得还缺少着什么 -- 系统的瓶颈依然在更接近底层的算子实现上。**暑假之后在去加州的飞机上**，我尝试在纸上画出为了优化矩阵乘法可能的循环变换，回来之后，我们决定推动一个更加大胆的项目 -- 尝试用自动编译生成的方式优化机器学习的底层代码。

两年间的不少关键技术问题的突破都是在有趣的时候发生的。**我在排队参观西雅图艺术博物馆的infinity mirror展览的途中**把加速器内存拷贝支持的第一个方案写在了一张星巴克的餐巾纸上。到后来是程序语言方向的同学们也继续参与进来。

我常想，如果我在焦虑死磕深度学习的时候我多开窍一些会发生什么，如果我并没有在实习结束的时候完成当时的实验，又会是什么。但现在看来，**很多困难和无助都是随机的涨落的一部分**，付出足够多的时间和耐心，随机过程总会收敛到和付出相对的稳态。



### 知乎回答

作为一个普通的科研工作者，我们可以从两个维度提高：思考的频率、广度和深度，执行力。1）把自己遇到的各种科研问题，不论多么不成熟都记下来在本子上，每过一阵子去翻看一遍学到的新知是否能够解决新问题。一边跑就一边天马行空想和这个主题相关的内容。你往往想着A主题，就会联想到B，时间长了就会有一些靠谱的点子出来。2）有了尚可的点子就先做实验，有了尚可的实验结果就写论文，今天发不了CVPR就先发ICIP，迭代式上升才是符合普通人的路线。如果想要奔着搞大新闻的目标做科研，往往会卡很久很久很久。但也不能总恰烂文章，必须逐步提高对自己的要求。（知乎：[微调](https://www.zhihu.com/people/breaknever)）



### Idealized Algorithm for Writing a Paper

**Find problem/data**

- Start writing  Start writing  (yes, start writing before and during research)
- Do research/solve problem
- Finish 95% draft
- Send preview to mock reviewers
- Send preview to the rival authors (virtually or literally) 
- ------ *one month before deadline* ------
- Revise using checklist.
- Submit



### What Makes a Good Research Problem?

- **It is important:** If you can solve it, you can make money, or save lives, or help children learn a new language, or...
- **You can get real data:** Doing DNA analysis of the Loch Ness Monster would be interesting, but…
- **You can make incremental progress:** Some problems are all all-or or-nothing. Such problems may be too risky for young nothing
- **There is a clear metric for success:** Some problems fulfill the criteria above, but it is hard to know when you are making progress on them



### Finding Research Problems

**Suppose you think idea** $X$ **is very good, then can you extend** $X$ **by…**
- Making it more accurate (statistically significantly more accurate)
- Making it faster (usually an order of magnitude, or no one cares) 
- Making it simpler
- Explaining why it works so well
- Making it an anytime algorithm
- Applying it in a novel setting (industrial/government track)
- Removing a parameter/assumption
- Making it an online (streaming) algorithm
- Making it work for a different data type (including uncertain data)
- Making it work for distributed systems
- Making it disk-aware (if it is currently a main memory algorithm)
- Making it work on low powered devices

------

另外加一些我自己的想法

### Paper Management

- 读好文章，benchmark 刷分的 $\rightarrow$ 提出好点子的 $\rightarrow$ 挖新坑的 $\rightarrow$ 推动领域关键问题的
- 阅读前首先检查实验设置是否合理，baseline 是否可信，可重复性是否强，否则理论不可信
- 根据 introduction 了解作者对问题的定义和描述，以及对所在领域的积累
- 算法层面不看方法特复杂且无公开源码的，一般是作者对问题的定义有冗余或保密缘由
- 阅读文章要尽量 abstract 以及实验部分当天看完，后续选择是否细读或者复现。



### Other Tips

- 别恰剩饭或者很明显的灌水，有些知乎网红都快把这个领域的年轻人带没了
- 不要把毕业论文对方向的统一性当作束缚，达到毕业要求后多做些自己感兴趣的
- 勿好高骛远，脚踏实地做好实验、阅读