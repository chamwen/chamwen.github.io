---
layout: post
title: 解决：任务栏上点Matlab会打开两个图标
date: 2020-05-31 9:25
tags: [windows]
categories: 经验分享
description: "任务栏出现两个一样的图标，为什么点击任务栏快捷方式会打开另外一个，win10系统任务栏同一程序两个图标的问题，Matlab"
mathjax: true
---

* content
{:toc}

## 什么情况

**在网上搜索了一些关键词，很少有 refer 到重点的**

任务栏出现两个一样的图标

为什么点击任务栏快捷方式会打开另外一个

win10系统任务栏同一程序两个图标的问题<!--more-->

Win10任务栏锁定程序图标之后打开程序就会有两个一样的图标？

matlab taskbar fixed a new icon

windows software taskbar fixed a new icon

windows software taskbar fixed a new icon ImplicitAppShortcuts



## 什么问题

1）黑框的原因是因为快捷方式是从 `C:\Program Files\Matlab\bin` 下的 `matlab.exe` 文件创建的，启动时需要验证证书；

2）打开新图标的原因，是已经添加到任务栏的图标为 `bin\win64\matlab.exe`，调用时还是从 `bin\matlab.exe` 下启动，应该是以为快捷方式不是自动生成，而是手动创建的，和任务栏不兼容。



## 怎么解决

**标准的添加任务栏图标的 pipeline 是**

1）将 matlab 安装目录 `C:\Program Files\Matlab\bin\win64\` 下的 `matlab.exe` 文件发送快捷方式到桌面；

2）右键桌面快捷方式，**修改属性，目标修改为** `"C:\Program Files\Matlab\bin\win64\matlab.exe" -c "C:\Program Files\Matlab\licenses\license_standalone.lic"`，起始位置定义为主代码的工作目录；

3）最关键的一步，打开 `C:\Users\wenzn\AppData\Roaming\Microsoft\Internet Explorer\Quick Launch\User Pinned`，其中有两个文件，`TaskBar`是任务栏上已经添加的正常的快捷方式，`ImplicitAppShortcuts` 里面是不能显式的通过点击添加到任务栏上的图标就启动的，换句话说你之前添加的快捷方式做的修改识别不了，还是默认打开 `ImplicitAppShortcuts` 下的 `MATLAB R2019a`（注意不同版本这里的文件名略微不一样）。**这个时候右键修改 `MATLAB R2019a` 的属性，把刚才的目标和起始位置粘贴到这里；或者把桌面上的快捷方式改成同样的名字覆盖过来。然后，将 `MATLAB R2019a` 发送到任务栏**。



## 问题总结

对于其他第三方软件在任务栏上出现两个图标都可以在 `C:\Users\wenzn\AppData\Roaming\Microsoft\Internet Explorer\Quick Launch\User Pinned` 中进行修改，特别是隐式调用的，需要修改 `ImplicitAppShortcuts` 下的快捷方式属性。



**参考资料：**

1. [Fix: Double Chrome icon on the Task bar in Windows 10](https://thegeekpage.com/fix-double-chrome-icon-on-the-task-bar-in-windows-10/)