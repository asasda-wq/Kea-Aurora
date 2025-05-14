<div align="center">
<h1>Kea-Aurora</h1>

 <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-orange'></a> &nbsp;&nbsp;&nbsp;
 <a><img src='https://img.shields.io/badge/python-3.9-blue'></a> &nbsp;&nbsp;&nbsp;

</div>

### 简介

Kea 是一个通用的测试工具，通过基于性质的测试发现移动（GUI）应用中的功能性错误, 目前支持 Android 和 HarmonyOS。
Aurora 是一个自动输入生成工具，它使用计算机视觉和自然语言处理技术来导航给定的屏幕。 
本项目在 Kea 框架基础上，结合了 Aurora 的页面分类器模型以及 LLM 对 UI 界面的理解能力
，完善了 Kea 的 LLM 探索策略，提高了 Kea 在遇到 UI 陷阱时的脱离能力。

#### 主要工作

在该项目中，我们为优化 kea 的 LLM 策略做出的主要工作有：
1. 通过界面截图的 dhash 值判断相似度，相似度大于阈值时判断陷入 UI 陷阱
2. 提取页面组件的属性作为 LLM 的输入以及生成输出备选事件
3. 通过 Aurora 的页面分类器给 LLM 生成启发式的提示词
4. 针对复杂的表格类型页面根据 LLM 提示填充表格
5. 提出了一个基于 utg 的路径多样化度量方法

### 安装和使用

**Windows环境配置**
+ 环境配置相关问题可以通过查看 [Kea 文档](https://github.com/asasda-wq/Kea-Aurora/blob/main/kea.pdf)寻找解决方案

1. 安装 Python 3.9
2. 安装 `adb` 命令行工具
+ 安装 [Android Studio](https://developer.android.com/studio)
+ 将 Android sdk 下的内容加入环境变量
+ 下载 [cmdline-tools](https://developer.android.com/tools?hl=zh-cn)，并根据指引将其加入环境变量
3. 使用以下指令创建一个安卓虚拟机
```
sdkmanager "build-tools;29.0.3" "platform-tools" "platforms;android-29"
sdkmanager "system-images;android-29;google_apis;x86"
avdmanager create avd --force --name Android10.0 --package 'system-images;android-29;google_apis;x86' --abi google_apis/x86 --sdcard 1024M --device "pixel_2"
```
4. 安装 cuda11.6

**工具安装**

1. 输入以下命令安装 Kea 并下载相应 python 环境。

```
git clone https://github.com/ecnusse/Kea.git
cd Kea
pip install -e .
```
2. 下载页面分类器的[模型权重文件](https://pan.baidu.com/s/1mDFxzcK5z7NYrKPV-VPyQw?pwd=cwy2)，
将其地址输入 Classify/ScreenClassifier.py 的第7行相应位置
3. 将希望调用 LLM 的 api 输入 kea/input_policy.py 的第1288行相应位置


**快速开始**

使用以下指令可以快速使用 LLM 策略对 omninotes 软件进行测试，其他 kea 相关的命令行参数可在 kea 文档中查看
```
kea -f example/test_properties/example_property.py -a example/apps/omninotes.apk -p llm
```

### 结果测试
使用两个工具:[acvtool](https://github.com/pilgun/acvtool)、[KeaPlusEvaluation](https://github.com/mengqianX/KeaPlusEvaluation)
对我们的方法进行了评估

#### [acvtool](https://github.com/asasda-wq/Kea-Aurora/blob/main/files/acvtool_result.txt)
对random+重启,llm,llm+分类三种方法在五个应用trivago, omninotes, amaze, FoxNews, learn上进行了对比实验，每种方法对每个app测试5次取覆盖度平均值，每次测试500个 UI 事件

#### [KeaPlusEvaluation](https://github.com/asasda-wq/Kea-Aurora/blob/main/files/keaplusevaluation_result.xlsx)
对random,random+重启,llm,llm+分类四种方法在五个应用trivago, omninotes, amaze, FoxNews, learn上进行了对比实验，每种方法对每个app测试5次取覆盖度平均值，每次测试500个 UI 事件

### Docs

[设计文档](https://github.com/asasda-wq/Kea-Aurora/blob/main/files/kea%2Bclassifier.docx)

[测试报告](https://github.com/asasda-wq/Kea-Aurora/tree/main/files)

### Kea  参考的开源工具

- [Kea](https://github.com/ecnusse/Kea)
- [Aurora](https://github.com/safwatalikhan/AURORA)


