# YOLO26 详解：新一代实时目标检测模型

YOLO26 是 Ultralytics 于 2026 年 1 月发布的新一代 YOLO 视觉模型系列，主要面向实时目标检测、实例分割、姿态估计等计算机视觉任务。它的重点不是单纯增加网络规模，而是让模型更轻量、推理更高效，并简化在边缘设备和实际应用中的部署。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs+1



如果你了解 YOLOv8、YOLO11，可以把 YOLO26 理解为一次面向部署效率和端到端检测的升级。

## 一、YOLO26 能做什么？

[A Weekend AI Project: Object Detection with YOLO on PC and Raspberry Pi | by Dmitrii Eliuseev | TDS Archive | Medium](https://images.openai.com/static-rsc-4/eWj0aQaNIpJvVQ0Qq0kt7C6O2V25X1VW_XhnlQj4m400pkLZHzc5WAA1kkxmgeKVER_JO8iDfssICJOGZjfjE0urUmrXaZ8FlJgXqAkyf5md1W0y8emS8xM_qmveZRC3ELvISp922DL1nuEYnQvwtpV5WABKGlEyNCqiGKxPg4c?purpose=inline)

1. 目标检测（Object Detection）

识别图像或视频中的人、车辆、产品等目标，同时给出类别、置信度和边界框。适用于工业质检、交通监控、机器人视觉等场景。

[Real Time Image Segmentation Using 5 Lines of Code - KDnuggets](https://images.openai.com/static-rsc-4/cE1ttP5IDq-86kRqAW6DYldgEZ1Gvt44CnntWz5YEJLzceY2rES53k741EeDmoq4VL9D09r9D2JXOG0G2kBkEjcqgu668zOU7HFCTJWJ8w0ZcgS6ta2VJC1V2bnSgBztmucxYEpNEtUZ9dEl2nvNc2HMSTpGsl47F37iZlIJEPg?purpose=inline)

2. 图像分割（Segmentation）

不只是框出物体，还可以识别物体对应的像素区域，适合缺陷定位、医学图像分析和精细化视觉任务。

[WoS上yolov8人体姿态检测_yolov8n-pose.pt在那里下载-CSDN博客](https://images.openai.com/static-rsc-4/IX14YLDhVfkwDObVi4C8C9n2gyfXot2kua7ym4HA3g1NIVzw-04ebpkRfEj_6u9g3WloxxgLmHIpW3-O44LeM4gSDp411MDIHlcLNIKBp5swnNct5WPUaJs0UpoAm4nLQ8nZ6cnD8v1RmNx7g4euUSTGMAjNugQDkBNeWDHEqzw?purpose=inline)

3. 姿态估计（Pose Estimation）

检测人体或其他对象的关键点，可用于动作分析、运动识别和人机交互。

[The Only YOLOv11 Multi-Labeling Guide You’ll Ever Need](https://images.openai.com/static-rsc-4/6MhtCAM4ubBDdvpUF7ayy8XWBWcs-huqeKaoWZKt-vXuRxoGlGlPq_ityCLqyqnUe7eCogIattlCnrl5Ew3k3Y90Cn-7z3x2rQ38DWJBK0Bbls6i9e-083EP4QfEJ3cI89kURPVI-vF26Iaj2ZBrX4YNiekrGm8ANCKFiHfYkEw?purpose=inline)

4. 其他视觉任务

还支持图像分类、旋转目标检测（OBB）、单目深度估计等任务。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs



## 二、YOLO26 的核心技术改进

### 1. 支持端到端、免 NMS 推理

传统 YOLO 检测器通常会先生成大量候选框，再通过 NMS（非极大值抑制）去除重复框。

YOLO26 提供了一种 One-to-One 检测头，可以直接输出最终检测结果，免去单独的 NMS 后处理步骤。

传统检测流程

输入图像

特征提取

候选框

NMS

最终结果

YOLO26 免 NMS 模式

输入图像

特征提取

One-to-One 检测头

最终结果

示意图。YOLO26 默认仍采用 One-to-Many 检测头与 NMS；需要显式设置 nms=False 才会选择免 NMS 模式。

它的实际意义是：

- 减少后处理开销。
- 简化 ONNX、TensorRT 等推理部署流程。
- 在边缘计算和低延迟应用中提供更灵活的性能选择。

需要注意：免 NMS 不代表精度必然更高。官方文档指出，默认的 One-to-Many 模式在一些情况下具有略高的检测精度。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics+1



### 2. 移除 DFL，简化检测头

DFL（Distribution Focal Loss）是一种用于边界框回归的机制，YOLOv8 和 YOLO11 都采用了它。

YOLO26 移除了这部分机制，直接回归边界框坐标，减少检测头的复杂度。

优势主要有：

- 减少部分计算和模型结构开销。
- 简化模型导出后的计算图。
- 便于在不同推理后端上进行部署优化。

但这不意味着所有任务、所有硬件上都会获得同等幅度的加速。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics



### 3. 更新训练策略

YOLO26 还引入了几项训练方面的改进：

| 技术               | 主要作用                        |
| ---------------- | --------------------------- |
| MuSGD            | 将 Muon 与 SGD 的优化思想结合，用于模型训练 |
| Progressive Loss | 逐步调整训练损失的侧重点                |
| STAL             | 改善小目标的正样本分配和监督覆盖            |

这些改进旨在提升训练效果，并使训练目标更好地适配实际推理过程。它们并不意味着小目标漏检问题会自动消失，实际表现仍取决于数据集、目标尺寸、图像分辨率和训练设置。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs



## 三、YOLO26 有哪些模型版本？

YOLO26 的常规模型分为五个尺寸，分别是 n、s、m、l、x。

| 模型      | 参数量约  | 特点      | 适用方向        |
| ------- | ----- | ------- | ----------- |
| YOLO26n | 2.4M  | 最轻量     | 边缘设备、嵌入式    |
| YOLO26s | 9.5M  | 轻量与精度折中 | 实时检测、普通 GPU |
| YOLO26m | 20.4M | 中等规模    | 精度与速度平衡     |
| YOLO26l | 24.8M | 更高检测精度  | GPU 推理、复杂场景 |
| YOLO26x | 55.7M | 最高精度潜力  | 高算力服务器、精度优先 |

参数量为官方文档中融合后的检测模型统计值；不同任务的模型大小会有所不同。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs



如何选择？

- 算力有限，或部署在边缘设备：先试 `YOLO26n`。
- 需要速度和精度兼顾：先试 `YOLO26s`。
- 使用 NVIDIA GPU，且对精度要求较高：可以测试 `YOLO26m` 或 `YOLO26l`。
- 需要尽可能高的检测精度，并且算力充足：考虑 `YOLO26x`。

模型越大不一定越适合你的项目。例如，在低功耗设备上，较小模型可能有更好的实际帧率和响应时间。

## 四、YOLO26 的性能怎么样？

官方公布了 COCO 数据集上的目标检测结果，以下是五个检测模型的指标：

| 模型      | COCO mAP50–95 | T4 TensorRT 延迟 |
| ------- | ------------- | -------------- |
| YOLO26n | 40.9          | 1.7 ms         |
| YOLO26s | 48.6          | 2.5 ms         |
| YOLO26m | 53.1          | 4.7 ms         |
| YOLO26l | 55.0          | 6.2 ms         |
| YOLO26x | 57.5          | 11.8 ms        |

来源：[Ultralytics YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26?utm_source=chatgpt.com)。测试输入为 640×640，延迟来自 NVIDIA T4 TensorRT 测试配置。mAP50–95 越高通常表示检测精度越好，延迟越低通常表示推理越快。

这些数据说明，模型规模增大时，通常能获得更高的检测精度，但推理延迟和计算成本也会增加。

注意，这里的毫秒数是特定硬件和推理引擎下的测试结果，不能直接等同于你自己电脑上的 FPS。实际速度还受到 GPU、CPU、图像尺寸、批次大小、数据预处理及后处理等因素影响。

## 五、如何使用 YOLO26？

如果你之前用过 YOLOv8 或 YOLO11，YOLO26 的基本使用方式会比较熟悉。

### 1. 安装环境

建议先创建 Python 虚拟环境，再安装 Ultralytics：

```
pip install -U ultralytics
```

### 2. 使用预训练模型进行检测

```
from ultralytics import YOLO# 加载预训练模型model = YOLO("yolo26n.pt")# 检测图片results = model("bus.jpg")# 显示检测结果results[0].show()
```

首次加载时，如果本地没有模型权重，程序通常会尝试下载相应文件。

### 3. 使用免 NMS 模式

```
from ultralytics import YOLOmodel = YOLO("yolo26n.pt")results = model.predict(    source="bus.jpg",    nms=False)results[0].show()
```

这里的 `nms=False` 会选择 One-to-One 检测头。建议在自己的数据集上分别测试默认模式和免 NMS 模式，再根据精度与延迟决定使用哪一种。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs+1



### 4. 使用自己的数据集训练

假设你的数据集已经整理成 Ultralytics YOLO 格式，并创建了 `data.yaml`：

```
from ultralytics import YOLOmodel = YOLO("yolo26n.pt")model.train(    data="data.yaml",    epochs=100,    imgsz=640,    batch=16,    device=0)
```

其中：

- `data`：数据集配置文件。
- `epochs`：训练轮数。
- `imgsz`：输入图像尺寸。
- `batch`：批次大小，需要根据显存调整。
- `device=0`：使用编号为 0 的 GPU；如果使用 CPU，可设置 `device="cpu"`。

训练前还需要确认类别数量、标注格式、训练集与验证集划分正确。具体参数应根据数据规模和硬件进行调整。

## 六、YOLO26 与 YOLOv8、YOLO11 怎么选？

| 比较维度     | YOLOv8    | YOLO11     | YOLO26           |
| -------- | --------- | ---------- | ---------------- |
| 主要定位     | 通用实时视觉    | 更现代的通用视觉模型 | 强调端到端推理与部署效率     |
| DFL      | 使用        | 使用         | 移除               |
| 免 NMS 推理 | 不是其标准核心特性 | 不是其标准核心特性  | 原生提供可选的免 NMS 检测头 |
| 自定义数据集训练 | 支持        | 支持         | 支持               |
| 迁移旧项目    | 生态成熟      | 迁移成本通常较低   | 需要验证兼容性及性能       |

这里的对比是架构和使用方式上的概括，并不意味着 YOLO26 在所有数据集上都必然优于旧版本。尤其是已有生产系统，最好用相同数据、输入分辨率和硬件进行公平测试。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs+1



## 七、YOLO26 适合哪些实际项目？

[Design Defects Inspection in Manufacturing with Machine Vision AOI](https://images.openai.com/static-rsc-4/nR02uHosulnqHpQx4LFSCpUHIEEf33JU_F7HpxPUc_gdEFTPMOyQj_2Y4TlejVA_rMBN9wk-v2PAAOKbgtVgMcUkKxHTJYD5rMscn2a5ku3c9CRvXwc_npWJyKO2V0So57zVSTwWAQLbUgIDmha-Sluk_7PFgFB2HFKszVSvll4?purpose=inline)

工业视觉检测

检测产品缺陷、零件、异物和装配状态。小缺陷任务需要特别关注输入分辨率、标注质量与小目标检测效果。

[ASU entrepreneurs develop smart street cameras - Blog](https://images.openai.com/static-rsc-4/AS-_OJZ4RmT1V54bH87sYjhWk9eHYvhcMhnFUTSHTiEWOIrZ5C7YWDg7RrSmkalkokUbckMQMG3mjOxqU0rm5EeiTZtt4260aAVPX40lToa9S6eC3Xs-lJr6v-XSGt_wQk9L1Z6a5cD42fw9fgPD4cn-QeWVvasNa7E7DwNqFvo?purpose=inline)

交通与视频分析

检测车辆、行人及其他交通参与者，并可与跟踪算法组合实现计数和轨迹分析。

嵌入式与边缘 AI

可先从 YOLO26n 开始，评估模型在目标设备上的速度、内存占用和功耗，再决定是否升级到更大的模型。

还需要注意，如果你打算将 YOLO26 用于商业项目，建议在选型时核对 Ultralytics 当前的许可证条款。官方文档列出了 AGPL-3.0 与 Enterprise 等许可选项，不能简单地认为所有使用场景都没有授权要求。

[image](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com\&sz=32)

Ultralytics Docs



## 总结

YOLO26 的主要价值可以归纳为三个方面：更简化的检测头、可选的免 NMS 推理，以及针对训练和多种视觉任务的优化。 它适合新建的实时视觉项目，也值得已有 YOLO 项目进行性能对比测试。

如果你准备进一步学习或实战，我建议先明确你的具体方向：

你想从哪个方向深入了解 YOLO26？

自定义数据集训练：从数据标注到模型训练

网络结构：Backbone、Neck、Head 和核心改进

与 YOLOv8、YOLO11 的详细实验对比

部署实战：ONNX、TensorRT、Jetson 或边缘设备

小目标检测与工业缺陷检测

你目前的基础？

刚接触目标检测

了解 YOLO，能运行 Python 代码

已经训练过 YOLO 模型

你主要使用什么计算设备？

NVIDIA GPU 电脑或服务器

普通 CPU 电脑

Jetson、树莓派或其他边缘设备

暂时没有设备，准备使用云端环境

开始深入学习
