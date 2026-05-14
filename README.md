# YOLOv8 行人检测

基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 的行人检测项目，支持自定义数据集训练、密集人群微调、视频推理。

## 文件说明

| 文件 | 用途 |
|------|------|
| `convert_ped.py` | 将 PennFudanPed 标注转为 YOLO 格式 |
| `fix.py` | 清洗 CrowdHuman 标签，只保留 person 类别 |
| `train_ped.py` | 在自定义行人数据集上训练 YOLOv8 |
| `weitiao.py` | 用 CrowdHuman 密集人群数据集微调模型 |
| `contiue.py` | 中断训练后继续训练 |
| `video.py` | 视频推理，带抗闪烁目标跟踪 |
| `crowdhuman.yaml` | CrowdHuman 数据集配置 |
| `mydata.yaml` | 自定义数据集配置 |
| `yolov8n.pt` | YOLOv8n 预训练权重 |

## 环境安装

```bash
pip install ultralytics opencv-python numpy
```

## 使用步骤

### 1. 准备数据集
两个数据集都需要自己下载
将 PennFudanPed 标注转为 YOLO 格式：

```bash
python convert_ped.py
```

清洗 CrowdHuman 标签（只保留行人）：

```bash
python fix.py
```

### 2. 训练

基础训练：

```bash
python train_ped.py
```

密集人群微调：

```bash
python weitiao.py
```

继续训练：

```bash
python contiue.py
```

### 3. 视频推理

```bash
python video.py
```

## 目录结构

```
├── mydata/            # 自定义数据集
│   ├── images/
│   └── labels/
├── CrowdHuman/        # CrowdHuman 数据集
│   ├── images/
│   └── labels/
├── convert_ped.py     # 数据集格式转换
├── fix.py             # 标签清洗
├── train_ped.py       # 训练脚本
├── weitiao.py         # 微调脚本
├── contiue.py         # 继续训练
├── video.py           # 视频推理
└── yolov8n.pt         # 预训练模型
|——result.csv          # 训练结果
```
## 项目特点

- 使用 YOLOv8n 进行轻量级行人检测
- 基于 CrowdHuman 提升密集人群鲁棒性
- 加入视频时序稳定策略减少框闪烁
- 支持中断恢复训练