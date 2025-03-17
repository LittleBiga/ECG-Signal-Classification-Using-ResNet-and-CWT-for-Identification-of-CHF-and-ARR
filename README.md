# 一、数据声明与引用要求 Data Attribution and Citation Requirements

## 1.1 数据来源声明 Data Provenance
本项目中使用的ECG数据来源于MathWorks Wavelet Toolbox示例库[1]，其原始数据基于以下三个PhysioNet数据库构建：  
The ECG data used in this project originates from MathWorks Wavelet Toolbox example repository [1], with raw data constructed from three PhysioNet databases:
- MIT-BIH Arrhythmia Database [2][3]
- MIT-BIH Normal Sinus Rhythm Database [2]
- BIDMC Congestive Heart Failure Database [2][4]

## 1.2 版权声明 Copyright Notice
此数据遵循MathWorks BSD-3条款[5]及PhysioNet复制政策：  
This data adheres to MathWorks BSD-3 Clause [5] and PhysioNet copying policy:

```text
Copyright (c) 2016, The MathWorks, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. In all cases, the software is, and all modifications and derivatives of the software shall be, licensed to you solely for use in conjunction with MathWorks products and service offerings. 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 1.3 数据处理说明 Data Processing
原始数据经过以下标准化处理（完整描述请参见项目中的Modified_physionet_data.txt文件）：  
The raw data underwent the following standardization processes (full description in Modified_physionet_data.txt):

1. **应用PhysioNet .info文件定义的缩放比例**  
   Applied scaling defined in PhysioNet .info files
2. **重采样至128Hz统一频率**  
   Resampled to common rate of 128Hz
3. **分割双通道ECG记录为独立数据**  
   Separated dual-channel ECG into individual records
4. **截断为65536样本统一长度**  
   Truncated to 65536-sample uniform length

## 1.4 强制引用文献 Mandatory Citations
在学术出版物中必须包含以下引用（按出现顺序编号）：  
The following citations must be included in academic publications (numbered sequentially):

1. [MathWorks ECG Data Repository](https://github.com/mathworks/physionet_ECG_data)  
2. Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000;101(23):e215-e220.  
3. Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol. 2001;20(3):45-50.  
4. Baim DS, et al. Survival with oral milrinone therapy. J Am Coll Cardiol. 1986;7(3):661-670.  
5. [MathWorks BSD-3 License](https://docs.oracle.com/cloud/latest/big-data-discovery-cloud/BDDLG/cl_bsd_license.htm#BDDLG-concept_1F381CA11423445A90B7C3D6CB4DF612)

# 二、ECG信号转换为RGB图像 | Convert ECG Signal to RGB Image

## 2.1 📁 文件说明 | File Description

### 2.1.1 `ecg2cwtscg.m` - 小波散射图像生成核心函数 | Core Function for Wavelet Scattering Image Generation
**功能**：将ECG信号通过连续小波变换转换为散射图像 | **Function**: Convert ECG signals to scattering images via continuous wavelet transform

**参数说明** | **Parameters**:
- `ecgdata`: 输入的ECG信号矩阵（行表示样本，列表示信号点） | Input ECG signal matrix (rows: samples, columns: data points)
- `cwtfb`: 预定义的小波滤波器组（使用`cwtfilterbank`创建） | Predefined wavelet filter bank (created with `cwtfilterbank`)
- `ecgtype`: 心电类型标识符（'ARR'心律失常/'CHF'心衰/'NSR'正常节律） | ECG type identifier ('ARR'Arrhythmia/'CHF'Heart Failure/'NSR'Normal Rhythm)

**核心处理流程** | **Core Process**:
- `定义图像参数`：生成128色阶的JET色谱图 | Define image parameters: Generate 128-color JET colormap
- `根据ECG类型创建目标存储路径（示例路径需根据实际调整）` | Create target storage paths based on ECG types (sample paths should be adjusted)
- `信号分帧处理`：每512点为一个信号片段（符合MIT-BIH数据库标准）| Signal framing: 512-point segments (MIT-BIH standard compliant)
- `小波变换计算`：cfs = abs(cwtfb.wt(ecgsignal)) 获取时频特征 | Wavelet transform: cfs = abs(cwtfb.wt(ecgsignal)) for time-frequency features
- `图像标准化与转换`：将小波系数转换为227x227像素的RGB图像（适配CNN输入尺寸）| Image standardization: Convert coefficients to 227x227 RGB images (CNN input compatible)

### 2.1.2 `main_script.m` - 数据处理主程序 | Main Data Processing Script
**功能**：数据集预处理与任务分发 | **Function**: Dataset preprocessing and task distribution

**执行流程** | **Workflow**:

#### 数据加载 | Data Loading
```matlab
load('ECGData.mat');  % 加载标准化ECG数据集[1,2](@ref)
data = ECGData.Data;  % 162个样本（96 ARR/30 CHF/36 NSR）
labels = ECGData.Labels;
```

#### 数据分割 | Data Segmentation
```matlab
ARR = data(1:96,:);  % 心律失常数据
CHF = data(97:126,:); % 心衰数据
NSR = data(127:162,:); % 正常节律数据
```
#### 文件系统初始化 | Filesystem Initialization
```matlab
mkdir('ecgdataset2'); % 主存储目录
mkdir('ecgdataset2\arr'); % 心律失常图像子目录
mkdir('ecgdataset2\chf'); % 心衰图像子目录
mkdir('ecgdataset2\nsr'); % 正常节律图像子目录
```
#### 批量转换执行 | Batch Conversion Execution
```matlab
ecg2cwtscg(ARR, fb, 'ARR'); % 处理心律失常数据
ecg2cwtscg(CHF, fb, 'CHF'); % 处理心衰数据
ecg2cwtscg(NSR, fb, 'NSR'); % 处理正常数据
```
# 三、基于ResNet的心电图分类系统 | ECG Classification System Based on ResNet

## 3.1 项目概述 | Project Overview
- 目标类别：ARR（房颤）、CHF（心力衰竭）、NSR（正常窦性心律） | Target Classes: ARR (Atrial Fibrillation), CHF (Congestive Heart Failure), NSR (Normal Sinus Rhythm)
- 技术特征 | Technical Features:
  - 数据增强 | Data Augmentation
  - 类别平衡 | Class Balancing
  - 学习率调度 | Learning Rate Scheduling
  - 多维度评估指标 | Multi-dimensional Evaluation Metrics

## 3.2 核心功能 | Core Features

### 3.2.1 数据处理模块 | Data Processing Module
- 自动数据集划分（训练/验证/测试 6:2:2） | Automatic dataset split (Train/Val/Test 6:2:2)
- 图像标准化处理（224x224分辨率） | Image standardization (224x224 resolution)
- 数据增强策略（随机缩放、归一化） | Augmentation strategies (Random scaling, normalization)

### 3.2.2 模型架构 | Model Architecture
- 自定义ResNet30结构 | Custom ResNet30 architecture
- 4个残差模块（3/4/6/3层设计） | 4 residual blocks (3/4/6/3-layer design)
- 自适应全局池化层 | Adaptive Global Pooling Layer

### 3.2.3 训练流程 | Training Pipeline
- 带类别权重的交叉熵损失 | Class-weighted cross entropy loss
- Adam优化器 + StepLR学习率调度 | Adam optimizer + StepLR scheduler
- 30个训练周期的进度条显示 | 30-epoch training with progress bar

## 3.3 评估指标 | Evaluation Metrics
- 常规指标：准确率、精确率、召回率 | Basic Metrics: Accuracy, Precision, Recall
- 特殊指标 | Advanced Metrics:
  - 类别特异性（Specificity） | Class-specific Specificity
  - F-measure（F1分数） | F-measure (F1 Score)
  - G-mean（几何平均数） | Geometric Mean (G-mean)

## 3.4 使用说明 | Usage Guide

### 3.4.1 环境要求 | Requirements
- Python 3.6+
- 关键依赖：PyTorch 1.8+、scikit-learn、matplotlib | Key dependencies: PyTorch 1.8+, scikit-learn, matplotlib

### 3.4.2 数据准备 | Data Preparation
- 目录结构 | Directory structure:
  ```text
  ecgdataset/
  ├── ARR/
  ├── CHF/
  └── NSR/
- 图像格式：PNG/JPG格式的ECG信号图像 | Image format: PNG/JPG ECG signal images

### 3.4.3 运行方式 | Execution
- 可调参数：`BATCH_SIZE`、`num_epochs`、`class_weights` | Tunable parameters: `BATCH_SIZE`, `num_epochs`, `class_weights`

## 3.5 输出结果 | Output Results

### 3.5.1 训练过程 | Training Process
- 实时显示各epoch损失值 | Real-time epoch loss display
- 验证集多指标输出（每个epoch） | Multi-metric validation output (per epoch)

### 3.5.2 最终测试 | Final Evaluation
- 混淆矩阵输出 | Confusion matrix generation
- 四维评估指标表格 | Four-dimensional metric table

### 3.5.3 自动生成训练曲线 | Training Curves
- 损失曲线 | Loss curve
- 各类别准确率曲线 | Class-specific accuracy curves
- F-measure/G-mean趋势图 | F-measure/G-mean trends
