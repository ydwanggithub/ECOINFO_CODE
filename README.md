# STGPR+GeoShapley 时空高斯过程回归可解释性分析框架

## 🎯 项目概述

本项目实现了基于时空高斯过程回归（STGPR）与GeoShapley可解释性分析的植被健康分析框架，使用2000-2024年数据集进行建模和可解释性分析。

## 📊 数据集说明

### 📁 数据文件

```
data/
├── ALL_DATA_with_VHI_PCA_res5.csv  # H3分辨率5 (1.4MB)
├── ALL_DATA_with_VHI_PCA_res6.csv  # H3分辨率6 (8.0MB)
└── ALL_DATA_with_VHI_PCA_res7.csv  # H3分辨率7 (45.8MB)
```

### ⏰ 时间范围

- 2000-2024

### 🗺️ 空间覆盖

- **研究区域**: 赣州市全域
- **空间组织**: H3六边形网格系统
- **多分辨率**: res5/res6/res7三个精度级别

### 🔧 特征体系

**14个优化特征**：

1. **空间信息** (2个): `latitude`, `longitude`
2. **气候特征** (2个): `temperature`, `precipitation`
3. **人类活动** (4个): `nightlight`, `population_density`, `road_density`, `mining_density`
4. **地形特征** (2个): `elevation`, `slope`
5. **土地覆盖** (3个): `forest_area_percent`, `cropland_area_percent`, `impervious_area_percent`
6. **时间特征** (1个): `year`

**目标变量**: `VHI` (植被健康指数)

## 🚀 快速开始

### 📋 环境要求

```bash
# Python 3.8+
# 主要依赖
pip install pandas numpy scikit-learn torch gpytorch h3 matplotlib seaborn
```

### ⚡ 基本运行

```bash
# 运行完整STGPR+GeoShapley分析
python main.py

# 指定特定分辨率
python main.py --resolutions res5 res6

# 跳过验证询问，自动继续
python main.py --skip_validation

# 强制重新处理数据（通常不需要）
python main.py --force_reprocess
```

### 🧪 测试数据加载

```bash
# 测试数据加载功能
python test_data_loading.py
```

## 📖 核心功能

### 1. **数据加载**

- 直接读取2000-2024年完整数据集
- 自动识别和加载多分辨率数据
- 内置数据质量检查和验证

### 2. **STGPR建模**

- 时空高斯过程回归核心算法
- 自动超参数优化
- 多分辨率并行训练

### 3. **GeoShapley可解释性分析**

- SHAP值分析
- GeoShapley空间归因
- 偏依赖图(PDP)分析
- 特征重要性评估

### 4. **可视化输出**

- 时空分布图
- 特征重要性对比
- 模型性能评估图
- 空间聚类分析图

## 📂 项目结构

```
ST-Gassian-Process_Regression_github/
├── data/                           # 2000-2024年完整数据集
├── data_processing/                # 数据处理模块
│   └── preprocessing.py           # 数据加载和预处理
├── model_analysis/                 # 模型分析核心模块
│   ├── stgpr_model.py            # STGPR模型实现
│   ├── stgpr_geoshapley.py       # GeoShapley可解释性
│   └── feature_importance.py     # 特征重要性分析
├── visualization/                  # 可视化模块
│   ├── pdp_plots.py              # 偏依赖图
│   └── shap_distribution_plots.py # SHAP分布图
├── output/                        # 结果输出目录
├── main.py                        # 主程序入口
├── test_data_loading.py           # 数据加载测试
└── README.md                      # 项目说明
```

## 🎮 使用示例

### 基础分析

```python
from data_processing.preprocessing import load_complete_dataset

# 加载完整数据集
data = load_complete_dataset(resolutions=['res5'])
res5_data = data['res5']

print(f"数据形状: {res5_data.shape}")
print(f"时间范围: {res5_data['year'].min()}-{res5_data['year'].max()}")
print(f"空间网格: {res5_data['h3_index'].nunique()}个")
```

### 运行STGPR+GeoShapley分析

```python
# 完整分析流程
python main.py --resolutions res5 --skip_validation
```

## 📊 输出结果

### 模型文件

- `output/res5/stgpr_model_res5.pt` - 训练好的模型
- `output/res5/res5_model_results.pkl` - 模型评估结果

### 可解释性分析

- `output/res5/res5_geoshapley_data.pkl` - GeoShapley分析结果
- `output/feature_importance_comparison.png` - 特征重要性对比图

### 可视化图表

- `output/all_resolutions_pdp_grid.png` - 偏依赖图网格
- `output/geoshapley_spatial_top3.png` - 空间归因分析
- `output/temporal_feature_heatmap.png` - 时间特征热图

## 🔍 技术特点

### 高效数据处理

- ✅ 直接使用预处理完成的2000-2024年数据集
- ✅ 避免重复的时间外推计算
- ✅ 优化的内存使用和加载速度

### 先进建模方法

- ✅ 时空高斯过程回归(STGPR)
- ✅ 自动化超参数优化
- ✅ 多分辨率并行处理

### 全面可解释性

- ✅ SHAP值全局解释
- ✅ GeoShapley空间归因
- ✅ 偏依赖图特征关系
- ✅ 时空聚类分析

## 📚 论文引用

如果使用本代码，请引用相关论文：

```bibtex
@article{wang2024vegetation,
  title={Multi-scale Mountain Vegetation Health Analysis in Ganzhou of China via Interpretable Spatiotemporal Machine Learning},
  author={Yuandong Wang},
  email={wangyuandong@gnnu.edu.cn},
  journal={期刊名称},
  year={2025}
}
```

## 🤝 贡献指南

欢迎提交Issues和Pull Requests来改进项目！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
