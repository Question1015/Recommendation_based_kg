# 知识驱动的音乐推荐系统

## 项目概述

本项目实现了一个基于知识图谱的音乐推荐系统，利用 TransE 模型学习实体和关系嵌入，结合用户画像生成个性化的音乐推荐。系统包含完整的数据处理、知识图谱构建、模型训练和 Web 交互界面。

## 项目结构

```
rec_based_kg/
├── README_project.md    # 项目详细说明文档
├── config.py            # 配置文件
├── requirements.txt     # 依赖列表
├── run.py               # 启动脚本
└── src/                 # 源代码目录
    ├── app.py           # Flask主应用
    ├── data_processing/ # 数据处理模块
    ├── models/          # 模型定义
    ├── recommendation/  # 推荐系统核心
    ├── templates/       # HTML模板
    └── utils/           # 工具函数
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- Flask 2.0+

> **注意**：使用内存中的知识图谱实现

## 安装步骤

1. **创建虚拟环境**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
. venv/bin/activate
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动应用

运行以下命令启动完整应用（数据加载、知识图谱构建、模型训练和 Web 界面）：

```bash
python run.py
```

### 访问 Web 界面

启动后，访问 http://localhost:5000 可使用 Web 界面：

- 在"选择您喜欢的音乐风格"模块中，可选择 10 种标准流派（Genre 1 至 Genre 10）
- 在"添加您常听的歌曲"模块中，可搜索并添加最多 5 首已听歌曲
- 点击"生成推荐"按钮获取个性化推荐结果

## 核心功能

### 数据处理

- 自动模拟生成音乐数据集（艺术家、专辑、曲目、流派、用户收听记录）
- 生成的流派格式为"Genre X"（X 为 1-10 的数字）

### 知识图谱

- 构建内存中的知识图谱，包含艺术家、专辑、曲目、流派等实体
- 定义 CREATED、CONTAINS、HAS_GENRE 等关系

### 推荐系统

- 基于用户选择的流派偏好构建用户画像
- 支持通过已听歌曲进一步优化推荐
- 提供推荐解释功能，说明推荐原因

## API 接口

- **获取推荐**：`POST /api/recommend`

  - 参数：`genres`（流派偏好字典）、`listened_tracks`（已听曲目 ID 列表）、`top_k`（推荐数量）

- **获取音乐流派**：`GET /api/genres`

  - 返回：标准格式的流派列表 `["Genre 1", "Genre 2", ..., "Genre 10"]`

- **搜索曲目**：`GET /api/tracks/search`

  - 参数：`q`（搜索关键词）、`limit`（结果数量限制）

- **健康检查**：`GET /health`

## 注意事项

1. 首次运行时会自动生成模拟数据
2. 推荐引擎会在应用启动时自动初始化
3. 应用默认运行在端口 5000
# Recommendation_based_kg
