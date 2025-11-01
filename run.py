#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识驱动的音乐推荐系统 - 启动脚本

这个脚本用于初始化项目环境并启动Web应用。
它会按照以下步骤执行：
1. 检查必要的目录是否存在
2. 加载数据（如果需要）
3. 构建知识图谱（如果需要）
4. 训练模型（如果需要）
5. 启动Flask Web应用
"""

import os
import sys
import argparse
import subprocess
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("music_recommendation")

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR


def check_directories():
    """
    检查并创建必要的目录
    """
    logger.info("检查必要的目录结构...")
    
    directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"创建目录: {directory}")
        else:
            logger.info(f"目录已存在: {directory}")


def run_data_loader():
    """
    运行数据加载脚本
    """
    logger.info("开始加载和处理数据...")
    
    data_loader_path = os.path.join("src", "data_processing", "data_loader.py")
    try:
        # 不捕获输出，允许显示数据处理过程
        subprocess.run(
            [sys.executable, data_loader_path],
            check=True
        )
        logger.info("数据加载完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"数据加载失败")
        return False


def run_kg_builder():
    """
    运行知识图谱构建脚本
    """
    logger.info("开始构建知识图谱...")
    
    kg_builder_path = os.path.join("src", "data_processing", "kg_builder.py")
    try:
        # 不捕获输出，允许显示构建过程
        subprocess.run(
            [sys.executable, kg_builder_path],
            check=True
        )
        logger.info("知识图谱构建完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"知识图谱构建失败")
        logger.error("请确保Neo4j数据库已启动并且连接配置正确")
        return False


def run_model_training():
    """
    运行模型训练脚本
    """
    logger.info("开始训练TransE模型...")
    
    model_training_path = os.path.join("src", "models", "transe.py")
    try:
        # 不捕获输出，让进度条能够正常显示
        subprocess.run(
            [sys.executable, model_training_path],
            check=True
        )
        logger.info("模型训练完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"模型训练失败")
        return False


def start_web_app():
    """
    启动Flask Web应用
    """
    logger.info("启动Web应用...")
    
    # 检查必要的文件是否存在
    required_files = [
        os.path.join(MODELS_DIR, "entity_embeddings.npy"),
        os.path.join(MODELS_DIR, "relation_embeddings.npy")
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"缺少必要的文件: {file}")
            logger.error("请先运行数据处理和模型训练步骤")
            return False
    
    # 启动Web应用
    app_path = os.path.join("src", "app.py")
    try:
        logger.info("正在启动Flask应用...")
        logger.info("请访问 http://localhost:5000 查看应用")
        logger.info("按 Ctrl+C 停止应用")
        
        # 使用subprocess运行应用，这样可以显示实时输出
        subprocess.run([sys.executable, app_path], check=True)
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止应用...")
    except subprocess.CalledProcessError as e:
        logger.error(f"应用启动失败: {e}")
        return False


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='知识驱动的音乐推荐系统启动脚本')
    parser.add_argument('--data', action='store_true', help='只运行数据加载')
    parser.add_argument('--kg', action='store_true', help='只运行知识图谱构建')
    parser.add_argument('--train', action='store_true', help='只运行模型训练')
    parser.add_argument('--app', action='store_true', help='只启动Web应用')
    parser.add_argument('--all', action='store_true', help='运行所有步骤（数据加载、图谱构建、模型训练、启动应用）')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认运行Web应用
    if not any([args.data, args.kg, args.train, args.app, args.all]):
        args.app = True
    
    # 检查目录
    check_directories()
    
    # 执行指定的步骤
    if args.data or args.all:
        if not run_data_loader():
            logger.error("数据加载失败，停止执行")
            return
    
    if args.kg or args.all:
        if not run_kg_builder():
            logger.error("知识图谱构建失败，停止执行")
            return
    
    if args.train or args.all:
        if not run_model_training():
            logger.error("模型训练失败，停止执行")
            return
    
    if args.app or args.all:
        start_web_app()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"发生严重错误: {e}")
        sys.exit(1)