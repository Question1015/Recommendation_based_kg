# 项目配置文件
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 文件路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# 知识图谱配置
KNOWLEDGE_GRAPH_PATH = os.path.join(PROCESSED_DATA_DIR, 'knowledge_graph')

# 模型配置
EMBEDDING_DIM = 128
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
EPOCHS = 100
MARGIN = 1.0

# Flask配置
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
SECRET_KEY = "112233"

# 用户画像权重配置
GENRE_WEIGHT = 0.4
TRACK_WEIGHT = 0.6

# 推荐配置
TOP_K = 10
SIMILARITY_THRESHOLD = 0.5

# 创建必要的目录
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)