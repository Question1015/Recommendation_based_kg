import json
import logging

logger = logging.getLogger(__name__)

def load_json(file_path, encoding='utf-8'):
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        encoding: 文件编码，默认为utf-8
    
    Returns:
        加载的数据
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON文件失败: {file_path}, 错误: {e}")
        raise

def save_json(data, file_path, encoding='utf-8', indent=2):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: JSON文件路径
        encoding: 文件编码，默认为utf-8
        indent: 缩进空格数，默认为2
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"成功保存数据到: {file_path}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
        raise