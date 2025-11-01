# 知识图谱构建模块
import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_DATA_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('music_recommendation')


class KnowledgeGraphBuilder:
    """
    知识图谱构建器，使用内存数据结构实现知识图谱功能
    """
    
    def __init__(self):
        """
        初始化内存知识图谱
        """
        logger.info("初始化内存知识图谱")
        self.mock_nodes = {}
        self.mock_relationships = []
        # 添加索引以加速查询
        self.node_index = defaultdict(dict)  # {type: {id/name: node}}
        self.relationship_index = defaultdict(list)  # {(start_type, start_id, rel_type): [relationships]}
        
        # 加载持久化的知识图谱（如果存在）
        graph_dir = os.path.join(PROCESSED_DATA_DIR, 'knowledge_graph')
        if os.path.exists(graph_dir) and os.path.exists(os.path.join(graph_dir, 'nodes.json')):
            try:
                self.load_from_disk(graph_dir)
                logger.info("从磁盘加载知识图谱完成")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"加载知识图谱失败: {str(e)}，将创建新的知识图谱")
                # 确保数据结构清空
                self.clear_graph()
        else:
            logger.info("创建新的知识图谱")
    
    def get_node_by_id(self, node_type, node_id):
        """
        根据节点类型和ID获取节点
        
        Args:
            node_type: 节点类型
            node_id: 节点ID
            
        Returns:
            节点对象，如果不存在返回None
        """
        if node_type in self.node_index and node_id in self.node_index[node_type]:
            return self.node_index[node_type][node_id]
        return None
    
    def query_related_nodes(self, start_node_type, start_node_id, relationship_type):
        """
        查询与指定节点有特定关系的相关节点
        
        Args:
            start_node_type: 起始节点类型
            start_node_id: 起始节点ID
            relationship_type: 关系类型
            
        Returns:
            相关节点列表
        """
        key = (start_node_type, start_node_id, relationship_type)
        relationships = self.relationship_index.get(key, [])
        
        # 获取目标节点
        related_nodes = []
        for rel in relationships:
            target_type = rel['to']['type']
            target_id = rel['to'].get('id') or rel['to'].get('name')
            target_node = self.get_node_by_id(target_type, target_id)
            if target_node:
                related_nodes.append(target_node)
        
        return related_nodes
    
    def close(self):
        """
        关闭知识图谱，保存到磁盘
        """
        # 保存知识图谱到磁盘
        self.save_to_disk(os.path.join(PROCESSED_DATA_DIR, 'knowledge_graph'))
        logger.info("知识图谱已关闭并保存")
    
    def clear_graph(self):
        """
        清空现有图谱
        """
        self.mock_nodes = {}
        self.mock_relationships = []
        self.node_index = defaultdict(dict)
        self.relationship_index = defaultdict(list)
        logger.info("知识图谱已清空")
    
    def _convert_numpy_types(self, data):
        """
        递归地将numpy类型转换为Python原生类型
        
        Args:
            data: 需要转换的数据
        
        Returns:
            转换后的数据
        """
        if isinstance(data, dict):
            return {key: self._convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def save_to_disk(self, output_dir):
        """
        将知识图谱保存到磁盘
        
        Args:
            output_dir: 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换numpy类型为Python原生类型
        converted_nodes = self._convert_numpy_types(self.mock_nodes)
        converted_relationships = self._convert_numpy_types(self.mock_relationships)
        
        # 保存节点
        with open(os.path.join(output_dir, 'nodes.json'), 'w', encoding='utf-8') as f:
            json.dump(converted_nodes, f, ensure_ascii=False, indent=2)
        
        # 保存关系
        with open(os.path.join(output_dir, 'relationships.json'), 'w', encoding='utf-8') as f:
            json.dump(converted_relationships, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识图谱已保存到 {output_dir}")
    
    def load_from_disk(self, input_dir):
        """
        从磁盘加载知识图谱
        
        Args:
            input_dir: 输入目录路径
        
        Raises:
            json.JSONDecodeError: JSON文件格式错误
            FileNotFoundError: 文件不存在
        """
        # 检查两个必要文件是否都存在
        nodes_file = os.path.join(input_dir, 'nodes.json')
        relationships_file = os.path.join(input_dir, 'relationships.json')
        
        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"节点文件不存在: {nodes_file}")
        if not os.path.exists(relationships_file):
            raise FileNotFoundError(f"关系文件不存在: {relationships_file}")
        
        try:
            # 加载节点
            with open(nodes_file, 'r', encoding='utf-8') as f:
                self.mock_nodes = json.load(f)
            
            # 加载关系
            with open(relationships_file, 'r', encoding='utf-8') as f:
                self.mock_relationships = json.load(f)
            
            # 重建索引
            self._rebuild_indexes()
            logger.info(f"知识图谱已从 {input_dir} 加载")
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {str(e)}")
            raise
    
    def _rebuild_indexes(self):
        """
        重建节点和关系索引
        """
        self.node_index = defaultdict(dict)
        self.relationship_index = defaultdict(list)
        
        # 重建节点索引
        for node_type, nodes in self.mock_nodes.items():
            for node in nodes:
                if 'id' in node:
                    self.node_index[node_type][node['id']] = node
                if 'name' in node:
                    self.node_index[node_type][node['name']] = node
        
        # 重建关系索引
        for rel in self.mock_relationships:
            key = (rel['from']['type'], rel['from'].get('id') or rel['from'].get('name'), rel['type'])
            self.relationship_index[key].append(rel)
    
    def create_artists(self, artists_df):
        """
        创建艺术家节点
        """
        if 'Artist' not in self.mock_nodes:
            self.mock_nodes['Artist'] = []
            
        for _, row in artists_df.iterrows():
            artist_node = {
                'id': row['artist_id'],
                'name': row['artist_name']
            }
            self.mock_nodes['Artist'].append(artist_node)
            # 更新索引
            self.node_index['Artist'][row['artist_id']] = artist_node
            self.node_index['Artist'][row['artist_name']] = artist_node
            
        logger.info(f"已创建 {len(artists_df)} 个艺术家节点")
    
    def create_tracks(self, tracks_df):
        """
        创建曲目节点
        """
        if 'Track' not in self.mock_nodes:
            self.mock_nodes['Track'] = []
            
        for _, row in tracks_df.iterrows():
            track_node = {
                'id': row['track_id'],
                'name': row['track_name'],
                'duration': row['duration'],
                'release_year': row['release_year'],
                'popularity': row['popularity']
            }
            self.mock_nodes['Track'].append(track_node)
            # 更新索引
            self.node_index['Track'][row['track_id']] = track_node
            self.node_index['Track'][row['track_name']] = track_node
            
        logger.info(f"已创建 {len(tracks_df)} 个曲目节点")
    
    def create_albums(self, albums_df):
        """
        创建专辑节点
        """
        if 'Album' not in self.mock_nodes:
            self.mock_nodes['Album'] = []
        
        for _, row in albums_df.iterrows():
            album_node = {
                'id': row['album_id'],
                'name': row['album_name'],
                'release_date': row['release_date']
            }
            self.mock_nodes['Album'].append(album_node)
            # 更新索引
            self.node_index['Album'][row['album_id']] = album_node
            self.node_index['Album'][row['album_name']] = album_node
            
        logger.info(f"已创建 {len(albums_df)} 个专辑节点")
    
    def create_artist_track_relationships(self, tracks_df):
        """
        创建艺术家-创作->曲目的关系
        """
        count = 0
        for _, row in tracks_df.iterrows():
            relationship = {
                'type': 'CREATED',
                'from': {'type': 'Artist', 'id': row['artist_id']},
                'to': {'type': 'Track', 'id': row['track_id']}
            }
            self.mock_relationships.append(relationship)
            # 更新索引
            key = ('Artist', row['artist_id'], 'CREATED')
            self.relationship_index[key].append(relationship)
            count += 1
        
        logger.info(f"已创建 {count} 个艺术家-曲目关系")
    
    def create_album_track_relationships(self, album_tracks_df):
        """
        创建专辑-包含->曲目的关系
        """
        count = 0
        for _, row in album_tracks_df.iterrows():
            relationship = {
                'type': 'CONTAINS',
                'from': {'type': 'Album', 'id': row['album_id']},
                'to': {'type': 'Track', 'id': row['track_id']},
                'properties': {'track_number': row['track_number']}
            }
            self.mock_relationships.append(relationship)
            # 更新索引
            key = ('Album', row['album_id'], 'CONTAINS')
            self.relationship_index[key].append(relationship)
            count += 1
        
        logger.info(f"已创建 {count} 个专辑-曲目关系")
    
    def create_artist_album_relationships(self, albums_df):
        """
        创建艺术家-发行->专辑的关系
        """
        count = 0
        for _, row in albums_df.iterrows():
            relationship = {
                'type': 'RELEASED',
                'from': {'type': 'Artist', 'id': row['artist_id']},
                'to': {'type': 'Album', 'id': row['album_id']}
            }
            self.mock_relationships.append(relationship)
            # 更新索引
            key = ('Artist', row['artist_id'], 'RELEASED')
            self.relationship_index[key].append(relationship)
            count += 1
        
        logger.info(f"已创建 {count} 个艺术家-专辑关系")
    
    def create_genre_relationships(self, artists_df):
        """
        创建艺术家-风格->流派的关系
        """
        # 模拟创建流派节点和关系
        if 'Genre' not in self.mock_nodes:
            self.mock_nodes['Genre'] = []
        
        all_genres = set()
        for _, row in artists_df.iterrows():
            genres_str = row['genres'].strip('[]')
            genres = [g.strip('"') for g in genres_str.split(',') if g.strip()]
            all_genres.update(genres)
        
        # 创建流派节点
        for genre_name in all_genres:
            genre_node = {'name': genre_name}
            self.mock_nodes['Genre'].append(genre_node)
            # 更新索引
            self.node_index['Genre'][genre_name] = genre_node
        
        # 创建艺术家-流派关系
        rel_count = 0
        for _, row in artists_df.iterrows():
            artist_id = row['artist_id']
            genres_str = row['genres'].strip('[]')
            genres = [g.strip('"') for g in genres_str.split(',') if g.strip()]
            
            for genre in genres:
                relationship = {
                    'type': 'HAS_GENRE',
                    'from': {'type': 'Artist', 'id': artist_id},
                    'to': {'type': 'Genre', 'name': genre}
                }
                self.mock_relationships.append(relationship)
                # 更新索引
                key = ('Artist', artist_id, 'HAS_GENRE')
                self.relationship_index[key].append(relationship)
                rel_count += 1
        
        logger.info(f"已创建 {len(all_genres)} 个流派节点")
        logger.info(f"已创建 {rel_count} 个艺术家-流派关系")
    
    def build_graph(self, clear_existing=True):
        """
        构建完整的知识图谱
        """
        # 读取处理后的数据
        artists_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'artists.csv'))
        tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'tracks.csv'))
        albums_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'albums.csv'))
        album_tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'album_tracks.csv'))
        
        # 清空现有图谱
        if clear_existing:
            self.clear_graph()
        
        logger.info("开始构建知识图谱...")
        
        # 创建节点
        self.create_artists(artists_df)
        self.create_albums(albums_df)
        self.create_tracks(tracks_df)
        
        # 创建关系
        self.create_artist_track_relationships(tracks_df)
        self.create_album_track_relationships(album_tracks_df)
        self.create_artist_album_relationships(albums_df)
        self.create_genre_relationships(artists_df)
        
        # 统计节点和关系数量
        node_count = sum(len(nodes) for nodes in self.mock_nodes.values())
        rel_count = len(self.mock_relationships)
        
        logger.info(f"知识图谱构建完成！")
        logger.info(f"节点数量: {node_count}")
        logger.info(f"关系数量: {rel_count}")
        
        # 保存构建好的图谱
        self.save_to_disk(os.path.join(PROCESSED_DATA_DIR, 'knowledge_graph'))
        
        return True


def main():
    """
    主函数：构建知识图谱
    """
    # 初始化知识图谱构建器
    kg_builder = KnowledgeGraphBuilder()
    
    try:
        # 构建知识图谱
        kg_builder.build_graph(clear_existing=True)
        
        # 查询示例
        # 获取一个艺术家节点
        artist = kg_builder.get_node_by_id('Artist', '2CIMQHirSU0MQqyYHq0eOx')
        print(f"艺术家: {artist}")
        
        # 查询相关节点
        related_tracks = kg_builder.query_related_nodes('Artist', '2CIMQHirSU0MQqyYHq0eOx', 'CREATED')
        print(f"该艺术家创建的曲目数量: {len(related_tracks)}")
        
    except Exception as e:
        logger.error(f"构建知识图谱时出错: {str(e)}")
        raise
    finally:
        # 关闭连接
        kg_builder.close()


if __name__ == "__main__":
    main()