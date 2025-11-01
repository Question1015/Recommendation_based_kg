# 推荐系统核心模块
import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    MODELS_DIR, 
    PROCESSED_DATA_DIR, GENRE_WEIGHT, TRACK_WEIGHT, TOP_K, SIMILARITY_THRESHOLD
)
from src.data_processing.kg_builder import KnowledgeGraphBuilder
from src.utils.utils import load_json, save_json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRecommendationEngine:
    """
    推荐系统引擎，负责用户画像构建、推荐生成和解释生成
    """
    
    def __init__(self):
        """
        初始化推荐引擎
        """
        # 初始化知识图谱构建器（使用内存中的知识图谱）
        self.kg_builder = KnowledgeGraphBuilder()
        # 尝试从磁盘加载知识图谱
        try:
            self.kg_builder.load_from_disk(os.path.join(PROCESSED_DATA_DIR, 'knowledge_graph'))
            logger.info("成功加载内存知识图谱")
        except Exception as e:
            logger.warning(f"加载知识图谱失败: {e}，将使用空的知识图谱")
            
        # 加载嵌入向量
        self.entity_embeddings = np.load(os.path.join(MODELS_DIR, 'entity_embeddings.npy'))
        self.relation_embeddings = np.load(os.path.join(MODELS_DIR, 'relation_embeddings.npy'))
        
        # 加载实体和关系映射
        with open(os.path.join(PROCESSED_DATA_DIR, 'entity_to_id.json'), 'r', encoding='utf-8') as f:
            self.entity_to_id = json.load(f)
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        
        with open(os.path.join(PROCESSED_DATA_DIR, 'relation_to_id.json'), 'r', encoding='utf-8') as f:
            self.relation_to_id = json.load(f)
        
        # 预加载曲目信息
        self.track_info = self._load_track_info()
    
    def close(self):
        """
        清理资源
        """
        # 知识图谱不需要特殊的关闭操作
        pass
    
    def _load_track_info(self):
        """
        加载曲目的基本信息
        """
        track_info_file = os.path.join(PROCESSED_DATA_DIR, 'track_info.json')
        
        # 优先从文件加载
        if os.path.exists(track_info_file):
            logger.info("从文件加载轨道信息")
            return load_json(track_info_file)
        
        # 如果文件不存在，则从CSV文件加载
        logger.info("从CSV文件加载轨道信息")
        tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'tracks.csv'))
        track_info = {}
        
        for _, row in tracks_df.iterrows():
            track_id = f'track_{row["track_id"]}'
            track_info[track_id] = {
                'track_id': row['track_id'],
                'title': row['track_name'],
                'artist_id': row['artist_id'],
                'duration': row['duration'],
                'release_year': row['release_year'],
                'popularity': row['popularity']
            }
        
        # 补充艺术家名称信息
        artists_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'artists.csv'))
        artist_map = {row['artist_id']: row['artist_name'] for _, row in artists_df.iterrows()}
        
        for track_id, info in track_info.items():
            info['artist'] = artist_map.get(info['artist_id'], 'Unknown Artist')
        
        # 保存到文件以便下次加载
        save_json(track_info, track_info_file)
        
        return track_info
    
    def _contains_chinese(self, text):
        """
        检查文本是否包含中文字符
        """
        import re
        return bool(re.search(r'[\u4e00-\u9fa5]', text))
    
    def build_user_profile(self, user_id, genres, listened_tracks):
        """
        构建用户画像
        
        参数：
        - user_id: 用户ID
        - genres: 字典，键为流派名称，值为偏好程度（0-1）
        - listened_tracks: 列表，包含用户听过的曲目ID
        
        返回：
        - user_vector: 用户向量表示
        """
        # 初始化用户向量
        user_vector = np.zeros_like(self.entity_embeddings[0])
        
        # 流派向量部分
        genre_vectors = []
        genre_weights = []
        
        # 处理用户选择的流派 - 针对data_loader中生成的"Genre X"格式
        for genre_name, original_weight in genres.items():
            # 方法1: 直接匹配流派名称
            if genre_name in self.entity_to_id:
                genre_idx = self.entity_to_id[genre_name]
                genre_vector = self.entity_embeddings[genre_idx]
                genre_vectors.append(genre_vector)
                genre_weights.append(original_weight)
            else:
                # 方法2: 遍历所有实体，寻找与用户选择流派相关的实体
                found = False
                for entity_name, entity_idx in self.entity_to_id.items():
                    # 检查是否有直接匹配或包含关系
                    if (genre_name == entity_name or 
                        genre_name in entity_name or 
                        entity_name in genre_name):
                        genre_vector = self.entity_embeddings[entity_idx]
                        genre_vectors.append(genre_vector)
                        genre_weights.append(original_weight)
                        found = True
                        break
                
                # 方法3: 特别处理Genre格式，尝试在实体中查找包含数字的流派
                if not found and genre_name.startswith('Genre '):
                    # 提取数字部分
                    genre_number = genre_name.split(' ')[1] if len(genre_name.split(' ')) > 1 else ''
                    if genre_number.isdigit():
                        for entity_name, entity_idx in self.entity_to_id.items():
                            if genre_number in entity_name and 'Genre' in entity_name:
                                genre_vector = self.entity_embeddings[entity_idx]
                                genre_vectors.append(genre_vector)
                                genre_weights.append(original_weight)
                                found = True
                                break
        
        if genre_vectors:
            genre_vectors = np.array(genre_vectors)
            genre_weights = np.array(genre_weights) / sum(genre_weights)  # 归一化权重
            weighted_genre_vector = np.sum(genre_vectors * genre_weights[:, np.newaxis], axis=0)
        else:
            weighted_genre_vector = np.zeros_like(user_vector)
        
        # 已听曲目向量部分
        track_vectors = []
        
        for track_id in listened_tracks:
            entity_name = f'track_{track_id}'
            if entity_name in self.entity_to_id:
                track_idx = self.entity_to_id[entity_name]
                track_vector = self.entity_embeddings[track_idx]
                track_vectors.append(track_vector)
        
        if track_vectors:
            track_vectors = np.array(track_vectors)
            avg_track_vector = np.mean(track_vectors, axis=0)
        else:
            avg_track_vector = np.zeros_like(user_vector)
        
        # 组合用户向量
        # 使用向量维度和长度来判断是否有效
        genre_vectors_valid = len(genre_vectors) > 0
        track_vectors_valid = len(track_vectors) > 0
        
        if genre_vectors_valid and track_vectors_valid:
            user_vector = GENRE_WEIGHT * weighted_genre_vector + TRACK_WEIGHT * avg_track_vector
        elif genre_vectors_valid:
            user_vector = weighted_genre_vector
        elif track_vectors_valid:
            user_vector = avg_track_vector
        
        # 归一化用户向量
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        
        return user_vector
    
    def recommend(self, user_vector, excluded_tracks=None, top_k=TOP_K):
        """
        生成音乐推荐
        
        参数：
        - user_vector: 用户向量
        - excluded_tracks: 要排除的曲目ID列表（如已听过的曲目）
        - top_k: 返回的推荐数量
        
        返回：
        - recommendations: 推荐结果列表
        """
        if excluded_tracks is None:
            excluded_tracks = []
        
        # 获取所有曲目的嵌入向量
        track_entity_ids = [e for e in self.entity_to_id if e.startswith('track_')]
        track_indices = [self.entity_to_id[e] for e in track_entity_ids]
        track_embeddings = self.entity_embeddings[track_indices]
        
        # 计算用户与所有曲目的相似度
        similarities = cosine_similarity([user_vector], track_embeddings)[0]
        
        # 创建曲目ID和相似度的映射
        track_similarities = list(zip(track_entity_ids, similarities))
        
        # 过滤已排除的曲目
        excluded_entity_ids = [f'track_{tid}' for tid in excluded_tracks]
        track_similarities = [(tid, sim) for tid, sim in track_similarities 
                             if tid not in excluded_entity_ids]
        
        # 根据相似度排序并选择top_k
        track_similarities.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = track_similarities[:top_k]
        
        # 构建推荐结果
        recommendations = []
        for track_entity_id, score in top_recommendations:
            if score >= SIMILARITY_THRESHOLD:
                track_id = track_entity_id.split('_')[-1]
                # 从知识图谱中获取详细信息
                track_info = self._get_track_details(track_id)
                recommendations.append({
                    'track_id': track_id,
                    'title': track_info.get('title', 'Unknown Title'),
                    'artist': track_info.get('artist', 'Unknown Artist'),
                    'album': track_info.get('album', 'Unknown Album'),
                    'score': float(score)
                })
        
        return recommendations
        
    def _get_track_details(self, track_id):
        """
        从知识图谱中获取歌曲的详细信息
        """
        # 首先尝试从缓存的track_info中获取
        entity_name = f'track_{track_id}'
        if entity_name in self.track_info:
            return self.track_info[entity_name]
        
        # 如果缓存中没有，则从知识图谱中查询
        track_node = self.kg_builder.get_node_by_id('Track', track_id)
        if not track_node:
            return {}
        
        # 构建track_info
        track_info = {
            'track_id': track_id,
            'title': track_node.get('name', 'Unknown Title'),
            'artist': track_node.get('artist', 'Unknown Artist'),
            'album': track_node.get('album', 'Unknown Album')
        }
        
        # 查询艺术家信息
        key = ('Track', track_id, 'HAS_ARTIST')
        if key in self.kg_builder.relationship_index:
            artist_relationships = self.kg_builder.relationship_index[key]
            if artist_relationships:
                artist_id = artist_relationships[0]['to'].get('id')
                if artist_id:
                    artist_node = self.kg_builder.get_node_by_id('Artist', artist_id)
                    if artist_node:
                        track_info['artist'] = artist_node.get('name', 'Unknown Artist')
        
        # 保存到缓存
        self.track_info[entity_name] = track_info
        
        return track_info
    
    def generate_explanation(self, user_vector, recommended_track, genres, listened_tracks):
        """
        为推荐结果生成解释
        """
        explanations = []
        
        # 1. 基于用户历史偏好的解释
        if listened_tracks:
            # 获取最相似的已听歌曲
            most_similar = None
            max_similarity = 0
            
            for track_id in listened_tracks:
                # 计算与已听歌曲的相似度
                sim = self._calculate_track_similarity(recommended_track['track_id'], track_id)
                if sim > max_similarity:
                    max_similarity = sim
                    most_similar = track_id
            
            if most_similar and max_similarity > 0.7:  # 相似度阈值
                explanations.append({
                    'type': 'similar_to_heard',
                    'content': f"这首歌曲与您之前喜欢的《{self._get_track_name(most_similar)}》很相似",
                    'confidence': max_similarity
                })
        
        # 2. 基于艺术家的解释
        # 检查推荐歌曲的艺术家是否是用户喜欢的
        if recommended_track['artist'] in self._get_user_favorite_artists(listened_tracks):
            explanations.append({
                'type': 'favorite_artist',
                'content': f"由您喜欢的艺术家{recommended_track['artist']}创作",
                'confidence': 0.8
            })
        
        # 3. 基于流派的解释
        track_genres = self._get_track_genres(recommended_track['track_id'])
        
        # 进行流派匹配
        genre_matches = set()
        for track_genre in track_genres:
            # 直接匹配用户选择的流派
            for user_genre in genres.keys():
                # 精确匹配
                if track_genre == user_genre:
                    genre_matches.add(user_genre)
                    break
                
                # 包含匹配 - 检查流派名称中是否包含相同的数字部分
                if 'Genre' in track_genre and 'Genre' in user_genre:
                    # 提取数字部分
                    track_num = ''.join(filter(str.isdigit, track_genre))
                    user_num = ''.join(filter(str.isdigit, user_genre))
                    if track_num and user_num and track_num == user_num:
                        genre_matches.add(user_genre)
                        break
                
                # 其他包含关系
                if track_genre in user_genre or user_genre in track_genre:
                    genre_matches.add(user_genre)
                    break
        
        if genre_matches:
            explanations.append({
                'type': 'preferred_genre',
                'content': f"属于您喜欢的流派：{', '.join(genre_matches)}",
                'confidence': 0.7
            })
        
        # 按置信度排序
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 组合解释为文本格式
        if explanations:
            explanation_text = "推荐理由："
            explanation_text += "，且".join([e['content'] for e in explanations])
            return explanation_text
        else:
            return "推荐理由：基于您的音乐偏好推荐"
    
    def _get_track_name(self, track_id):
        """
        获取歌曲名称
        """
        entity_name = f'track_{track_id}'
        if entity_name in self.track_info:
            return self.track_info[entity_name].get('title', 'Unknown Title')
        
        # 从知识图谱中获取
        track_node = self.kg_builder.get_node_by_id('Track', track_id)
        if track_node:
            return track_node.get('name', 'Unknown Title')
        
        return 'Unknown Title'
    
    def _calculate_track_similarity(self, track1_id, track2_id):
        """
        计算两首歌曲之间的相似度
        """
        entity1 = f'track_{track1_id}'
        entity2 = f'track_{track2_id}'
        
        if entity1 in self.entity_to_id and entity2 in self.entity_to_id:
            idx1 = self.entity_to_id[entity1]
            idx2 = self.entity_to_id[entity2]
            return cosine_similarity([self.entity_embeddings[idx1]], [self.entity_embeddings[idx2]])[0][0]
        
        return 0
    
    def _get_user_favorite_artists(self, listened_tracks):
        """
        获取用户喜欢的艺术家列表
        """
        artists = set()
        
        for track_id in listened_tracks:
            track_info = self._get_track_details(track_id)
            if track_info:
                artist = track_info.get('artist', '')
                if artist:
                    artists.add(artist)
        
        return artists
    
    def _get_track_genres(self, track_id):
        """
        获取歌曲的流派信息
        """
        genres = []
        
        # 从内存知识图谱中查询歌曲的流派
        # 1. 找到与歌曲相关的艺术家
        key = ('Track', track_id, 'HAS_ARTIST')
        if key in self.kg_builder.relationship_index:
            artist_relationships = self.kg_builder.relationship_index[key]
            for rel in artist_relationships:
                artist_id = rel['to'].get('id')
                if artist_id:
                    # 2. 找到艺术家的流派
                    artist_genre_key = ('Artist', artist_id, 'HAS_GENRE')
                    if artist_genre_key in self.kg_builder.relationship_index:
                        genre_relationships = self.kg_builder.relationship_index[artist_genre_key]
                        for genre_rel in genre_relationships:
                            genre_name = genre_rel['to'].get('name')
                            if genre_name:
                                genres.append(genre_name)
        
        return genres
    
    def get_recommendations_with_explanations(self, genres, listened_tracks, top_k=TOP_K):
        """
        获取带解释的推荐结果
        
        参数：
        - genres: 用户的流派偏好
        - listened_tracks: 用户已听曲目
        - top_k: 返回的推荐数量
        
        返回：
        - recommendations: 带解释的推荐结果列表
        """
        # 使用默认用户ID构建用户画像，以兼容app.py的调用
        default_user_id = "current_user"
        
        # 构建用户画像
        user_vector = self.build_user_profile(default_user_id, genres, listened_tracks)
        
        # 生成推荐
        recommendations = self.recommend(user_vector, listened_tracks, top_k)
        
        # 为每个推荐生成解释
        for rec in recommendations:
            rec['explanation'] = self.generate_explanation(
                user_vector, rec, genres, listened_tracks
            )
        
        return recommendations


    def get_recommendations(self, user_id, genres, listened_tracks, top_n=TOP_K):
        """
        获取推荐结果
        
        参数：
        - user_id: 用户ID
        - genres: 用户的流派偏好
        - listened_tracks: 用户已听曲目
        - top_n: 返回的推荐数量
        
        返回：
        - recommendations: 推荐结果列表
        """
        # 构建用户画像
        user_vector = self.build_user_profile(user_id, genres, listened_tracks)
        
        # 生成推荐
        return self.recommend(user_vector, listened_tracks, top_n)

    def get_user_profile(self, user_id, user_history=None):
        """
        根据用户历史行为构建用户画像
        """
        if not user_history:
            # 如果没有提供用户历史，从知识图谱中获取
            user_history = self._get_user_history(user_id)
        
        if not user_history:
            return None
        
        # 构建用户兴趣向量
        user_embedding = np.zeros(self.entity_embeddings.shape[1])
        
        # 对用户喜欢的每首歌曲，计算其嵌入向量并加权平均
        for track_id, weight in user_history.items():
            entity_name = f'track_{track_id}'
            if entity_name in self.entity_to_id:
                track_embedding = self.entity_embeddings[self.entity_to_id[entity_name]]
                user_embedding += weight * track_embedding
        
        # 归一化
        if np.linalg.norm(user_embedding) > 0:
            user_embedding = user_embedding / np.linalg.norm(user_embedding)
        
        return user_embedding
    
    def _get_user_history(self, user_id):
        """
        从知识图谱中获取用户历史
        """
        user_history = {}
        
        # 从内存知识图谱中查询用户-歌曲关系
        key = ('User', user_id, 'LIKES')
        if key in self.kg_builder.relationship_index:
            relationships = self.kg_builder.relationship_index[key]
            for rel in relationships:
                track_id = rel['to'].get('id')
                if track_id:
                    # 默认权重为1.0，可以根据实际情况调整
                    user_history[track_id] = 1.0
        
        return user_history
    
    def get_similar_tracks(self, track_id, top_k=10):
        """
        获取与给定歌曲相似的歌曲
        """
        entity_name = f'track_{track_id}'
        if entity_name not in self.entity_to_id:
            return []
        
        track_embedding = self.entity_embeddings[self.entity_to_id[entity_name]]
        similarities = cosine_similarity([track_embedding], self.entity_embeddings)[0]
        
        # 获取最相似的top_k个实体
        similar_indices = np.argsort(similarities)[::-1][:top_k+1]  # +1 是因为包括自己
        
        similar_tracks = []
        for idx in similar_indices:
            entity_id = self.id_to_entity[idx]
            # 只返回歌曲类型的实体
            if entity_id in self.track_info:
                similar_tracks.append({
                    'id': self.track_info[entity_id]['track_id'],
                    'name': self.track_info[entity_id]['title'],
                    'artists': self.track_info[entity_id]['artist'],
                    'similarity': float(similarities[idx])
                })
        
        # 移除自己并只保留前top_k个
        return similar_tracks[1:top_k+1]

def main():
    """演示推荐系统的使用"""
    # 示例使用
    engine = MusicRecommendationEngine()
    print("推荐系统初始化完成")
    
    # 演示用户画像构建
    user_id = "user1"
    favorite_genres = {"rock": 0.8, "pop": 0.6}
    listened_tracks = ["track1", "track2", "track3"]
    
    user_profile = engine.build_user_profile(user_id, favorite_genres, listened_tracks)
    print(f"用户画像: {user_profile}")
    
    # 演示音乐推荐
    recommendations = engine.get_recommendations(user_id, favorite_genres, listened_tracks, top_n=5)
    print(f"推荐结果: {recommendations}")
    
    # 关闭连接
    engine.close()

if __name__ == "__main__":
    main()