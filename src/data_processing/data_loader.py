# 数据加载模块
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_lastfm_dataset():
    """
    下载并加载Last.fm数据集的模拟实现
    在实际使用时，需要替换为真实的数据下载和加载逻辑
    """
    print("开始加载Last.fm数据集...")
    
    # 创建示例数据（模拟从公开数据集下载）
    sample_data = {
        'artists': {
            'artist_id': range(1, 101),
            'artist_name': [f'Artist {i}' for i in range(1, 101)],
            'genres': [[f'Genre {np.random.randint(1, 11)}' for _ in range(np.random.randint(1, 4))]
                       for _ in range(100)]
        },
        'tracks': {
            'track_id': range(1, 501),
            'track_name': [f'Track {i}' for i in range(1, 501)],
            'artist_id': [np.random.randint(1, 101) for _ in range(500)],
            'duration': [np.random.randint(120, 360) for _ in range(500)],
            'release_year': [np.random.randint(2000, 2024) for _ in range(500)],
            'popularity': [np.random.randint(1, 100) for _ in range(500)]
        },
        'user_listens': {
            # 先生成user_id，然后根据其长度生成其他字段
            'user_id': [f'user_{i}' for i in range(1, 1001) for _ in range(np.random.randint(1, 10))]
        }
    }
    
    # 处理user_listens数据，确保所有字段长度一致
    user_ids = sample_data['user_listens']['user_id']
    num_records = len(user_ids)
    sample_data['user_listens']['track_id'] = [np.random.randint(1, 501) for _ in range(num_records)]
    sample_data['user_listens']['play_count'] = [np.random.randint(1, 100) for _ in range(num_records)]
    
    # 保存示例数据
    for data_name, data_dict in sample_data.items():
        df = pd.DataFrame(data_dict)
        output_path = os.path.join(PROCESSED_DATA_DIR, f'{data_name}.csv')
        df.to_csv(output_path, index=False)
        print(f"已保存 {data_name} 数据到 {output_path}")
        print(f"{data_name} 数据共 {len(df)} 条记录")
    
    return sample_data


def load_musicbrainz_subset():
    """
    加载MusicBrainz子集数据
    """
    print("开始加载MusicBrainz子集数据...")
    
    # 创建模拟的专辑数据
    albums_data = {
        'album_id': range(1, 201),
        'album_name': [f'Album {i}' for i in range(1, 201)],
        'artist_id': [np.random.randint(1, 101) for _ in range(200)],
        'release_date': [f'{np.random.randint(2000, 2024)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}' 
                        for _ in range(200)]
    }
    
    # 为每个专辑分配曲目
    album_tracks = []
    for album_id in range(1, 201):
        num_tracks = np.random.randint(5, 15)
        for track_num in range(1, num_tracks + 1):
            track_id = np.random.randint(1, 501)
            album_tracks.append({
                'album_id': album_id,
                'track_id': track_id,
                'track_number': track_num
            })
    
    # 保存专辑数据
    albums_df = pd.DataFrame(albums_data)
    albums_output = os.path.join(PROCESSED_DATA_DIR, 'albums.csv')
    albums_df.to_csv(albums_output, index=False)
    
    # 保存专辑-曲目关系数据
    album_tracks_df = pd.DataFrame(album_tracks)
    album_tracks_output = os.path.join(PROCESSED_DATA_DIR, 'album_tracks.csv')
    album_tracks_df.to_csv(album_tracks_output, index=False)
    
    print(f"已保存专辑数据到 {albums_output}")
    print(f"已保存专辑-曲目关系到 {album_tracks_output}")
    
    return albums_data, album_tracks


def extract_triplets():
    """
    从处理后的数据中提取用于TransE训练的三元组
    """
    print("开始提取三元组数据...")
    
    # 读取处理后的数据
    tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'tracks.csv'))
    artists_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'artists.csv'))
    albums_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'albums.csv'))
    album_tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'album_tracks.csv'))
    
    triplets = []
    
    # 艺术家-创作->曲目的三元组
    for _, row in tqdm(tracks_df.iterrows(), total=len(tracks_df), desc="处理曲目"):
        triplets.append((f'artist_{row["artist_id"]}', 'CREATED', f'track_{row["track_id"]}'))
    
    # 专辑-包含->曲目的三元组
    for _, row in tqdm(album_tracks_df.iterrows(), total=len(album_tracks_df), desc="处理专辑-曲目关系"):
        triplets.append((f'album_{row["album_id"]}', 'CONTAINS', f'track_{row["track_id"]}'))
    
    # 艺术家-发行->专辑的三元组
    for _, row in tqdm(albums_df.iterrows(), total=len(albums_df), desc="处理专辑"):
        triplets.append((f'artist_{row["artist_id"]}', 'RELEASED', f'album_{row["album_id"]}'))
    
    # 艺术家-风格->流派的三元组
    for _, row in tqdm(artists_df.iterrows(), total=len(artists_df), desc="处理艺术家流派"):
        # 解析流派列表字符串
        genres_str = row['genres'].strip('[]')
        genres = [g.strip('"') for g in genres_str.split(',') if g.strip()]
        for genre in genres:
            triplets.append((f'artist_{row["artist_id"]}', 'HAS_GENRE', genre))
    
    # 保存三元组数据
    triplets_df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
    output_path = os.path.join(PROCESSED_DATA_DIR, 'kg_triplets.csv')
    triplets_df.to_csv(output_path, index=False)
    
    print(f"已提取 {len(triplets)} 个三元组并保存到 {output_path}")
    
    # 创建实体和关系映射
    entities = sorted(list(set([t[0] for t in triplets] + [t[2] for t in triplets])))
    relations = sorted(list(set([t[1] for t in triplets])))
    
    entity_to_id = {ent: i for i, ent in enumerate(entities)}
    relation_to_id = {rel: i for i, rel in enumerate(relations)}
    
    # 保存映射
    with open(os.path.join(PROCESSED_DATA_DIR, 'entity_to_id.json'), 'w', encoding='utf-8') as f:
        json.dump(entity_to_id, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'relation_to_id.json'), 'w', encoding='utf-8') as f:
        json.dump(relation_to_id, f, ensure_ascii=False, indent=2)
    
    print(f"已创建实体映射（{len(entities)}个实体）和关系映射（{len(relations)}个关系）")
    
    return triplets, entity_to_id, relation_to_id


def main():
    """
    主函数：加载和处理数据
    """
    # 确保处理后的数据目录存在
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # 定义所有数据名
    all_data_names = [
        "艺术家 (artists)",
        "曲目 (tracks)",
        "专辑 (albums)",
        "专辑-曲目关系 (album_tracks)",
        "用户收听记录 (user_listens)",
        "知识图谱三元组 (kg_triplets)",
        "实体映射 (entity_to_id.json)",
        "关系映射 (relation_to_id.json)"
    ]
    
    print("=== 数据预处理阶段 ===")
    print("本项目将处理以下类型的数据：")
    for i, data_name in enumerate(all_data_names, 1):
        print(f"{i}. {data_name}")
    print()
    
    # 加载Last.fm数据集
    load_lastfm_dataset()
    
    # 加载MusicBrainz子集
    load_musicbrainz_subset()
    
    # 提取三元组
    extract_triplets()
    
    print("\n=== 数据预处理完成 ===")
    print("所有数据已成功处理并保存！")
    print("处理的数据类型汇总：")
    for i, data_name in enumerate(all_data_names, 1):
        print(f"{i}. {data_name}")


if __name__ == "__main__":
    main()