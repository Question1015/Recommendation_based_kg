# TransE模型实现
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, EMBEDDING_DIM, BATCH_SIZE, LEARNING_RATE, EPOCHS, MARGIN

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class TransE(nn.Module):
    """
    TransE模型实现
    核心思想：h + r ≈ t，其中h是头实体，r是关系，t是尾实体
    """
    
    def __init__(self, entity_count, relation_count, embedding_dim):
        """
        初始化TransE模型
        
        参数：
        - entity_count: 实体数量
        - relation_count: 关系数量
        - embedding_dim: 嵌入维度
        """
        super(TransE, self).__init__()
        
        # 初始化实体嵌入
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim)
        # 初始化关系嵌入
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim)
        
        # 使用Xavier均匀分布初始化，提供更好的梯度流
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def forward(self, heads, relations, tails, corrupted_tails=None):
        """
        前向传播
        
        参数：
        - heads: 头实体索引
        - relations: 关系索引
        - tails: 尾实体索引
        - corrupted_tails: 被损坏的尾实体索引（可选）
        
        返回：
        - 正样本和负样本的距离
        """
        # 获取嵌入向量
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 计算正样本距离
        positive_dist = torch.norm(h + r - t, p=2, dim=1)
        
        # 如果提供了损坏的尾实体，计算负样本距离
        if corrupted_tails is not None:
            t_corr = self.entity_embeddings(corrupted_tails)
            negative_dist = torch.norm(h + r - t_corr, p=2, dim=1)
            return positive_dist, negative_dist
        
        return positive_dist
    
    def normalize_entity_embeddings(self):
        """
        归一化实体嵌入向量
        """
        with torch.no_grad():
            norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1, keepdim=True)
            self.entity_embeddings.weight.div_(norms)
    
    def get_entity_embedding(self, entity_idx):
        """
        获取指定实体的嵌入向量
        """
        return self.entity_embeddings(torch.tensor(entity_idx)).detach().numpy()
    
    def get_relation_embedding(self, relation_idx):
        """
        获取指定关系的嵌入向量
        """
        return self.relation_embeddings(torch.tensor(relation_idx)).detach().numpy()
    
    def save_weights(self, filepath):
        """
        保存模型权重到文件
        
        参数：
        - filepath: 保存权重的文件路径
        """
        torch.save({
            'entity_embeddings': self.entity_embeddings.state_dict(),
            'relation_embeddings': self.relation_embeddings.state_dict()
        }, filepath)
        print(f"模型权重已保存到 {filepath}")
    
    def load_weights(self, filepath):
        """
        从文件加载模型权重
        支持两种格式：
        1. 完整模型状态字典（直接使用model.load_state_dict保存的格式）
        2. 组件状态字典（使用save_weights保存的格式，包含entity_embeddings和relation_embeddings键）
        
        参数：
        - filepath: 加载权重的文件路径
        
        返回：
        - True: 加载成功
        - False: 加载失败
        """
        try:
            # 使用map_location确保在任何设备上都能正确加载
            checkpoint = torch.load(filepath, map_location=device)
            
            # 首先尝试判断是否为完整模型状态字典
            if isinstance(checkpoint, dict):
                # 检查是否包含组件状态字典的键
                if 'entity_embeddings' in checkpoint and 'relation_embeddings' in checkpoint:
                    # 格式1：组件状态字典
                    self.entity_embeddings.load_state_dict(checkpoint['entity_embeddings'])
                    self.relation_embeddings.load_state_dict(checkpoint['relation_embeddings'])
                    print(f"已从 {filepath} 加载组件状态字典格式的模型权重")
                else:
                    # 格式2：完整模型状态字典
                    # 尝试直接加载到当前模型
                    try:
                        self.load_state_dict(checkpoint)
                        print(f"已从 {filepath} 加载完整模型状态字典")
                    except RuntimeError:
                        # 如果直接加载失败，尝试作为组件状态字典处理
                        # 检查是否包含embedding层权重
                        if 'entity_embeddings.weight' in checkpoint:
                            # 创建entity_embeddings和relation_embeddings的状态字典
                            entity_emb_state = {'weight': checkpoint['entity_embeddings.weight']}
                            relation_emb_state = {'weight': checkpoint['relation_embeddings.weight']}
                            self.entity_embeddings.load_state_dict(entity_emb_state)
                            self.relation_embeddings.load_state_dict(relation_emb_state)
                            print(f"已从 {filepath} 加载分离的嵌入层权重")
                        else:
                            raise KeyError("无法识别的权重文件格式")
            else:
                raise TypeError("权重文件格式错误")
            
            return True
        except FileNotFoundError:
            print(f"错误：找不到预训练权重文件 {filepath}")
            return False
        except KeyError as e:
            print(f"错误：权重文件格式不正确，{e}")
            return False
        except RuntimeError as e:
            print(f"错误：权重维度不匹配或模型结构不同：{e}")
            return False
        except Exception as e:
            print(f"加载权重时发生未知错误：{e}")
            return False


def load_triplets_and_mappings():
    """
    加载三元组数据和实体/关系映射
    """
    # 加载三元组数据
    triplets_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'kg_triplets.csv'))
    
    # 加载实体和关系映射
    with open(os.path.join(PROCESSED_DATA_DIR, 'entity_to_id.json'), 'r', encoding='utf-8') as f:
        entity_to_id = json.load(f)
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'relation_to_id.json'), 'r', encoding='utf-8') as f:
        relation_to_id = json.load(f)
    
    # 将三元组转换为索引
    heads = [entity_to_id[h] for h in triplets_df['head']]
    relations = [relation_to_id[r] for r in triplets_df['relation']]
    tails = [entity_to_id[t] for t in triplets_df['tail']]
    
    return np.array(heads), np.array(relations), np.array(tails), entity_to_id, relation_to_id


def generate_corrupted_triplets(heads, relations, tails, entity_count, corrupt_strategy='tail'):
    """
    生成被损坏的三元组
    
    参数：
    - heads: 头实体索引
    - relations: 关系索引
    - tails: 尾实体索引
    - entity_count: 实体总数
    - corrupt_strategy: 损坏策略 ('tail', 'head', 'both')
    
    返回：
    - 损坏的头实体和尾实体
    """
    # 根据策略选择损坏方式
    if corrupt_strategy == 'tail':
        # 只损坏尾实体（原始方法）
        corrupted_tails = np.random.randint(0, entity_count, size=len(tails))
        # 确保损坏的尾实体与原始尾实体不同
        mask = corrupted_tails == tails
        while np.any(mask):
            corrupted_tails[mask] = np.random.randint(0, entity_count, size=mask.sum())
            mask = corrupted_tails == tails
        return None, corrupted_tails
    
    elif corrupt_strategy == 'head':
        # 只损坏头实体
        corrupted_heads = np.random.randint(0, entity_count, size=len(heads))
        # 确保损坏的头实体与原始头实体不同
        mask = corrupted_heads == heads
        while np.any(mask):
            corrupted_heads[mask] = np.random.randint(0, entity_count, size=mask.sum())
            mask = corrupted_heads == heads
        return corrupted_heads, None
    
    else:  # 'both'
        # 随机损坏头或尾实体
        corrupted_heads = np.copy(heads)
        corrupted_tails = np.copy(tails)
        
        # 随机选择50%损坏头实体，50%损坏尾实体
        mask = np.random.rand(len(heads)) < 0.5
        
        # 损坏头实体的部分
        corrupted_heads[mask] = np.random.randint(0, entity_count, size=mask.sum())
        # 确保损坏的头实体与原始头实体不同
        head_mask = (corrupted_heads == heads) & mask
        while np.any(head_mask):
            corrupted_heads[head_mask] = np.random.randint(0, entity_count, size=head_mask.sum())
            head_mask = (corrupted_heads == heads) & mask
        
        # 损坏尾实体的部分
        corrupted_tails[~mask] = np.random.randint(0, entity_count, size=(~mask).sum())
        # 确保损坏的尾实体与原始尾实体不同
        tail_mask = (corrupted_tails == tails) & (~mask)
        while np.any(tail_mask):
            corrupted_tails[tail_mask] = np.random.randint(0, entity_count, size=tail_mask.sum())
            tail_mask = (corrupted_tails == tails) & (~mask)
        
        return corrupted_heads, corrupted_tails


def train_model(pretrained_weights_path=None, do_train=True):
    """
    训练TransE模型 - 优化版本
    
    参数：
    - pretrained_weights_path: 预训练权重文件路径（可选）
    - do_train: 是否执行训练（默认为True）
    
    返回：
    - model: 训练好的模型
    - entity_embeddings: 实体嵌入向量
    - relation_embeddings: 关系嵌入向量
    - entity_to_id: 实体到ID的映射
    - relation_to_id: 关系到ID的映射
    - weights_loaded: 是否成功加载了预训练权重
    """
    # 加载数据
    heads, relations, tails, entity_to_id, relation_to_id = load_triplets_and_mappings()
    entity_count = len(entity_to_id)
    relation_count = len(relation_to_id)
    
    print(f"实体数量: {entity_count}")
    print(f"关系数量: {relation_count}")
    print(f"三元组数量: {len(heads)}")
    
    # 创建模型并移至合适的设备
    model = TransE(entity_count, relation_count, EMBEDDING_DIM).to(device)
    
    # 加载预训练权重（如果提供）
    weights_loaded = False
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"尝试加载预训练权重: {pretrained_weights_path}")
        # 使用增强版的load_weights方法，它可以处理多种权重文件格式
        weights_loaded = model.load_weights(pretrained_weights_path)
        
        if not weights_loaded:
            print("加载预训练权重失败，使用随机初始化权重开始训练")
    elif pretrained_weights_path and not os.path.exists(pretrained_weights_path):
        print(f"警告：指定的预训练权重文件不存在: {pretrained_weights_path}")
        print("使用随机初始化权重开始训练")
    else:
        print("未指定预训练权重文件，使用随机初始化权重开始训练")
    
    # 权重加载状态已在前面的代码中输出，此处无需重复
    
    # 如果不需要训练，直接返回模型和嵌入向量
    if not do_train:
        print("跳过训练过程，直接使用当前加载的模型")
        # 获取当前模型的嵌入向量
        entity_embeddings = model.entity_embeddings.weight.detach().numpy()
        relation_embeddings = model.relation_embeddings.weight.detach().numpy()
        # 保存当前嵌入向量
        np.save(os.path.join(MODELS_DIR, 'entity_embeddings.npy'), entity_embeddings)
        np.save(os.path.join(MODELS_DIR, 'relation_embeddings.npy'), relation_embeddings)
        print("嵌入向量已保存到文件")
        return model, entity_embeddings, relation_embeddings, entity_to_id, relation_to_id, weights_loaded


    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 添加学习率衰减器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 将数据转换为PyTorch张量并移至设备
    heads_tensor = torch.LongTensor(heads).to(device)
    relations_tensor = torch.LongTensor(relations).to(device)
    tails_tensor = torch.LongTensor(tails).to(device)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(heads_tensor, relations_tensor, tails_tensor)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # Windows系统设置为0
    )
    
    # 记录损失
    losses = []
    
    # 早停机制参数
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    # 训练循环 - 添加总进度条
    for epoch in tqdm(range(EPOCHS), desc="总体训练进度", position=0):
        epoch_loss = 0
        
        # 批量训练 - 调整内部进度条参数避免冲突
        for batch_heads, batch_relations, batch_tails in tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{EPOCHS}", 
            leave=False, 
            position=1
        ):
            # 生成损坏的三元组（使用both策略）
            corrupted_heads, corrupted_tails = generate_corrupted_triplets(
                batch_heads.cpu().numpy(), 
                batch_relations.cpu().numpy(), 
                batch_tails.cpu().numpy(), 
                entity_count,
                corrupt_strategy='both'  # 使用更全面的损坏策略
            )
            
            # 根据损坏策略处理输入
            if corrupted_tails is not None:
                corrupted_tails = torch.LongTensor(corrupted_tails).to(device)
            if corrupted_heads is not None:
                corrupted_heads = torch.LongTensor(corrupted_heads).to(device)
            
            # 初始化损失
            loss = 0
            
            # 如果损坏了尾实体
            if corrupted_tails is not None:
                positive_dist, negative_dist = model(
                    batch_heads, batch_relations, batch_tails, corrupted_tails
                )
                loss += torch.relu(MARGIN + positive_dist - negative_dist).mean()
            
            # 如果损坏了头实体
            if corrupted_heads is not None:
                # 计算h + r - t的距离
                pos_dist = model(batch_heads, batch_relations, batch_tails)
                # 计算h' + r - t的距离
                neg_dist = model(corrupted_heads, batch_relations, batch_tails)
                loss += torch.relu(MARGIN + pos_dist - neg_dist).mean()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 归一化实体嵌入
            model.normalize_entity_embeddings()
            
            # 累加损失
            epoch_loss += loss.item() * len(batch_heads)
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 学习率衰减
        scheduler.step(avg_loss)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(MODELS_DIR, 'trane_model_best.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发！在第{epoch+1}个epoch停止训练")
                break
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            model_path = os.path.join(MODELS_DIR, f'trane_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
            
            # 比较当前epoch的loss与最佳loss，如果更低则更新最佳模型
            if avg_loss < best_loss:
                print(f"当前epoch {epoch+1} 的loss ({avg_loss:.4f}) 低于当前最佳loss ({best_loss:.4f})")
                best_model_path = os.path.join(MODELS_DIR, 'trane_model_best.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"最佳模型已更新: {best_model_path}")
                # 更新最佳loss和计数器
                best_loss = avg_loss
                counter = 0
    
    # 训练结束后保存最终权重
    final_weights_path = os.path.join(MODELS_DIR, 'trane_model_final_weights.pth')
    model.save_weights(final_weights_path)
    print(f"最终模型权重已保存到 {final_weights_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    print(f"加载最佳模型: {best_model_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    loss_curve_path = os.path.join(MODELS_DIR, 'training_loss.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存到 {loss_curve_path}")
    
    # 保存最终的嵌入向量
    entity_embeddings = model.entity_embeddings.weight.detach().numpy()
    relation_embeddings = model.relation_embeddings.weight.detach().numpy()
    
    np.save(os.path.join(MODELS_DIR, 'entity_embeddings.npy'), entity_embeddings)
    np.save(os.path.join(MODELS_DIR, 'relation_embeddings.npy'), relation_embeddings)
    
    # 保存最终模型权重
    final_weights_path = os.path.join(MODELS_DIR, 'trane_model_final_weights.pth')
    model.save_weights(final_weights_path)
    
    print(f"实体嵌入已保存到 {os.path.join(MODELS_DIR, 'entity_embeddings.npy')}")
    print(f"关系嵌入已保存到 {os.path.join(MODELS_DIR, 'relation_embeddings.npy')}")
    
    return model, entity_embeddings, relation_embeddings, entity_to_id, relation_to_id, weights_loaded


def evaluate_embeddings(entity_embeddings, entity_to_id, top_k=10):
    """
    评估嵌入质量：计算相似实体
    """
    print(f"评估嵌入质量，计算Top-{top_k}相似实体...")
    
    # 选择一些代表性实体进行评估
    sample_entities = list(entity_to_id.items())[:20]  # 选择前20个实体
    
    # 使用PyTorch进行向量化计算，提高效率
    entity_embeddings_tensor = torch.FloatTensor(entity_embeddings)
    
    # 归一化嵌入向量以加速余弦相似度计算
    norms = torch.norm(entity_embeddings_tensor, dim=1, keepdim=True)
    normalized_embeddings = entity_embeddings_tensor / norms
    
    results = {}
    
    # 使用进度条显示评估进度
    for entity_name, entity_idx in tqdm(sample_entities, desc="评估实体相似度"):
        # 获取当前实体的嵌入向量
        entity_embedding = normalized_embeddings[entity_idx].unsqueeze(0)
        
        # 使用矩阵乘法计算余弦相似度
        similarities = torch.mm(entity_embedding, normalized_embeddings.t()).squeeze().numpy()
        
        # 找到最相似的实体（排除自己）
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己，取前top_k
        
        # 获取相似实体的名称，并将numpy float32转换为Python float
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        similar_entities = [(id_to_entity[idx], float(similarities[idx])) for idx in similar_indices]
        
        results[entity_name] = similar_entities
    
    # 保存评估结果
    with open(os.path.join(MODELS_DIR, 'similarity_evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"相似度评估结果已保存到 {os.path.join(MODELS_DIR, 'similarity_evaluation.json')}")
    
    # 打印部分结果用于展示
    print("\n相似度评估示例：")
    for entity_name, similar_entities in list(results.items())[:5]:
        print(f"\n{entity_name} 的相似实体：")
        for similar_entity, score in similar_entities:
            print(f"  - {similar_entity}: {score:.4f}")
    
    return results


def main():
    """
    主函数：训练TransE模型并评估
    支持命令行参数：
    --use-pretrained: 使用默认路径的预训练权重（优先使用final_weights，其次是best模型）
    --pretrained-path PATH: 使用指定路径的预训练权重
    --use-best-model: 使用最佳模型权重
    --use-epoch N: 使用指定epoch的模型权重
    """
    # 确保模型目录存在
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    pretrained_weights_path = None
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--use-pretrained':
            # 优先尝试使用final_weights
            final_weights_path = os.path.join(MODELS_DIR, 'trane_model_final_weights.pth')
            # 其次尝试使用best模型
            best_model_path = os.path.join(MODELS_DIR, 'trane_model_best.pth')
            
            if os.path.exists(final_weights_path):
                pretrained_weights_path = final_weights_path
                print(f"使用最终预训练权重: {pretrained_weights_path}")
            elif os.path.exists(best_model_path):
                pretrained_weights_path = best_model_path
                print(f"使用最佳模型权重: {pretrained_weights_path}")
            else:
                # 查找最新的epoch模型
                epoch_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('trane_model_epoch_') and f.endswith('.pth')]
                if epoch_files:
                    # 按epoch号排序，取最大的
                    epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                    pretrained_weights_path = os.path.join(MODELS_DIR, epoch_files[0])
                    print(f"使用最新epoch模型: {pretrained_weights_path}")
                else:
                    print("未找到任何预训练权重文件")
                    
        elif sys.argv[1] == '--pretrained-path' and len(sys.argv) > 2:
            pretrained_weights_path = sys.argv[2]
            print(f"使用指定预训练权重路径: {pretrained_weights_path}")
            
        elif sys.argv[1] == '--use-best-model':
            best_model_path = os.path.join(MODELS_DIR, 'trane_model_best.pth')
            if os.path.exists(best_model_path):
                pretrained_weights_path = best_model_path
                print(f"使用最佳模型权重: {pretrained_weights_path}")
            else:
                print("未找到最佳模型权重文件")
                
        elif sys.argv[1] == '--use-epoch' and len(sys.argv) > 2:
            try:
                epoch_num = int(sys.argv[2])
                epoch_model_path = os.path.join(MODELS_DIR, f'trane_model_epoch_{epoch_num}.pth')
                if os.path.exists(epoch_model_path):
                    pretrained_weights_path = epoch_model_path
                    print(f"使用epoch {epoch_num} 的模型权重: {pretrained_weights_path}")
                else:
                    print(f"未找到epoch {epoch_num} 的模型权重文件")
            except ValueError:
                print("--use-epoch 参数需要一个整数")
    else:
        # 如果没有指定命令行参数，自动尝试加载预训练权重
        print("未指定命令行参数，尝试自动加载预训练权重...")
        # 优先尝试使用final_weights
        final_weights_path = os.path.join(MODELS_DIR, 'trane_model_final_weights.pth')
        # 其次尝试使用best模型
        best_model_path = os.path.join(MODELS_DIR, 'trane_model_best.pth')
        
        if os.path.exists(final_weights_path):
            pretrained_weights_path = final_weights_path
            print(f"自动加载最终预训练权重: {pretrained_weights_path}")
        elif os.path.exists(best_model_path):
            pretrained_weights_path = best_model_path
            print(f"自动加载最佳模型权重: {pretrained_weights_path}")
        else:
            # 查找最新的epoch模型
            epoch_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('trane_model_epoch_') and f.endswith('.pth')]
            if epoch_files:
                # 按epoch号排序，取最大的
                epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                pretrained_weights_path = os.path.join(MODELS_DIR, epoch_files[0])
                print(f"自动加载最新epoch模型: {pretrained_weights_path}")
            else:
                print("未找到任何预训练权重文件，将使用随机初始化权重")
    
    # 确定是否需要训练
    do_train = True
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        # 如果指定了预训练权重，询问用户是否需要训练
        try:
            user_input = input("已找到预训练权重文件。是否需要在此基础上继续训练？(y/n，默认y): ").strip().lower()
            if user_input in ['n', 'no', '否']:
                do_train = False
                print("将跳过训练过程，直接使用预训练模型进行评估")
        except EOFError:
            # 在非交互式环境中默认进行训练
            print("非交互式环境，默认进行训练")
    
    # 训练模型 - 修改train_model函数调用
    import inspect
    train_model_sig = inspect.signature(train_model)
    
    # 检查train_model函数是否接受do_train参数
    if 'do_train' in train_model_sig.parameters:
        # 如果函数已更新，传入do_train参数并获取weights_loaded返回值
        model, entity_embeddings, relation_embeddings, entity_to_id, relation_to_id, weights_loaded = train_model(pretrained_weights_path, do_train=do_train)
    else:
        # 如果函数未更新，按照原方式调用（兼容旧版本）
        model, entity_embeddings, relation_embeddings, entity_to_id, relation_to_id = train_model(pretrained_weights_path)
    
    # 评估嵌入质量
    evaluate_embeddings(entity_embeddings, entity_to_id)
    
    print("\nTransE模型训练和评估完成！")


if __name__ == "__main__":
    main()