"""
训练脚本：基于扩散模型的二维HER催化材料生成
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import os
import argparse
from tqdm import tqdm
import json

from her_material_generator.models.diffusion_model import GraphDiffusionModel, DiffusionScheduler
from her_material_generator.models.property_predictor import PropertyPredictor
from her_material_generator.trainer import MaterialGeneratorTrainer
from her_material_generator.generator import MaterialGenerator
from her_material_generator.evaluator import MaterialEvaluator


class MaterialDataset(Dataset):
    """材料数据集"""
    
    def __init__(self, data_path: str = None, num_samples: int = 1000):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径（支持.pkl或.json格式）
            num_samples: 样本数量（用于模拟数据）
        """
        if data_path and os.path.exists(data_path):
            # 从文件加载数据
            self.data = self._load_from_file(data_path)
        else:
            # 生成模拟数据用于演示
            print(f"生成 {num_samples} 个模拟材料样本...")
            self.data = self._generate_synthetic_data(num_samples)
    
    def _load_from_file(self, path: str):
        """从文件加载数据"""
        import pickle
        
        if path.endswith('.pkl'):
            # 加载处理后的图数据
            print(f"从 {path} 加载处理后的数据...")
            with open(path, 'rb') as f:
                graph_data_list = pickle.load(f)
            
            # 转换为(图, 性质)元组格式
            data_list = []
            for graph_data in graph_data_list:
                properties = {
                    'dg_h': getattr(graph_data, 'dg_h', torch.tensor(0.0)).item(),
                    'thermodynamic_stability': getattr(graph_data, 'thermodynamic_stability', torch.tensor(0.0)).item(),
                    'kinetic_stability': getattr(graph_data, 'kinetic_stability', torch.tensor(1.0)).item(),
                    'synthesizability': getattr(graph_data, 'synthesizability', torch.tensor(0.5)).item()
                }
                data_list.append((graph_data, properties))
            
            print(f"加载了 {len(data_list)} 个材料")
            return data_list
        
        elif path.endswith('.json'):
            # 加载JSON格式的原始数据
            print(f"从 {path} 加载JSON数据...")
            with open(path, 'r', encoding='utf-8') as f:
                materials = json.load(f)
            
            # 使用processor处理数据
            from her_material_generator.data.processor import MaterialDataProcessor
            processor = MaterialDataProcessor()
            
            data_list = []
            for material in materials:
                properties = {
                    'dg_h': material.get('dg_h', 0.0),
                    'thermodynamic_stability': material.get('formation_energy_per_atom', 0.0),
                    'kinetic_stability': material.get('kinetic_stability', 1.0),
                    'synthesizability': material.get('synthesizability', 0.5)
                }
                
                structure = material.get('structure', {})
                graph_data = processor.structure_to_graph(structure, properties)
                data_list.append((graph_data, properties))
            
            print(f"处理并加载了 {len(data_list)} 个材料")
            return data_list
        
        else:
            raise ValueError(f"不支持的文件格式: {path}")
    
    def _generate_synthetic_data(self, num_samples: int):
        """生成合成数据用于演示"""
        data_list = []
        
        for i in range(num_samples):
            # 随机生成图结构
            num_nodes = np.random.randint(20, 100)
            num_edges = np.random.randint(num_nodes, num_nodes * 3)
            
            # 节点特征（原子类型、价态等）
            node_features = torch.randn(num_nodes, 92)  # 92维原子特征
            
            # 边索引
            edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
            
            # 边特征（键长、键角等）
            edge_attr = torch.randn(num_edges, 4)
            
            # 创建图数据
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            
            # 生成模拟性质标签
            # ΔG_H: 目标接近0，范围大约[-1, 1]
            dg_h = np.random.normal(0, 0.3)
            
            # 热力学稳定性（形成能，负值表示稳定）
            thermo_stab = np.random.normal(-0.5, 0.2)
            
            # 动力学稳定性（能垒，正值表示稳定）
            kinetic_stab = np.random.normal(1.0, 0.3)
            
            # 可合成性（概率）
            synthesizability = np.random.beta(5, 2)  # 偏向高值
            
            properties = {
                'dg_h': torch.tensor(dg_h, dtype=torch.float32),
                'thermodynamic_stability': torch.tensor(thermo_stab, dtype=torch.float32),
                'kinetic_stability': torch.tensor(kinetic_stab, dtype=torch.float32),
                'synthesizability': torch.tensor(synthesizability, dtype=torch.float32)
            }
            
            data_list.append((graph_data, properties))
        
        return data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """批次整理函数"""
    graphs, properties = zip(*batch)
    
    # 将图数据批量化
    batch_graph = Batch.from_data_list(graphs)
    
    # 将性质数据批量化（确保转换为tensor）
    batch_properties = {
        'dg_h': torch.tensor([p['dg_h'] if isinstance(p['dg_h'], (int, float)) else p['dg_h'].item() for p in properties], dtype=torch.float32),
        'thermodynamic_stability': torch.tensor([p['thermodynamic_stability'] if isinstance(p['thermodynamic_stability'], (int, float)) else p['thermodynamic_stability'].item() for p in properties], dtype=torch.float32),
        'kinetic_stability': torch.tensor([p['kinetic_stability'] if isinstance(p['kinetic_stability'], (int, float)) else p['kinetic_stability'].item() for p in properties], dtype=torch.float32),
        'synthesizability': torch.tensor([p['synthesizability'] if isinstance(p['synthesizability'], (int, float)) else p['synthesizability'].item() for p in properties], dtype=torch.float32)
    }
    
    return batch_graph, batch_properties


def main():
    parser = argparse.ArgumentParser(description='训练HER材料生成模型')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径（.pkl或.json格式）')
    parser.add_argument('--num_samples', type=int, default=1000, help='模拟数据样本数（当data_path为None时使用）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 数据集
    print("加载数据集...")
    
    # 如果没有指定数据路径，尝试使用默认路径
    if args.data_path is None:
        # 先尝试当前目录
        default_data_path = os.path.join('.', 'processed_materials.pkl')
        if os.path.exists(default_data_path):
            print(f"使用默认数据路径: {os.path.abspath(default_data_path)}")
            args.data_path = default_data_path
        else:
            # 再尝试 ./data 目录
            default_data_path = os.path.join('./data', 'processed_materials.pkl')
            if os.path.exists(default_data_path):
                print(f"使用默认数据路径: {os.path.abspath(default_data_path)}")
                args.data_path = default_data_path
            else:
                print("未找到数据文件，将使用模拟数据")
                print("提示：运行 'python download_material_data.py --process' 下载和处理数据")
                print("     数据将保存在当前文件夹")
    
    dataset = MaterialDataset(data_path=args.data_path, num_samples=args.num_samples)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 模型
    print("初始化模型...")
    diffusion_model = GraphDiffusionModel(
        node_dim=92,
        edge_dim=4,
        hidden_dim=256,
        num_layers=4,
        num_timesteps=1000
    )
    
    property_predictor = PropertyPredictor(
        node_dim=92,
        edge_dim=4,
        hidden_dim=256,
        num_layers=4
    )
    
    # 训练器
    trainer = MaterialGeneratorTrainer(
        diffusion_model=diffusion_model,
        property_predictor=property_predictor,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"从epoch {start_epoch} 恢复训练")
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_losses = trainer.train_epoch(train_loader)
        print(f"训练损失: {train_losses}")
        
        # 验证
        val_losses = trainer.validate(val_loader)
        print(f"验证损失: {val_losses}")
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
            print(f"保存最佳模型 (验证损失: {best_val_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
    
    print("\n训练完成！")
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.train_history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")


if __name__ == '__main__':
    main()

