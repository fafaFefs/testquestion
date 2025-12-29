"""
完整的HER材料生成工作流程
整合数据加载、训练、生成和评估
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# 添加模块路径 - 使用当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from her_material_generator.data.downloader import MaterialDataDownloader
    from her_material_generator.data.processor import MaterialDataProcessor
    from her_material_generator.models.diffusion_model import GraphDiffusionModel, DiffusionScheduler
    from her_material_generator.models.property_predictor import PropertyPredictor
    from her_material_generator.trainer import MaterialGeneratorTrainer
    from her_material_generator.generator import MaterialGenerator
    from her_material_generator.evaluator import MaterialEvaluator

    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n请确保已安装所有依赖：")
    print("  pip install torch torch-geometric numpy matplotlib seaborn tqdm requests pandas")
    sys.exit(1)


def step1_download_and_process_data(data_dir=".", force_download=False):
    """
    步骤1：下载和处理数据
    """
    print("\n" + "=" * 60)
    print("步骤 1：下载和处理数据")
    print("=" * 60)

    # 创建下载器
    downloader = MaterialDataDownloader(data_dir=data_dir)

    # 下载数据
    print("\n正在下载数据...")
    data_files = downloader.download_all()

    # 处理数据
    print("\n正在处理数据...")
    processor = MaterialDataProcessor()

    processed_data = []

    # 处理HER催化剂数据
    if 'her_catalyst' in data_files and os.path.exists(data_files['her_catalyst']):
        print(f"\n处理HER催化剂数据: {data_files['her_catalyst']}")
        try:
            graphs = processor.process_json_data(data_files['her_catalyst'])
            processed_data.extend(graphs)
            print(f"  ✓ 处理了 {len(graphs)} 个HER催化剂材料")
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    # 处理C2DB数据
    if 'c2db' in data_files and os.path.exists(data_files['c2db']):
        print(f"\n处理C2DB数据: {data_files['c2db']}")
        try:
            graphs = processor.process_json_data(data_files['c2db'])
            processed_data.extend(graphs)
            print(f"  ✓ 处理了 {len(graphs)} 个C2DB材料")
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    # 保存处理后的数据
    processed_path = os.path.join(data_dir, 'processed_materials.pkl')
    if processed_data:
        processor.save_processed_data(processed_data, processed_path)
        print(f"\n✓ 处理完成！共 {len(processed_data)} 个材料")
        print(f"  保存位置: {processed_path}")
        return processed_path
    else:
        print("\n⚠ 未处理任何数据，将使用模拟数据")
        return None


def step2_train_model(data_path, output_dir="./checkpoints", epochs=50, batch_size=16):
    """
    步骤2：训练扩散模型和性质预测器
    """
    print("\n" + "=" * 60)
    print("步骤 2：训练模型")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 导入训练相关模块
    from torch.utils.data import DataLoader
    from train_her_materials import MaterialDataset, collate_fn

    # 加载数据
    print("\n加载训练数据...")
    dataset = MaterialDataset(data_path=data_path, num_samples=1000)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    # 创建模型
    print("\n初始化模型...")
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

    # 创建训练器
    trainer = MaterialGeneratorTrainer(
        diffusion_model=diffusion_model,
        property_predictor=property_predictor,
        device=device,
        learning_rate=1e-4
    )

    # 训练
    print(f"\n开始训练（{epochs} 个epoch）...")
    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # 训练
        train_losses = trainer.train_epoch(train_loader)
        print(f"  训练损失: {train_losses['total']:.4f}")

        # 验证
        val_losses = trainer.validate(val_loader)
        print(f"  验证损失: {val_losses['total']:.4f}")

        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(output_dir, 'best_model.pt')
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
            print(f"  ✓ 保存最佳模型 (损失: {best_val_loss:.4f})")

    print(f"\n✓ 训练完成！模型保存在: {output_dir}")
    return os.path.join(output_dir, 'best_model.pt')


def extract_formula_from_material(material):
    """
    从材料图数据中提取原子式
    基于节点特征中的原子序数信息
    """
    # 元素周期表（前92个元素）
    elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U'
    ]
    
    # 获取节点特征
    node_features = material.x.cpu().numpy() if hasattr(material.x, 'cpu') else material.x
    
    # 从节点特征的第一维提取原子序数（归一化的，需要还原）
    # features[0] = idx / 92.0，所以 idx = features[0] * 92
    atomic_numbers = []
    for node in node_features:
        if len(node) > 0:
            # 原子序数在第一个特征（归一化后的）
            atomic_num_float = node[0] * 92.0
            atomic_num = int(round(atomic_num_float))
            # 限制在有效范围内
            atomic_num = max(1, min(92, atomic_num))
            atomic_numbers.append(atomic_num - 1)  # 转换为0-based索引
    
    # 统计元素数量
    element_counts = Counter([elements[num] for num in atomic_numbers if 0 <= num < len(elements)])
    
    # 生成原子式（按元素符号排序，常见元素优先）
    if not element_counts:
        return "Unknown"
    
    # 按元素符号排序生成原子式
    sorted_elements = sorted(element_counts.items())
    formula_parts = []
    for element, count in sorted_elements:
        if count > 1:
            formula_parts.append(f"{element}{count}")
        else:
            formula_parts.append(element)
    
    return "".join(formula_parts)


def step3_generate_materials(checkpoint_path, output_dir=".", num_materials=100):
    """
    步骤3：生成新材料并评估性能
    """
    print("\n" + "=" * 60)
    print("步骤 3：生成材料和性能评估")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    diffusion_model = GraphDiffusionModel(
        node_dim=92,
        edge_dim=4,
        hidden_dim=256,
        num_layers=4,
        num_timesteps=1000
    )
    diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])

    property_predictor = PropertyPredictor(
        node_dim=92,
        edge_dim=4,
        hidden_dim=256,
        num_layers=4
    )
    property_predictor.load_state_dict(checkpoint['property_predictor_state_dict'])

    # 创建生成器
    generator = MaterialGenerator(
        diffusion_model=diffusion_model,
        property_predictor=property_predictor,
        device=device
    )

    # 目标性质：ΔG_H接近0，高稳定性，高可合成性
    target_properties = {
        'dg_h': 0.0,  # 目标ΔG_H = 0 eV
        'thermodynamic_stability': -0.5,  # 形成能
        'synthesizability': 0.8  # 可合成性
    }

    print(f"\n生成 {num_materials} 个材料...")
    print(f"目标性质: {target_properties}")

    # 生成并评估
    materials, properties = generator.generate_and_evaluate(
        num_materials=num_materials,
        num_nodes=50,
        target_properties=target_properties
    )

    # 保存结果（如果是当前目录，不需要创建）
    if output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    results = {
        'target_properties': target_properties,
        'generated_materials': len(materials),
        'properties': properties
    }

    results_path = os.path.join(output_dir, 'generation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 生成完成！结果保存在: {output_dir}")

    # 可视化
    print("\n生成可视化图表...")
    evaluator = MaterialEvaluator(property_predictor, device)

    # ΔG_H分布图
    evaluator.visualize_properties(
        properties,
        save_path=os.path.join(output_dir, 'property_distribution.png')
    )

    # 稳定性曲线
    evaluator.plot_stability_curves(
        properties,
        save_path=os.path.join(output_dir, 'stability_curves.png')
    )

    # 找出最优材料
    print("\n筛选最优材料...")
    optimal_materials = evaluator.find_optimal_materials(
        properties,
        top_k=10,
        dg_h_threshold=0.2,
        synthesizability_threshold=0.7
    )

    print(f"\n找到 {len(optimal_materials)} 个最优材料:")
    for idx, (mat_idx, props) in enumerate(optimal_materials):
        # 提取材料原子式
        material = materials[mat_idx]
        formula = extract_formula_from_material(material)
        
        print(f"\n材料 #{idx + 1} ({formula}):")
        print(f"  ΔG_H: {props['dg_h']:.4f} eV")
        print(f"  热力学稳定性: {props['thermodynamic_stability']:.4f} eV/atom")
        print(f"  动力学稳定性: {props['kinetic_stability']:.4f} eV")
        print(f"  可合成性: {props['synthesizability']:.4f}")

    # 保存最优材料列表
    optimal_path = os.path.join(output_dir, 'optimal_materials.json')
    optimal_list = []
    for mat_idx, props in optimal_materials:
        material = materials[mat_idx]
        formula = extract_formula_from_material(material)
        optimal_list.append({
            'formula': formula,
            'properties': props
        })
    
    with open(optimal_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_list, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 最优材料列表保存在: {optimal_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='完整的HER材料生成工作流程')
    parser.add_argument('--data_dir', type=str, default='.', help='数据目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='.', help='生成结果目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_materials', type=int, default=100, help='生成材料数量')
    parser.add_argument('--skip_download', action='store_true', help='跳过数据下载步骤')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练步骤')
    parser.add_argument('--checkpoint', type=str, default=None, help='使用已有的检查点（跳过训练）')

    args = parser.parse_args()

    print("=" * 60)
    print("HER材料生成完整工作流程")
    print("=" * 60)

    # 步骤1：下载和处理数据
    if not args.skip_download:
        processed_path = step1_download_and_process_data(args.data_dir)
    else:
        processed_path = os.path.join(args.data_dir, 'processed_materials.pkl')
        if not os.path.exists(processed_path):
            print(f"\n⚠ 未找到处理后的数据: {processed_path}")
            processed_path = None

    # 步骤2：训练模型
    checkpoint_path = args.checkpoint
    if not args.skip_training and checkpoint_path is None:
        if processed_path and os.path.exists(processed_path):
            checkpoint_path = step2_train_model(
                processed_path,
                output_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            print("\n⚠ 未找到处理后的数据，使用模拟数据训练")
            checkpoint_path = step2_train_model(
                None,
                output_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
    elif checkpoint_path is None:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            print(f"\n✗ 未找到检查点: {checkpoint_path}")
            print("请先训练模型或指定检查点路径")
            return

    # 步骤3：生成材料和评估
    if checkpoint_path and os.path.exists(checkpoint_path):
        step3_generate_materials(
            checkpoint_path,
            output_dir=args.output_dir,
            num_materials=args.num_materials
        )
    else:
        print(f"\n✗ 未找到检查点: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("工作流程完成！")
    print("=" * 60)
    print(f"\n结果文件：")
    print(f"  - 模型检查点: {checkpoint_path}")
    print(f"  - 生成结果: {args.output_dir}")
    print(f"  - 可视化图表: {args.output_dir}/*.png")


if __name__ == "__main__":
    main()

