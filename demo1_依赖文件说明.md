# demo1.py 运行所需的文件清单

## 必需的核心文件

### 1. 主程序文件
- **demo1.py** - 主程序入口文件（当前文件）

### 2. 训练脚本
- **train_her_materials.py** - 提供 MaterialDataset 和 collate_fn（需要放在 `C:\Users\28371\Desktop` 目录下或Python路径中）

### 3. her_material_generator 模块目录（需要完整目录结构）
需要放在 `C:\Users\28371\Desktop\her_material_generator\` 目录下：

#### 3.1 模块根目录文件
- `her_material_generator/__init__.py`
- `her_material_generator/trainer.py` - 训练器
- `her_material_generator/generator.py` - 材料生成器
- `her_material_generator/evaluator.py` - 材料评估器
- `her_material_generator/losses.py` - 损失函数
- `her_material_generator/active_learning.py` - 主动学习
- `her_material_generator/dft_calculator.py` - DFT计算器
- `her_material_generator/rl_optimizer.py` - RL优化器

#### 3.2 data 子模块
- `her_material_generator/data/__init__.py`
- `her_material_generator/data/downloader.py` - 数据下载器（必需）
- `her_material_generator/data/processor.py` - 数据处理器（必需）
- `her_material_generator/data/structure_parser.py` - 结构解析器

#### 3.3 models 子模块
- `her_material_generator/models/__init__.py`
- `her_material_generator/models/diffusion_model.py` - 扩散模型（必需）
- `her_material_generator/models/property_predictor.py` - 性质预测器（必需）
- `her_material_generator/models/gnn_encoder.py` - GNN编码器

## Python依赖包（需要通过pip安装）

```bash
pip install torch torch-geometric numpy matplotlib seaborn tqdm requests pandas
```

## 运行时生成的文件

程序运行时会生成以下文件（保存在当前目录）：

### 数据文件
- `processed_materials.pkl` - 处理后的材料数据（如果执行数据下载和预处理步骤）
- `her_catalyst_data.json` - HER催化剂数据（如果执行数据下载步骤）
- `c2db_data.json` - C2DB数据（如果执行数据下载步骤）

### 模型文件
- `checkpoints/best_model.pt` - 训练后的模型检查点（如果执行训练步骤）

### 输出结果文件
- `generation_results.json` - 生成结果
- `optimal_materials.json` - 最优材料列表（包含原子式）
- `property_distribution.png` - 性质分布图
- `stability_curves.png` - 稳定性曲线图

## 目录结构要求

```
C:\Users\28371\Desktop\
├── demo1.py (如果放在这里运行)
├── train_her_materials.py
└── her_material_generator\
    ├── __init__.py
    ├── trainer.py
    ├── generator.py
    ├── evaluator.py
    ├── losses.py
    ├── active_learning.py
    ├── dft_calculator.py
    ├── rl_optimizer.py
    ├── data\
    │   ├── __init__.py
    │   ├── downloader.py
    │   ├── processor.py
    │   └── structure_parser.py
    └── models\
        ├── __init__.py
        ├── diffusion_model.py
        ├── property_predictor.py
        └── gnn_encoder.py
```

## 注意事项

1. **路径硬编码**：当前代码中硬编码了 `C:\Users\28371\Desktop` 路径，如果要在其他位置运行，需要修改 `demo1.py` 中的 `desktop_path` 变量。

2. **必需模块**：以下模块是必需的核心模块，缺少会导致程序无法运行：
   - `data.downloader`
   - `data.processor`
   - `models.diffusion_model`
   - `models.property_predictor`
   - `trainer`
   - `generator`
   - `evaluator`

3. **可选模块**：其他模块（如 `active_learning.py`、`rl_optimizer.py` 等）如果不使用相关功能，可能不需要。

4. **train_her_materials.py**：该文件需要与 `demo1.py` 在同一个Python路径下，或者在 `C:\Users\28371\Desktop` 目录下。

## 最小运行要求

如果要最小化运行，至少需要：
1. `demo1.py`
2. `train_her_materials.py`
3. `her_material_generator/` 目录及其所有必需子模块文件

