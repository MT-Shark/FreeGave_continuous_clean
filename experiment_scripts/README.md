# Mini Subset 批量实验脚本

## 关键文件列表

### 1. 核心脚本文件

| 文件 | 路径 | 功能描述 |
|------|------|----------|
| **init_scene_list.sh** | `experiment_scripts/init_scene_list.sh` | 初始化脚本：扫描场景目录，生成场景列表和状态追踪文件 |
| **run_single_scene.sh** | `experiment_scripts/run_single_scene.sh` | 单元执行脚本：处理单个场景的训练+评估流程 |
| **run_multi_gpu.sh** | `experiment_scripts/run_multi_gpu.sh` | 多进程调度脚本：并行处理多个场景，使用文件锁防止并发冲突 |
| **status.sh** | `experiment_scripts/status.sh` | 状态查看脚本：显示当前实验进度 |

### 2. 训练与评估脚本

| 文件 | 路径 | 功能描述 |
|------|------|----------|
| **train.py** | `train.py` | 训练脚本：Gaussian Splatting 模型训练 |
| **render_eval_mask.py** | `render_eval_mask.py` | 渲染评估脚本：渲染图像并计算指标（支持分割遮罩） |

### 3. 状态追踪文件（运行时生成）

| 文件 | 路径 | 功能描述 |
|------|------|----------|
| **all_scenes.txt** | `experiment_scripts/all_scenes.txt` | 所有场景的完整列表 |
| **incomplete.txt** | `experiment_scripts/incomplete.txt` | 待处理的场景列表 |
| **processing.txt** | `experiment_scripts/processing.txt` | 当前正在处理的场景 |
| **complete.txt** | `experiment_scripts/complete.txt` | 已完成的场景列表 |
| **failed.txt** | `experiment_scripts/failed.txt` | 失败的场景列表 |

### 4. 数据目录

| 目录 | 路径 | 说明 |
|------|------|------|
| **SinglePhysics** | `/media/SSD/siyuan/FreeGave/PhysicsBenchmark/mini_subset/Test/SinglePhysics` | 单物理场景 |
| **DoublePhysics** | `/media/SSD/siyuan/FreeGave/PhysicsBenchmark/mini_subset/Test/DoublePhysics` | 双物理场景 |
| **TriplePhysics** | `/media/SSD/siyuan/FreeGave/PhysicsBenchmark/mini_subset/Test/TriplePhysics` | 三物理场景 |

---

## 文件结构

```
FreeGave/
├── train.py                          # 训练脚本
├── render_eval_mask.py               # 渲染评估脚本（支持分割遮罩）
│
├── experiment_scripts/               # 批量执行脚本目录
│   ├── README.md                     # 本说明文件
│   ├── init_scene_list.sh            # 初始化脚本
│   ├── run_single_scene.sh           # 单场景执行脚本
│   ├── run_multi_gpu.sh              # 多GPU并行脚本
│   ├── status.sh                     # 进度查看脚本
│   │
│   ├── all_scenes.txt                # [生成] 所有场景
│   ├── incomplete.txt                # [生成] 待处理
│   ├── processing.txt                # [生成] 处理中
│   ├── complete.txt                  # [生成] 已完成
│   ├── failed.txt                    # [生成] 失败
│   └── logs/                         # [生成] 日志目录
│       ├── main.log
│       ├── worker_0_gpu0.log
│       └── ...
│
├── PhysicsBenchmark/mini_subset/Test/  # 数据目录
│   ├── SinglePhysics/
│   ├── DoublePhysics/
│   └── TriplePhysics/
│
└── output/mini_subset_exp/           # [生成] 输出目录
    ├── SinglePhysics/
    ├── DoublePhysics/
    └── TriplePhysics/
```

---

## 使用方法

### 1. 初始化（首次运行）

```bash
cd /media/SSD/siyuan/FreeGave
chmod +x experiment_scripts/*.sh

# 扫描场景目录，生成列表文件
./experiment_scripts/init_scene_list.sh
```

### 2. 配置 GPU

编辑 `run_multi_gpu.sh`，修改 `GPU_LIST`：

```bash
# 指定可用的 GPU 列表
GPU_LIST=(0 1 2 3)  # 修改为你的 GPU id
```

### 3. 运行实验

#### 多 GPU 并行运行（推荐）

```bash
./experiment_scripts/run_multi_gpu.sh
```

#### 单场景手动运行

```bash
./experiment_scripts/run_single_scene.sh <gpu_id> <scene_path> [output_base_dir]

# 示例
./experiment_scripts/run_single_scene.sh 0 /media/SSD/siyuan/FreeGave/PhysicsBenchmark/mini_subset/Test/SinglePhysics/scene1
```

### 4. 监控进度

```bash
# 使用 status.sh 查看进度
./experiment_scripts/status.sh

# 或手动查看
echo "Incomplete: $(wc -l < experiment_scripts/incomplete.txt)"
echo "Processing: $(wc -l < experiment_scripts/processing.txt)"
echo "Complete:   $(wc -l < experiment_scripts/complete.txt)"

# 实时查看某个 worker 的日志
tail -f experiment_scripts/logs/worker_0_gpu0.log
```

### 5. 断点续跑

多进程脚本支持断点续跑：
- 直接再次运行 `./experiment_scripts/run_multi_gpu.sh`
- 它会自动从 `incomplete.txt` 继续处理未完成的场景

### 6. 重新处理失败的场景

```bash
# 把失败的场景放回 incomplete 列表
cat experiment_scripts/failed.txt >> experiment_scripts/incomplete.txt
> experiment_scripts/failed.txt

# 重新运行
./experiment_scripts/run_multi_gpu.sh
```

---

## 训练参数配置

当前配置的训练和评估参数（在 `run_single_scene.sh` 中定义）：

### 训练 (train.py)

```bash
python train.py -s "$SCENE_PATH" -m "$OUTPUT_DIR" \
    --is_blender \
    --max_time 0.45 \
    --physics_code 16 \
    --densify_grad_threshold 0.00025 \
    --iterations 25000
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--is_blender` | - | 使用 Blender 格式数据 |
| `--max_time` | 0.45 | 最大时间范围 |
| `--physics_code` | 16 | 物理编码维度 |
| `--densify_grad_threshold` | 0.00025 | 密度化梯度阈值 |
| `--iterations` | 25000 | 训练迭代次数 |

### 评估 (render_eval_mask.py)

```bash
python render_eval_mask.py -m "$OUTPUT_DIR" \
    --seg_mask \
    --iteration 25000 \
    --save_images \
    --save_fid_threshold 0.75
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--seg_mask` | - | 启用分割遮罩 |
| `--iteration` | 25000 | 加载的迭代checkpoint |
| `--save_images` | - | 保存渲染图像 |
| `--save_fid_threshold` | 0.75 | 只保存 fid > 0.75 的帧（最后25%时间段） |

---

## 输出结构

```
output/mini_subset_exp/
├── SinglePhysics/
│   └── scene_name/
│       ├── point_cloud/iter_25000/       # 点云模型
│       ├── train/ours_25000/output/      # 训练集渲染结果
│       │   ├── full/                     # 完整图像
│       │   │   ├── renders/
│       │   │   └── gt/
│       │   ├── foreground/               # 前景（物体）
│       │   │   ├── renders/
│       │   │   └── gt/
│       │   └── background/               # 背景
│       │       ├── renders/
│       │       └── gt/
│       ├── test/ours_25000/output/       # 测试集渲染结果
│       └── metrics_summary.json          # 评估指标汇总
├── DoublePhysics/
└── TriplePhysics/
```

---

## 日志说明

每个 worker 的日志保存在 `logs/worker_{id}_gpu{gpu}.log`，包含：
- 开始/结束时间戳
- 正在处理的场景名称
- 训练和评估的输出
- 成功/失败状态
