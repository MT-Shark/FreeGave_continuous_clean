
import os
import shutil
import json
import imageio
import numpy as np
import OpenEXR
import Imath
import random
import open3d as o3d
import glob
import argparse
from multiprocessing import Pool, cpu_count



''' sample novel views'''

import re

random.seed(666)  # 你可以换成任何整数

# 参数设置


# 读取所有相机位姿（camera-to-world）矩阵
def load_camera_poses(root_path, pattern):
    pose_files = glob.glob(os.path.join(root_path, pattern))
    poses = []
    selected_files = []
    for file in pose_files:
        name = os.path.basename(file)
        if 'blender_CineCamera' in name and 'Moving' not in name:
            with open(file, 'r') as f:
                data = json.load(f)
                pose = np.array(data['frames'][0]['transform_matrix'])
                poses.append(pose)
                selected_files.append(file)
    return poses, selected_files

# 计算两个相机之间的位置距离（只考虑平移部分）
def compute_distance(pose1, pose2):
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    return np.linalg.norm(t1 - t2)

# 计算所有相机之间的距离矩阵
def compute_distance_matrix(poses):
    n = len(poses)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = compute_distance(poses[i], poses[j])
            else:
                dist_matrix[i, j] = np.inf
    return dist_matrix

# 采样 N 个代表性视角
def sample_views(poses, N):
    dist_matrix = compute_distance_matrix(poses)
    selected = []
    remaining = list(range(len(poses)))

    while len(selected) < N and remaining:
        min_dists = []
        for i in remaining:
            dists = dist_matrix[i, remaining]
            sorted_dists = np.sort(dists)
            min_dists.append((i, sorted_dists[0], sorted_dists[1]))

        min_dists.sort(key=lambda x: x[1])
        first = min_dists[0][0]
        second = min_dists[1][0] if min_dists[1][0] != first else random.choice([x[0] for x in min_dists[2:]])

        selected.append(first)
        remaining.remove(first)

    return selected

# 提取 view 编号
def extract_view_id(filename):
    match = re.search(r'_(\d+)\.json$', filename)
    return int(match.group(1)) if match else -1



def sample_novel_views(root_path , N =2):
    pattern = '*.json'
    # 主流程
    poses, pose_files = load_camera_poses(root_path, pattern)
    sampled_indices = sample_views(poses, N)
    sampled_files = [pose_files[i] for i in sampled_indices]
    train_files = [f for i, f in enumerate(pose_files) if i not in sampled_indices]

    train_frames = []
    val_frames = []
    test_frames = []

    meta_template = None  # 用于保存非 frames 的公共字段

    for f in train_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if meta_template is None:
                meta_template = {k: v for k, v in data.items() if k != 'frames'}
            frames = data['frames']
            view = extract_view_id(os.path.basename(f))
            for frame in frames:
                frame['view'] = view
                if frame['time'] > 0.5:
                    test_frames.append(frame)
                else:
                    train_frames.append(frame)

    for f in sampled_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if meta_template is None:
                meta_template = {k: v for k, v in data.items() if k != 'frames'}
            frames = data['frames']
            view = extract_view_id(os.path.basename(f))
            for frame in frames:
                frame['view'] = view
                if frame['time'] > 0.5:
                    test_frames.append(frame)
                else:
                    val_frames.append(frame)

    # 构建最终数据结构
    train_data = dict(meta_template)
    train_data['frames'] = train_frames

    val_data = dict(meta_template)
    val_data['frames'] = val_frames

    test_data = dict(meta_template)
    test_data['frames'] = test_frames

    # 保存结果
    with open(os.path.join(root_path, 'transforms_train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(root_path, 'transforms_val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(os.path.join(root_path, 'transforms_test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)

    # 输出采样结果
    print("✅ 采样出的相机视角文件：")
    for f in sampled_files:
        print(f)























''' compress '''
def load_exr_rgb_openexr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # 读取 R、G、B 通道
    r_str = exr_file.channel('R', pt)
    g_str = exr_file.channel('G', pt)
    b_str = exr_file.channel('B', pt)

    r = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
    g = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
    b = np.frombuffer(b_str, dtype=np.float32).reshape(size[1], size[0])

    # 合并为 RGB 图像
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

def tone_map(rgb, gamma=2.2):
    # 色调映射 + Gamma 校正
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 1.0 / gamma)
    rgb_8bit = (rgb * 255).astype(np.uint8)
    return rgb_8bit

def exr2jpg(exr_path, jpg_path):
    rgb = load_exr_rgb_openexr(exr_path)
    rgb_8bit = tone_map(rgb)
    imageio.imwrite(jpg_path, rgb_8bit)


def load_rgb_image(image_path):
    rgb = imageio.imread(image_path)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def depth_to_point_cloud(depth, intrinsics, transform_matrix):
    h, w = depth.shape
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_world = (transform_matrix @ points_hom.T).T[:, :3]
    return points_world

def depth_to_point_cloud_blender(depth, intrinsics, transform_matrix):
    """
    将 Blender 相机下的深度图转换为世界坐标系中的点云
    Blender 相机坐标系：+X 右，+Y 上，+Z 向后（所以相机向前是 -Z）
    """
    h, w = depth.shape
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy


    points = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_world = (transform_matrix @ points_hom.T).T[:, :3]

    return points_world

def load_exr_depth_openexr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
    return depth

def load_exr_seg_openexr(file_path, to_uint8=False):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    seg_str = exr_file.channel('R', pt)
    seg = np.frombuffer(seg_str, dtype=np.float32).reshape(size[1], size[0])

    if to_uint8:
        # 如果是离散标签（如语义分割），直接转换
        seg = seg.astype(np.uint8)

        # 如果是连续值（如概率图），可以归一化后再转
        # seg = ((seg - seg.min()) / (seg.max() - seg.min()) * 255).astype(np.uint8)

    return seg

def process_depth_folders(root):
    for subdir, dirs, files in os.walk(root):
        if os.path.basename(subdir) == 'depth':
            exr_files = glob.glob(os.path.join(subdir, '*.exr'))
            for exr_file in exr_files:
                depth = load_exr_depth_openexr(exr_file)
                npz_path = exr_file.replace('.exr', '.npz')
                np.savez_compressed(npz_path, depth=depth)
                os.remove(exr_file)


def process_single_folder(args):
    # root, log_file = args
    root, log_file, issue_log_file, ue_path_log_file = args
    sequence_path_cache = set()

    all_success = True
    files_to_delete = []

    for subdir, dirs, files in os.walk(root):
        base = os.path.basename(subdir)

        if base in ['depth', 'seg']:
            exr_files = glob.glob(os.path.join(subdir, '*.exr'))
            success_list = []

            for exr_file in exr_files:
                try:
                    if base == 'depth':
                        data = load_exr_depth_openexr(exr_file)
                        npz_path = exr_file.replace('.exr', '.npz')
                        np.savez_compressed(npz_path, depth=data)
                    elif base == 'seg':
                        data = load_exr_seg_openexr(exr_file , to_uint8=False)
                        npz_path = exr_file.replace('.exr', '.npz')
                        np.savez_compressed(npz_path, seg=data)
                    success_list.append(exr_file)
                except Exception as e:
                    # print(f"❌ Failed to process {exr_file}: {e}")
                    msg = f"Failed to process {exr_file}: {e}"
                    print(f"❌ {msg}")
                    log_issue(issue_log_file, ue_path_log_file, msg, sequence_path_cache)
                    all_success = False

            if len(success_list) == len(exr_files):
                files_to_delete.extend(success_list)
            else:
                # print(f"⚠️ Not all EXR files in {subdir} were successfully converted. Skipping deletion.")
                msg = f"⚠️ Not all EXR files in {subdir} were successfully converted. Skipping deletion."
                print(msg)
                log_issue(issue_log_file, ue_path_log_file, msg, sequence_path_cache,)
                all_success = False

    # 如果整个文件夹所有 exr 都成功转换，统一删除
    if all_success:
        for exr_file in files_to_delete:
            try:
                os.remove(exr_file)
            except Exception as e:
                print(f"❌ Failed to delete {exr_file}: {e}")
        print(f"🗑️ Deleted {len(files_to_delete)} EXR files from {root}")
    else:
        print(f"⚠️ Skipped deletion for {root} due to partial failure.")

    # 调用视角采样函数（确保数据准备完毕后执行）
    try:
        sample_novel_views(root, N=2)
        print(f"📸 Sampled novel views for: {root}")
    except Exception as e:
        print(f"❌ Failed to sample novel views in {root}: {e}")

    # 记录处理完成的路径
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{root}\n")
    print(f"✅ Finished: {root}")

def get_sequence_list(root):
    sequence_list = []
    for first_level in os.listdir(root):
        first_path = os.path.join(root, first_level)
        if os.path.isdir(first_path):
            for second_level in os.listdir(first_path):
                second_path = os.path.join(first_path, second_level)
                if os.path.isdir(second_path):
                    sequence_list.append(second_path)
    sequence_list = sorted(sequence_list)
    return sequence_list


import os
import re
from typing import Optional, Set

# 正则匹配 MovieRender 序列目录（以 _sequence 结尾）
PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^<>:\"|?*\r\n]+?_sequence)", re.IGNORECASE)

# 清理脚本命令模板
SCRIPT_COMMAND_TEMPLATE = (
    "python P:\\PhysicBenchmark\\Content\\PhysicsScenario\\Scripts\\DataProcess\\clean_progress.py "
    "{path} --delete_sequence True --delete_record_file True --delete_movingCamera True"
)


def sequence_path_to_ue_path(sequence_path: str) -> Optional[str]:
    """
    将 MovieRender 序列目录转换为 UE 资源路径。
    规则：
      - TwoCombinations → /Game/.../TwoCombinations/two_combs/Asset.Asset
      - BasicScene → /Game/.../BasicScene/Asset.Asset
      - TripleCombinations → 若路径中显式包含 TripleCombinations，则使用其后一级目录作为轮次；
                             若没有显式标识，则默认类别本身即轮次（SecondRound、ThirdRound 等）。
    """
    normalized = os.path.normpath(sequence_path)
    parts = normalized.split(os.sep)
    lower_parts = [part.lower() for part in parts]

    try:
        mr_idx = lower_parts.index("movierenders")
    except ValueError:
        return None

    if mr_idx + 1 >= len(parts):
        return None

    category = parts[mr_idx + 1]
    asset_segment = parts[-1]

    if not asset_segment.lower().endswith("_sequence"):
        return None

    asset_name = asset_segment[:-len("_sequence")]
    root = "/Game/PhysicsScenario/Levels"

    if category.lower() == "twocombinations":
        ue_path = f"{root}/TwoCombinations/two_combs/{asset_name}.{asset_name}"
    elif category.lower() == "basicscene":
        ue_path = f"{root}/BasicScene/{asset_name}.{asset_name}"
    elif category.lower() == "triplecombinations":
        if mr_idx + 2 >= len(parts):
            return None
        round_name = parts[mr_idx + 2]
        ue_path = f"{root}/TripleCombinations/{round_name}/{asset_name}.{asset_name}"
    else:
        # 默认视为 TripleCombinations 的轮次目录，如 SecondRound、ThirdRound 等
        ue_path = f"{root}/TripleCombinations/{category}/{asset_name}.{asset_name}"

    return ue_path.replace("\\", "/")


def log_issue(issue_log_path: str, ue_only_log_path: str, message: str, sequence_path_cache: Set[str]) -> None:
    """
    从 message 中提取 _sequence 结尾的目录，写入 issue_log，并追加对应 UE 清理命令。
    使用 sequence_path_cache（存储小写路径）去重，确保每个目录仅记录一次。
    """
    message = message.strip().replace('"', '')
    recorded = False

    for match in PATH_PATTERN.finditer(message):
        raw_path = match.group(1).rstrip("\\/")
        normalized_path = os.path.normpath(raw_path)
        cache_key = normalized_path.lower()

        if cache_key in sequence_path_cache:
            continue

        ue_path = sequence_path_to_ue_path(normalized_path)


        command_line = (
            SCRIPT_COMMAND_TEMPLATE.format(path=ue_path)
            if ue_path
            else "# 未能解析出对应的 UE 资源路径"
        )



        with open(issue_log_path, "a", encoding="utf-8") as f:
            f.write(f"{normalized_path}\n")
            f.write(f"{command_line}\n\n")


        if ue_path:
            with open(ue_only_log_path, "a", encoding="utf-8") as f:
                f.write(f"{ue_path}\n")

        sequence_path_cache.add(cache_key)
        recorded = True

    if not recorded:
        # 若未匹配到合法路径，可在此处视需求打印或记录调试信息
        pass


def main():
    parser = argparse.ArgumentParser(description="Compress EXR files to NPZ and clean up.")
    parser.add_argument("--root", default=r"P:\PhysicBenchmark\Saved\MovieRenders" , help="Root directory containing sequences to process")
    parser.add_argument("--log", default=r"P:\PhysicBenchmark\Content\PhysicsScenario\Scripts\DataProcess\compress_log.txt", help="Path to log file")
    parser.add_argument("--workers", default= 12, type=int, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()


    issue_log_file = args.log.replace('compress_log.txt', 'issue_log.txt')
    ue_path_log_file = args.log.replace('compress_log.txt', 'delete_render_ue_path_log.txt')

    open(issue_log_file, 'w').close()  # Clears the file at the start
    open(ue_path_log_file, 'w').close()

    root = args.root
    log_file = args.log
    workers = args.workers

    if not os.path.isdir(root):
        print(f"❌ Error: '{root}' is not a valid directory.")
        return

    sequence_list = get_sequence_list(root)
    total = len(sequence_list)

    print(f"📁 Found {total} sequences under: {root}")
    # task_args = [(sequence_path, log_file) for sequence_path in sequence_list]
    task_args = [
        (sequence_path, log_file, issue_log_file, ue_path_log_file)
        for sequence_path in sequence_list
    ]

    with Pool(processes=workers) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_single_folder, task_args), 1):
            print(f"🔄 Progress: {idx}/{total}")


if __name__ == "__main__":
    main()