import os
import shutil

# 生成索引列表
generate_index = list(range(2, 29))

# 获取当前 Python 文件所在目录
dir_cur = os.path.dirname(os.path.abspath(__file__))

for i in generate_index:
    # 创建文件夹路径
    folder_name = f"disk{i}_val"
    folder_path = os.path.join(dir_cur, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 创建空的 txt 文件
    open(os.path.join(folder_path, f"disk{i}_val_complete.txt"), 'w').close()
    open(os.path.join(folder_path, f"disk{i}_val_fail.txt"), 'w').close()
    open(os.path.join(folder_path, "log.txt"), 'w').close()

    # 复制 val_file 中的 txt 文件
    src_txt = f"/media/HDD2/siyuan/PhysicsBenchmark/val_file/disk{i}_val.txt"
    dst_txt = os.path.join(folder_path, f"disk{i}_val.txt")
    if os.path.exists(src_txt):
        shutil.copy(src_txt, dst_txt)
    else:
        print(f"源文件不存在: {src_txt}")

    # 复制 disk0_val 中的 train.sh 文件
    src_sh = os.path.join(dir_cur, "disk0_val", "train.sh")
    dst_sh = os.path.join(folder_path, "train.sh")
    if os.path.exists(src_sh):
        shutil.copy(src_sh, dst_sh)
    else:
        print(f"train.sh 文件不存在: {src_sh}")

print("所有文件夹和文件处理完成。")