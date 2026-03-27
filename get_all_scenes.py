import json
import os

# 输入文件路径
val_file = '/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/val_all.json'
root = ''
output_txt = '/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/val.txt'

# 读取 JSON 文件
with open(val_file, 'r') as f:
    data = json.load(f)

# 提取所有 value 并拼接 root 路径
full_paths = [os.path.join(root, path) for path in data.values()]

# 写入到输出文件
with open(output_txt, 'w') as f:
    for path in full_paths:
        f.write(path + '\n')

print(f"成功将 {len(full_paths)} 个路径写入到 {output_txt} 文件中。")