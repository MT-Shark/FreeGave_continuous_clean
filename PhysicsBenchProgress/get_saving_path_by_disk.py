import os
import json

val_json = '/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchProgress/val_all.json'
disk = 'disk0'
root = '/media/HDD2/siyuan/PhysicsBenchmark'
output_txt = f'/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchProgress/{disk}_val.txt'

# 读取 val_json, 遍历所有 value，将包含 'disk0' 的value 取出， 将 join(root,value) 写入 output_txt
with open(val_json, 'r') as f:
    data = json.load(f)

selected_paths = []
for value in data.values():
    if f'{disk}/' in value:
        full_path = os.path.join(root, value)
        selected_paths.append(full_path)

# 写入到 output_txt
with open(output_txt, 'w') as f:
    for path in selected_paths:
        f.write(path + '\n')
