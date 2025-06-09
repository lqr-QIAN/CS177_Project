import pandas as pd
import json

# 读取 CSV 文件并创建双向映射
def create_bidirectional_mapping(csv_file):
    # 读取 cell_info_used.csv 文件
    cell_info_df = pd.read_csv(csv_file)
    
    # 创建双向映射字典
    depmap_to_cellname = dict(zip(cell_info_df['depmap_id'], cell_info_df['cell_line_name']))
    cellname_to_depmap = dict(zip(cell_info_df['cell_line_name'], cell_info_df['depmap_id']))
    
    # 保存映射到 JSON 文件
    with open('depmap_to_cellname.json', 'w') as f:
        json.dump(depmap_to_cellname, f, indent=4)
    
    with open('cellname_to_depmap.json', 'w') as f:
        json.dump(cellname_to_depmap, f, indent=4)
    
    print("双向映射已成功创建并保存为 JSON 文件。")
    
    return depmap_to_cellname, cellname_to_depmap

# 检查特定 depmap_id 或 cell_line_name 是否存在映射
def check_mapping(depmap_to_cellname, cellname_to_depmap, key):
    if key in depmap_to_cellname:
        print(f"{key} -> {depmap_to_cellname[key]} (depmap_id -> cell_line_name)")
    elif key in cellname_to_depmap:
        print(f"{key} -> {cellname_to_depmap[key]} (cell_line_name -> depmap_id)")
    else:
        print(f"未找到 {key} 的映射。")

# 指定要检查的 CSV 文件
csv_file = '/root/lanyun-tmp/Project/SynergyX/cell_info_used.csv'

# 创建双向映射
depmap_to_cellname, cellname_to_depmap = create_bidirectional_mapping(csv_file)

# 检查一个 depmap_id 或 cell_line_name
key_to_check = 'ACH-000288'  # 替换成你要检查的 depmap_id 或 cell_line_name
check_mapping(depmap_to_cellname, cellname_to_depmap, key_to_check)
