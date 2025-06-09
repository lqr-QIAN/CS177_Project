import pandas as pd
import re

# ========== 文件路径 ==========
comb_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/newdrugcombs_response_filtered.csv'
cell_info_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_cell_info.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/newdrugcombs_response_cell_filtered.csv'

# ========== 加载数据 ==========
comb_data = pd.read_csv(comb_path)
cell_info = pd.read_csv(cell_info_path)

# ========== 标准化函数 ==========
def standardize_name(name):
    return re.sub(r'[\s\-_]', '', str(name).lower())

# ========== 构建 cell_line_name 标准化集合 ==========
cell_names_standardized = set(standardize_name(name) for name in cell_info['cell_line_name'].dropna().unique())

# ========== 过滤 comb_data ==========
filtered_data = comb_data[comb_data['cell_line'].apply(lambda x: standardize_name(x) in cell_names_standardized)].reset_index(drop=True)

# ========== 保存结果 ==========
filtered_data.to_csv(output_path, index=False)

print(f"✅ 过滤完成！原始行数: {len(comb_data)} -> 过滤后行数: {len(filtered_data)}")
print(f"✅ 已保存至: {output_path}")
