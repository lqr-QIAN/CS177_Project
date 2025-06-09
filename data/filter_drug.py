import pandas as pd

# 路径
comb_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/drugcombs_response_with_cellline.csv'
drug_info_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_drug_info.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/newdrugcombs_response_filtered.csv'

# 加载数据
comb_data = pd.read_csv(comb_path)
drug_info = pd.read_csv(drug_info_path)

# 提取 dname 列，去掉缺失值，转为小写集合
drug_names = drug_info['dname'].dropna().unique()
drug_names_lower = set([d.lower() for d in drug_names])

# 判断每行是否合法
def is_valid_row(row):
    drug1 = str(row['Drug1']).lower()
    drug2 = str(row['Drug2']).lower()
    return (drug1 in drug_names_lower) and (drug2 in drug_names_lower)

# 过滤数据
filtered_data = comb_data[comb_data.apply(is_valid_row, axis=1)].reset_index(drop=True)

# 保存
filtered_data.to_csv(output_path, index=False)

print(f"✅ 已保存过滤后的数据到: {output_path}")
print(f"原始数据行数: {len(comb_data)} -> 过滤后行数: {len(filtered_data)}")
