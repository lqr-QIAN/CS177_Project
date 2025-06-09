import numpy as np
import pandas as pd

# ========== 路径 ==========
all_items_path = '/root/lanyun-tmp/Project/SynergyX/data/split/all_items.npy'
drug_info_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_drug_info.csv'
cell_info_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/merged_cell_info.csv'
comb_data_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/newdrugcombs_response_cell_filtered.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/split/all_items_new.npy'

# ========== 加载数据 ==========
all_items = np.load(all_items_path, allow_pickle=True)
drug_info = pd.read_csv(drug_info_path)
cell_info = pd.read_csv(cell_info_path)
comb_data = pd.read_csv(comb_data_path)

# ========== 建立映射 ==========
drug_name_to_smi = {k.lower(): v for k, v in zip(drug_info['dname'], drug_info['canonical_smiles'])}
smi_to_drugname = {v: k for k, v in drug_name_to_smi.items()}
depmap_to_cellname = {row['depmap_id']: row['cell_line_name'] for _, row in cell_info.iterrows()}

# ========== 过滤 all_items ==========
filtered_items = []
for item in all_items:
    drugA_smiles = item[0]
    drugB_smiles = item[1]
    depmap_id = item[2]
    
    # SMILES -> 药物名 -> 确认存在
    drugA_name = smi_to_drugname.get(drugA_smiles)
    drugB_name = smi_to_drugname.get(drugB_smiles)
    if drugA_name is None or drugB_name is None:
        continue
    
    # depmap_id -> cell_line_name
    cell_line_name = depmap_to_cellname.get(depmap_id)
    if cell_line_name is None:
        continue
    
    # 保存
    filtered_items.append([drugA_name, drugB_name, cell_line_name])

print(f"✅ 过滤后的 all_items 数量: {len(filtered_items)}")

# ========== 加入浓度和响应 ==========
# 构建组合查找
comb_lookup = {}
for _, row in comb_data.iterrows():
    key = (str(row['Drug1']).lower(), str(row['Drug2']).lower(), str(row['cell_line']).lower())
    comb_lookup[key] = (row['Concentration1'], row['Concentration2'], row['Response'])

new_items = []
for drugA_name, drugB_name, cell_line_name in filtered_items:
    key1 = (drugA_name.lower(), drugB_name.lower(), cell_line_name.lower())
    key2 = (drugB_name.lower(), drugA_name.lower(), cell_line_name.lower())
    
    added = False
    if key1 in comb_lookup:
        dose1, dose2, response = comb_lookup[key1]
        drugA_smiles = drug_name_to_smi[drugA_name.lower()]
        drugB_smiles = drug_name_to_smi[drugB_name.lower()]
        new_items.append([drugA_smiles, drugB_smiles, cell_line_name, dose1, dose2, response])
        added = True
    if key2 in comb_lookup:
        dose2, dose1, response = comb_lookup[key2]  # 交换剂量
        drugA_smiles = drug_name_to_smi[drugB_name.lower()]
        drugB_smiles = drug_name_to_smi[drugA_name.lower()]
        new_items.append([drugA_smiles, drugB_smiles, cell_line_name, dose1, dose2, response])
        added = True

print(f"✅ 最终生成的 all_items3 样本数: {len(new_items)}")

# ========== 记录被过滤原因 ==========
filtered_out = []

for item in all_items:
    drugA_smiles = item[0]
    drugB_smiles = item[1]
    depmap_id = item[2]

    reason = []

    # 检查药物名是否存在
    drugA_name = smi_to_drugname.get(drugA_smiles)
    drugB_name = smi_to_drugname.get(drugB_smiles)
    if drugA_name is None:
        reason.append("drugA_not_in_info")
    if drugB_name is None:
        reason.append("drugB_not_in_info")
    
    # 检查细胞系是否存在
    cell_line_name = depmap_to_cellname.get(depmap_id)
    if cell_line_name is None:
        reason.append("depmap_id_not_in_cell_info")

    # 如果任何一个字段缺失，就记录这条数据
    if reason:
        filtered_out.append({
            "drugA_smiles": drugA_smiles,
            "drugB_smiles": drugB_smiles,
            "depmap_id": depmap_id,
            "missing_reason": ";".join(reason)
        })

# 保存为 CSV
filtered_out_df = pd.DataFrame(filtered_out)
filtered_out_df.to_csv("/root/lanyun-tmp/Project/SynergyX/data/split/all_items_filtered_out.csv", index=False)
print(f"📄 被过滤记录已保存，共 {len(filtered_out_df)} 条")

# ========== 保存 ==========
new_items = np.array(new_items, dtype=object)
np.save(output_path, new_items)
print(f"✅ all_items3.npy 已保存到: {output_path}")
