import json
import pandas as pd
import os

def process_mapping_json():
    # 确保输出目录存在
    output_dir = 'data/raw_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取mapping.json文件
    with open('/root/lanyun-tmp/Project/SynergyX/data/processed/mappings.json', 'r') as f:
        mapping_data = json.load(f)
    
    # 提取drug_smiles_map数据
    drug_smiles_data = []
    for drug_name, smiles in mapping_data['drug_smiles_map'].items():
        drug_smiles_data.append({
            'dname': drug_name,
            'canonical_smiles': smiles
        })
    
    # 提取cell_depmap_map数据
    cell_depmap_data = []
    for depmap_id, cell_line_name in mapping_data['cell_depmap_map'].items():
        cell_depmap_data.append({
            'depmap_id': depmap_id,
            'cell_line_name': cell_line_name
        })
    
    # 创建DataFrame并保存为CSV
    drug_smiles_df = pd.DataFrame(drug_smiles_data)
    cell_depmap_df = pd.DataFrame(cell_depmap_data)
    
    # 保存为CSV文件到processed目录
    drug_smiles_df.to_csv(os.path.join(output_dir, 'drug_info_used.csv'), index=False)
    cell_depmap_df.to_csv(os.path.join(output_dir, 'cell_info_used.csv'), index=False)
    
    print("处理完成！")
    print(f"drug_info_used.csv 包含 {len(drug_smiles_df)} 条记录")
    print(f"cell_info_used.csv 包含 {len(cell_depmap_df)} 条记录")
    print(f"\n文件保存在: {output_dir}")

if __name__ == "__main__":
    process_mapping_json() 
