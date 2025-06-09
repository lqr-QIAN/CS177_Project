import pandas as pd
import os

# 设置输入输出路径
input_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/drug_info.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/drug_info_extracted.csv'

def extract_drug_info():
    print("开始提取药物信息...")
    
    # 读取原始数据
    try:
        drug_info = pd.read_csv(input_path)
        print(f"✅ 成功读取原始数据，共 {len(drug_info)} 条记录")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 提取需要的列
    try:
        extracted_info = drug_info[['dname', 'canonical_smiles']].copy()
        print(f"✅ 成功提取药物名称和SMILES信息")
    except Exception as e:
        print(f"❌ 提取列失败: {e}")
        return
    
    # 保存到新文件
    try:
        extracted_info.to_csv(output_path, index=False)
        print(f"✅ 成功保存提取的数据到: {output_path}")
        print(f"✅ 保存的数据包含 {len(extracted_info)} 条记录")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return

if __name__ == "__main__":
    extract_drug_info() 
