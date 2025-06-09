import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem

def canonicalize_smiles(smiles):
    """将SMILES转换为规范形式"""
    try:
        # 处理空值
        if pd.isna(smiles) or smiles == 'nan' or smiles == 'NaN':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return smiles
    except:
        return smiles

def match_drug_combinations():
    # 读取文件
    print("Reading files...")
    mapped_df = pd.read_csv('data/processed/drugcombs_response_mapped.csv', low_memory=False)
    all_items_df = pd.read_csv('data/split/all_items.csv', header=None)
    
    # 打印数据基本信息
    print(f"\nData information:")
    print(f"mapped_df shape: {mapped_df.shape}")
    print(f"all_items_df shape: {all_items_df.shape}")
    
    # 检查空值
    print("\nChecking for missing values:")
    print("\nMissing values in mapped_df:")
    print(mapped_df[['Drug1_SMILES', 'Drug2_SMILES', 'depmap_id']].isna().sum())
    print("\nMissing values in all_items_df:")
    print(all_items_df.isna().sum())
    
    # 检查SMILES格式
    print("\nChecking SMILES formats...")
    print("\nSample SMILES from mapped_df:")
    sample_mapped = mapped_df[['Drug1_SMILES', 'Drug2_SMILES']].head()
    print(sample_mapped)
    
    print("\nSample SMILES from all_items_df:")
    sample_all_items = all_items_df[[0, 1]].head()
    print(sample_all_items)
    
    # 检查是否需要规范化SMILES
    print("\nChecking if SMILES need canonicalization...")
    test_smiles = sample_all_items.iloc[0, 0]  # 取第一个SMILES作为测试
    if pd.notna(test_smiles) and test_smiles != 'nan':
        canonical_smiles = canonicalize_smiles(test_smiles)
        if test_smiles != canonical_smiles:
            print(f"SMILES needs canonicalization:")
            print(f"Original: {test_smiles}")
            print(f"Canonical: {canonical_smiles}")
        else:
            print("SMILES is already in canonical form")
    else:
        print("Test SMILES is empty or invalid")
    
    print("\nCreating lookup dictionary...")
    # 创建查找字典，使用元组(smiles1, smiles2, depmap_id)作为键
    lookup_dict = defaultdict(list)
    for _, row in tqdm(mapped_df.iterrows(), total=len(mapped_df), desc="Building index"):
        # 规范化SMILES
        smiles1 = canonicalize_smiles(str(row['Drug1_SMILES']))
        smiles2 = canonicalize_smiles(str(row['Drug2_SMILES']))
        depmap_id = str(row['depmap_id'])
        
        # 跳过无效的SMILES
        if smiles1 is None or smiles2 is None:
            continue
            
        key1 = (smiles1, smiles2, depmap_id)
        key2 = (smiles2, smiles1, depmap_id)
        lookup_dict[key1].append(row.to_dict())
        lookup_dict[key2].append(row.to_dict())
    
    print(f"\nNumber of unique combinations in lookup_dict: {len(lookup_dict)}")
    
    # 创建结果列表
    matched_rows = []
    skipped_rows = 0
    
    # 对all_items.csv中的每一行进行处理
    print("\nProcessing combinations...")
    for idx, row in tqdm(all_items_df.iterrows(), total=len(all_items_df), desc="Matching combinations"):
        # 规范化SMILES
        smiles1 = canonicalize_smiles(str(row[0]))
        smiles2 = canonicalize_smiles(str(row[1]))
        depmap_id = str(row[2])
        
        # 跳过无效的SMILES
        if smiles1 is None or smiles2 is None:
            skipped_rows += 1
            continue
        
        # 使用字典查找匹配
        key = (smiles1, smiles2, depmap_id)
        if key in lookup_dict:
            matched_rows.extend(lookup_dict[key])
        
        # 打印前几个未匹配的组合
        if idx < 5 and key not in lookup_dict:
            print(f"\nUnmatched combination {idx}:")
            print(f"Original SMILES1: {row[0]}")
            print(f"Original SMILES2: {row[1]}")
            print(f"Canonical SMILES1: {smiles1}")
            print(f"Canonical SMILES2: {smiles2}")
            print(f"depmap_id: {depmap_id}")
    
    print(f"\nSkipped {skipped_rows} rows due to invalid SMILES")
    
    # 转换为DataFrame
    if matched_rows:
        result_df = pd.DataFrame(matched_rows)
        
        # 保存结果
        output_file = 'data/processed/matched_drug_combinations.csv'
        result_df.to_csv(output_file, index=False)
        print(f"\nFound {len(result_df)} matching combinations")
        print(f"Results saved to {output_file}")
        
        # 打印一些统计信息
        print("\nStatistics:")
        print(f"Total unique drug combinations: {len(result_df[['Drug1_SMILES', 'Drug2_SMILES']].drop_duplicates())}")
        print(f"Total unique cell lines: {result_df['depmap_id'].nunique()}")
    else:
        print("\nNo matching combinations found")
        print("Please check if the data formats match between the two files")

if __name__ == '__main__':
    match_drug_combinations()
