import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import requests
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import json

def get_smiles_from_pubchem(drug_name):
    """从PubChem获取SMILES"""
    try:
        # 首先搜索化合物
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
        response = requests.get(search_url)
        if response.status_code != 200:
            print(f"Failed to get CID for {drug_name}: {response.status_code}")
            return None
        
        data = response.json()
        if 'IdentifierList' not in data:
            print(f"No CID found for {drug_name}")
            return None
            
        cid = data['IdentifierList']['CID'][0]
        print(f"Found CID {cid} for {drug_name}")
        
        # 获取SMILES
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        response = requests.get(smiles_url)
        if response.status_code != 200:
            print(f"Failed to get SMILES for {drug_name}: {response.status_code}")
            return None
            
        data = response.json()
        if 'PropertyTable' not in data:
            print(f"No SMILES found for {drug_name}")
            return None
            
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        print(f"Found SMILES for {drug_name}: {smiles}")
        
        # 验证SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES for {drug_name}: {smiles}")
            return None
            
        return smiles
    except Exception as e:
        print(f"Error getting SMILES for {drug_name}: {str(e)}")
        return None

def load_cell_info():
    """加载cell line信息"""
    print("Loading cell line information...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    cell_info_path = osp.join(data_root, 'raw_data/merged_cell_info.csv')
    
    try:
        cell_info_df = pd.read_csv(cell_info_path)
        print(f"Loaded cell info with {len(cell_info_df)} entries")
        print("Columns in cell info:", cell_info_df.columns.tolist())
        return cell_info_df
    except Exception as e:
        print(f"Error loading cell info: {str(e)}")
        return None

def get_depmap_id(cell_line, cell_info_df):
    """从本地cell info文件获取depmap_id"""
    try:
        # 尝试不同的匹配方式
        matches = cell_info_df[
            (cell_info_df['cell_line_name'].str.lower() == cell_line.lower()) |
            (cell_info_df['cell_line_name'].str.replace('-', '').str.lower() == cell_line.replace('-', '').lower()) |
            (cell_info_df['cell_line_name'].str.upper() == cell_line.upper())
        ]
        
        if len(matches) > 0:
            depmap_id = matches.iloc[0]['depmap_id']
            print(f"Found depmap_id for {cell_line}: {depmap_id}")
            return depmap_id
        else:
            print(f"No depmap_id found for {cell_line}")
            return None
            
    except Exception as e:
        print(f"Error getting depmap_id for {cell_line}: {str(e)}")
        return None

def load_data():
    """加载response数据"""
    print("Loading data...")
    data_root = '/root/lanyun-tmp/Project/SynergyX/data'
    
    # 读取response数据
    response_path = osp.join(data_root, 'processed/drugcombs_response_with_cellline.csv')
    response_df = pd.read_csv(response_path)
    print(f"Response data shape: {response_df.shape}")
    
    return response_df

def process_data(response_df, cell_info_df):
    """处理数据，使用API获取SMILES和depmap_id信息"""
    print("\nProcessing data...")
    processed_df = response_df.copy()
    
    # 获取唯一的drug names
    unique_drugs = pd.concat([processed_df['Drug1'], processed_df['Drug2']]).unique()
    print(f"Found {len(unique_drugs)} unique drugs")
    
    # 创建drug name到SMILES的映射
    drug_smiles_map = {}
    for drug in tqdm(unique_drugs, desc="Getting SMILES"):
        if drug not in drug_smiles_map:
            smiles = get_smiles_from_pubchem(drug)
            drug_smiles_map[drug] = smiles
            time.sleep(0.5)  # 避免API限制
    
    # 获取唯一的cell lines
    unique_cell_lines = processed_df['cell_line'].unique()
    print(f"Found {len(unique_cell_lines)} unique cell lines")
    
    # 创建cell line到depmap_id的映射
    cell_depmap_map = {}
    for cell_line in tqdm(unique_cell_lines, desc="Getting depmap_ids"):
        if cell_line not in cell_depmap_map:
            depmap_id = get_depmap_id(cell_line, cell_info_df)
            cell_depmap_map[cell_line] = depmap_id
    
    # 添加SMILES信息
    print("Adding SMILES information...")
    processed_df['Drug1_SMILES'] = processed_df['Drug1'].map(drug_smiles_map)
    processed_df['Drug2_SMILES'] = processed_df['Drug2'].map(drug_smiles_map)
    
    # 添加depmap_id信息
    print("Adding depmap_id information...")
    processed_df['depmap_id'] = processed_df['cell_line'].map(cell_depmap_map)
    
    return processed_df, drug_smiles_map, cell_depmap_map

def save_processed_data(processed_df, drug_smiles_map, cell_depmap_map, output_path):
    """保存处理后的数据"""
    print("\nSaving processed data...")
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    # 保存映射关系
    mappings = {
        'drug_smiles_map': {k: v for k, v in drug_smiles_map.items() if v is not None},
        'cell_depmap_map': {k: v for k, v in cell_depmap_map.items() if v is not None}
    }
    with open(osp.join(os.path.dirname(output_path), 'mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=2)

def main():
    print("Starting data processing...")
    # 1. 加载数据
    response_df = load_data()
    cell_info_df = load_cell_info()
    
    if cell_info_df is None:
        print("Failed to load cell info, exiting...")
        return
    
    # 2. 处理数据
    processed_df, drug_smiles_map, cell_depmap_map = process_data(response_df, cell_info_df)
    
    # 3. 保存处理后的数据
    output_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/drugcombs_response_mapped.csv'
    save_processed_data(processed_df, drug_smiles_map, cell_depmap_map, output_path)
    
    # 4. 显示处理结果
    print("\nProcessing completed!")
    print(f"Original data rows: {len(response_df)}")
    print(f"Processed data rows: {len(processed_df)}")
    print(f"Number of unique drugs with SMILES: {processed_df['Drug1_SMILES'].notna().sum()}")
    print(f"Number of unique cell lines with depmap_id: {processed_df['depmap_id'].notna().sum()}")
    
    # 5. 显示处理后的数据样本
    print("\nSample of processed data (first 5 rows):")
    print(processed_df[['Drug1', 'Drug1_SMILES', 'Drug2', 'Drug2_SMILES', 'cell_line', 'depmap_id']].head().to_string())

if __name__ == '__main__':
    main() 
