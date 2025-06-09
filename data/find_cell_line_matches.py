import requests
import pandas as pd
import json
from time import sleep
import re

def clean_cell_line_name(name):
    """清理细胞系名称，使其更容易匹配"""
    # 移除特殊字符
    name = re.sub(r'[^a-zA-Z0-9-]', '', name)
    # 转换为大写
    name = name.upper()
    return name

def query_cellosaurus(cell_line_name):
    """查询Cellosaurus数据库"""
    base_url = "https://api.cellosaurus.org/v1/cell-line"
    headers = {
        "Accept": "application/json"
    }
    
    try:
        # 清理细胞系名称
        clean_name = cell_line_name.replace('\\/', '/')
        response = requests.get(f"{base_url}/{clean_name}", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            # 提取关键信息
            result = {
                'accession': data.get('accession'),
                'name': data.get('name'),
                'synonyms': data.get('synonyms', []),
                'category': data.get('category'),
                'species': data.get('species'),
                'tissue': data.get('tissue'),
                'diseases': data.get('diseases', [])
            }
            return result
        else:
            print(f"Cellosaurus API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error querying Cellosaurus: {e}")
        return None

def query_ccle(cell_line_name):
    """查询CCLE数据库"""
    base_url = "https://portals.broadinstitute.org/ccle/api/v1/cell_lines"
    headers = {
        "Accept": "application/json"
    }
    
    try:
        # 清理细胞系名称
        clean_name = clean_cell_line_name(cell_line_name)
        response = requests.get(f"{base_url}?name={clean_name}", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"CCLE API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error querying CCLE: {e}")
        return None

def query_cosmic(cell_line_name):
    """查询COSMIC数据库"""
    base_url = "https://cancer.sanger.ac.uk/cosmic/cell_lines"
    headers = {
        "Accept": "application/json"
    }
    
    try:
        # 清理细胞系名称
        clean_name = cell_line_name.replace('\\/', '/')
        response = requests.get(f"{base_url}/{clean_name}", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"COSMIC API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error querying COSMIC: {e}")
        return None

def find_matches_in_cell_info(cell_line_name, cell_info_df):
    """在cell info文件中查找匹配"""
    # 清理细胞系名称
    clean_name = clean_cell_line_name(cell_line_name)
    
    # 1. 直接匹配
    direct_matches = cell_info_df[cell_info_df['cell_line_name'].apply(clean_cell_line_name) == clean_name]
    
    # 2. 部分匹配
    partial_matches = cell_info_df[cell_info_df['cell_line_name'].apply(clean_cell_line_name).str.contains(clean_name, case=False)]
    
    # 3. 移除连字符后匹配
    no_hyphen_name = clean_name.replace('-', '')
    no_hyphen_matches = cell_info_df[cell_info_df['cell_line_name'].apply(lambda x: clean_cell_line_name(x).replace('-', '')) == no_hyphen_name]
    
    return {
        'direct_matches': direct_matches.to_dict('records'),
        'partial_matches': partial_matches.to_dict('records'),
        'no_hyphen_matches': no_hyphen_matches.to_dict('records')
    }

def main():
    # 读取缺失的cell lines
    missing_df = pd.read_csv('data/processed/missing_celllines.csv')
    
    # 读取cell info数据
    cell_info_df = pd.read_csv('data/raw_data/merged_cell_info.csv')
    
    results = []
    for _, row in missing_df.iterrows():
        cell_line = row['cell_line']
        count = row['count']
        
        print(f"\nProcessing {cell_line}...")
        
        # 1. 查询外部数据库
        cellosaurus_info = query_cellosaurus(cell_line)
        sleep(1)  # 避免请求过快
        
        ccle_info = query_ccle(cell_line)
        sleep(1)
        
        cosmic_info = query_cosmic(cell_line)
        sleep(1)
        
        # 2. 在cell info中查找匹配
        cell_info_matches = find_matches_in_cell_info(cell_line, cell_info_df)
        
        # 保存结果
        result = {
            'cell_line': cell_line,
            'count': count,
            'cellosaurus_info': cellosaurus_info,
            'ccle_info': ccle_info,
            'cosmic_info': cosmic_info,
            'cell_info_matches': cell_info_matches
        }
        results.append(result)
        
        # 打印找到的信息
        print(f"\nResults for {cell_line}:")
        if cellosaurus_info:
            print("Cellosaurus info found")
            print(f"Accession: {cellosaurus_info.get('accession')}")
            print(f"Synonyms: {cellosaurus_info.get('synonyms')}")
        if ccle_info:
            print("CCLE info found")
        if cosmic_info:
            print("COSMIC info found")
        if cell_info_matches['direct_matches']:
            print("Direct matches in cell info found")
        if cell_info_matches['partial_matches']:
            print("Partial matches in cell info found")
    
    # 保存所有结果
    with open('data/processed/cell_line_detailed_info.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to data/processed/cell_line_detailed_info.json")

if __name__ == '__main__':
    main() 
