import pandas as pd
import os

# 设置文件路径
input_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/cell_info.csv'
output_path = '/root/lanyun-tmp/Project/SynergyX/data/raw_data/cell_info_extracted.csv'

def extract_cell_info():
    print("开始提取细胞信息...")
    
    try:
        # 读取原始数据
        print("读取 cell_info.csv...")
        df = pd.read_csv(input_path)
        print(f"✅ 成功读取，共 {len(df)} 条记录")
        
        # 提取指定列
        print("\n提取 depmap_id 和 cell_line_name 列...")
        extracted_df = df[['depmap_id', 'cell_line_name']].copy()
        
        # 检查是否有空值
        null_counts = extracted_df.isnull().sum()
        if null_counts.any():
            print("\n⚠️ 发现空值:")
            print(null_counts[null_counts > 0])
            print("\n删除包含空值的行...")
            original_len = len(extracted_df)
            extracted_df = extracted_df.dropna()
            print(f"- 原始行数: {original_len}")
            print(f"- 删除空值后行数: {len(extracted_df)}")
            print(f"- 删除的行数: {original_len - len(extracted_df)}")
        
        # 保存结果
        extracted_df.to_csv(output_path, index=False)
        print(f"\n✅ 成功保存提取的数据到: {output_path}")
        
        # 打印统计信息
        print("\n数据统计:")
        print(f"- 总记录数: {len(extracted_df)}")
        print(f"- 唯一 depmap_id 数: {extracted_df['depmap_id'].nunique()}")
        print(f"- 唯一 cell_line_name 数: {extracted_df['cell_line_name'].nunique()}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return

if __name__ == "__main__":
    extract_cell_info() 
