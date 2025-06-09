import os
import pandas as pd
from tqdm import tqdm

# 主目录路径
base_dir = '/root/lanyun-tmp/Project/SynergyX/cell_line_items'

# 获取所有细胞系目录
cell_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"📊 找到 {len(cell_dirs)} 个细胞系目录")

# 统计每个细胞系的数据条数
total_rows = 0
cell_stats = []

for cell_name in tqdm(cell_dirs, desc="统计数据", ncols=100):
    cell_folder = os.path.join(base_dir, cell_name)
    csv_path = os.path.join(cell_folder, f'{cell_name}_items.csv')
    
    if not os.path.exists(csv_path):
        tqdm.write(f"❌ 未找到 {cell_name}_items.csv，跳过")
        continue

    try:
        df = pd.read_csv(csv_path)
        rows = len(df)
        total_rows += rows
        cell_stats.append({
            'cell_line': cell_name,
            'rows': rows
        })
        tqdm.write(f"📈 {cell_name}: {rows} 条数据")
    except Exception as e:
        tqdm.write(f"❌ {cell_name} 处理失败：{e}")

# 按数据条数排序
cell_stats.sort(key=lambda x: x['rows'], reverse=True)

# 打印统计结果
print("\n📊 统计结果:")
print(f"总数据条数: {total_rows}")
print("\n前10个数据量最大的细胞系:")
for i, stat in enumerate(cell_stats[:10], 1):
    print(f"{i}. {stat['cell_line']}: {stat['rows']} 条数据")

print("\n前10个数据量最小的细胞系:")
for i, stat in enumerate(cell_stats[-10:], 1):
    print(f"{i}. {stat['cell_line']}: {stat['rows']} 条数据") 
