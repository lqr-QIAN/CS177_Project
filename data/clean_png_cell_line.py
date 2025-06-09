import os
from tqdm import tqdm

# 主目录路径
base_dir = '/root/lanyun-tmp/Project/SynergyX/cell_line_items'

# 获取所有细胞系目录
cell_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"📊 找到 {len(cell_dirs)} 个细胞系目录")

# 计数器
total_deleted = 0

# 遍历所有子目录
for cell_name in tqdm(cell_dirs, desc="清理PNG文件", ncols=100):
    cell_folder = os.path.join(base_dir, cell_name)
    
    # 遍历目录中的所有文件
    for file_name in os.listdir(cell_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(cell_folder, file_name)
            try:
                os.remove(file_path)
                total_deleted += 1
            except Exception as e:
                tqdm.write(f"❌ 删除 {file_path} 失败：{e}")

print(f"\n✅ 清理完成！共删除 {total_deleted} 个PNG文件") 
