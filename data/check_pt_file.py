import torch

# 加载.pt文件
file_path = '/root/lanyun-tmp/Project/SynergyX/data/processed/0_fold_val_4079g_TransDrug_norm.pt'
data = torch.load(file_path)

# 打印数据类型和形状
print("数据类型:", type(data))
print("元组长度:", len(data))

# 遍历元组中的每个元素
for i, item in enumerate(data):
    print(f"\n元组第{i+1}个元素:")
    if isinstance(item, torch.Tensor):
        print("类型: torch.Tensor")
        print("形状:", item.shape)
        print("数据类型:", item.dtype)
        print("前5个元素:")
        print(item[:5])
    elif isinstance(item, dict):
        print("类型: dict")
        print("键:", list(item.keys()))
        for key, value in item.items():
            print(f"\n{key}:")
            if isinstance(value, torch.Tensor):
                print("形状:", value.shape)
                print("类型:", value.dtype)
                print("前5个元素:", value[:5])
            else:
                print("类型:", type(value))
                print("值:", value)
    else:
        print("类型:", type(item))
        print("值:", item) 
