import numpy as np
from sklearn.model_selection import train_test_split

def split_data():
    # 读取数据
    print("Loading data...")
    all_items_dose_all = np.load('data/split/all_items_dose_all.npy', allow_pickle=True)
    all_items_dose_matched = np.load('data/split/all_items_dose_matched.npy', allow_pickle=True)
    
    print(f"\nData shapes:")
    print(f"all_items_dose_all: {all_items_dose_all.shape}")
    print(f"all_items_dose_matched: {all_items_dose_matched.shape}")
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 分割 all_items_dose_all
    print("\nSplitting all_items_dose_all...")
    # 首先分割出训练集和临时集（80% 训练，20% 临时）
    tr_all, temp_all = train_test_split(all_items_dose_all, train_size=0.8, random_state=42)
    # 然后将临时集平均分为验证集和测试集（各占10%）
    val_all, test_all = train_test_split(temp_all, train_size=0.5, random_state=42)
    
    # 分割 all_items_dose_matched
    print("Splitting all_items_dose_matched...")
    # 首先分割出训练集和临时集（80% 训练，20% 临时）
    tr_matched, temp_matched = train_test_split(all_items_dose_matched, train_size=0.8, random_state=42)
    # 然后将临时集平均分为验证集和测试集（各占10%）
    val_matched, test_matched = train_test_split(temp_matched, train_size=0.5, random_state=42)
    
    # 保存分割后的数据
    print("\nSaving split data...")
    # 保存 all_items_dose_all 的分割结果
    np.save('data/split/0_fold_tr_items_dose_all.npy', tr_all)
    np.save('data/split/0_fold_val_items_dose_all.npy', val_all)
    np.save('data/split/0_fold_test_items_dose_all.npy', test_all)
    
    # 保存 all_items_dose_matched 的分割结果
    np.save('data/split/0_fold_tr_items_dose_matched.npy', tr_matched)
    np.save('data/split/0_fold_val_items_dose_matched.npy', val_matched)
    np.save('data/split/0_fold_test_items_dose_matched.npy', test_matched)
    
    # 打印分割结果信息
    print("\nSplit results:")
    print("\nFor all_items_dose_all:")
    print(f"Training set: {tr_all.shape}")
    print(f"Validation set: {val_all.shape}")
    print(f"Test set: {test_all.shape}")
    
    print("\nFor all_items_dose_matched:")
    print(f"Training set: {tr_matched.shape}")
    print(f"Validation set: {val_matched.shape}")
    print(f"Test set: {test_matched.shape}")
    
    # 验证数据完整性
    print("\nVerifying data integrity...")
    all_count = len(tr_all) + len(val_all) + len(test_all)
    matched_count = len(tr_matched) + len(val_matched) + len(test_matched)
    
    print(f"\nData integrity check:")
    print(f"all_items_dose_all: {all_count} == {len(all_items_dose_all)}")
    print(f"all_items_dose_matched: {matched_count} == {len(all_items_dose_matched)}")

if __name__ == '__main__':
    split_data() 
