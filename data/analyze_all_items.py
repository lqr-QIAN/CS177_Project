import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def analyze_smiles_standardization():
    print("开始分析SMILES的标准化情况...")
    
    try:
        # 加载数据
        print("加载数据...")
        data = np.load('/root/lanyun-tmp/Project/SynergyX/data/split/all_items.npy', allow_pickle=True)
        
        # 打印数据的基本信息
        print("\n数据基本信息:")
        print(f"- 数据形状: {data.shape}")
        print(f"- 数据类型: {data.dtype}")
        
        # 提取SMILES数据
        drug1_smiles = data[:, 0]
        drug2_smiles = data[:, 1]
        
        # 分析第一个药物的SMILES
        print("\n分析第一个药物的SMILES:")
        unique_drug1 = np.unique(drug1_smiles)
        print(f"- 唯一SMILES数量: {len(unique_drug1)}")
        
        # 检查SMILES有效性
        valid_drug1 = []
        invalid_drug1 = []
        for smiles in unique_drug1:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_drug1.append(smiles)
            else:
                invalid_drug1.append(smiles)
        
        print(f"- 有效SMILES数量: {len(valid_drug1)}")
        print(f"- 无效SMILES数量: {len(invalid_drug1)}")
        
        if invalid_drug1:
            print("\n无效SMILES示例:")
            for smiles in invalid_drug1[:5]:
                print(f"- {smiles}")
        
        # 检查标准化情况
        if valid_drug1:
            print("\n检查标准化情况...")
            standardized_smiles = set()
            non_standardized = []
            
            for smiles in valid_drug1[:1000]:  # 只检查前1000个
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 标准化SMILES
                    std_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if std_smiles != smiles:
                        non_standardized.append((smiles, std_smiles))
                    standardized_smiles.add(std_smiles)
            
            print(f"- 检查的SMILES数量: {min(1000, len(valid_drug1))}")
            print(f"- 标准化后唯一SMILES数量: {len(standardized_smiles)}")
            print(f"- 需要标准化的SMILES数量: {len(non_standardized)}")
            
            if non_standardized:
                print("\n需要标准化的SMILES示例:")
                for orig, std in non_standardized[:5]:
                    print(f"- 原始: {orig}")
                    print(f"  标准化后: {std}")
                    print()
        
        # 分析第二个药物的SMILES
        print("\n分析第二个药物的SMILES:")
        unique_drug2 = np.unique(drug2_smiles)
        print(f"- 唯一SMILES数量: {len(unique_drug2)}")
        
        # 检查SMILES有效性
        valid_drug2 = []
        invalid_drug2 = []
        for smiles in unique_drug2:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_drug2.append(smiles)
            else:
                invalid_drug2.append(smiles)
        
        print(f"- 有效SMILES数量: {len(valid_drug2)}")
        print(f"- 无效SMILES数量: {len(invalid_drug2)}")
        
        if invalid_drug2:
            print("\n无效SMILES示例:")
            for smiles in invalid_drug2[:5]:
                print(f"- {smiles}")
        
        # 检查标准化情况
        if valid_drug2:
            print("\n检查标准化情况...")
            standardized_smiles = set()
            non_standardized = []
            
            for smiles in valid_drug2[:1000]:  # 只检查前1000个
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 标准化SMILES
                    std_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if std_smiles != smiles:
                        non_standardized.append((smiles, std_smiles))
                    standardized_smiles.add(std_smiles)
            
            print(f"- 检查的SMILES数量: {min(1000, len(valid_drug2))}")
            print(f"- 标准化后唯一SMILES数量: {len(standardized_smiles)}")
            print(f"- 需要标准化的SMILES数量: {len(non_standardized)}")
            
            if non_standardized:
                print("\n需要标准化的SMILES示例:")
                for orig, std in non_standardized[:5]:
                    print(f"- 原始: {orig}")
                    print(f"  标准化后: {std}")
                    print()
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        print(traceback.format_exc())
        return

if __name__ == "__main__":
    analyze_smiles_standardization()
