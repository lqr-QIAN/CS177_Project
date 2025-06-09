import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset
import psutil
from rdkit import Chem
from rdkit.Chem import MolStandardize
import json
import pandas as pd

def clean_and_canonicalize(smiles: str) -> str:
    """清洗并canonical化 SMILES 字符串"""
    smiles = smiles.strip().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return smiles

class MyInMemoryDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 dgi_data=None,
                 transform=None,
                 pre_transform=None,
                 args=None,
                 max_node_num=155,
                 force_reprocess=False,
                 include_dose=True):

        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)

        if args.celldataset == 1:
            self.name = osp.basename(data_items).split('items')[0] + '18498g'
        elif args.celldataset == 2:
            self.name = osp.basename(data_items).split('items')[0] + '4079g'
        elif args.celldataset == 3:
            self.name = osp.basename(data_items).split('items')[0] + '963g'

        self.name = self.name + '_TransDrug_norm'

        if args.mode == 'infer':
            self.name = osp.basename(data_items).split('items')[0]

        self.args = args
        self.data_items = np.load(data_items, allow_pickle=True)
        print(f"[Debug] Loaded {len(self.data_items)} items from {data_items}")

        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.drugs = np.load(drugs_data, allow_pickle=True).item()
        self.dgi = np.load(dgi_data, allow_pickle=True).item() if dgi_data else {}

        self.max_node_num = max_node_num
        self.include_dose = include_dose

        # 生成 depmap_to_cellname 映射文件（如果没有找到）
        self.depmap_to_cellname = {}
        if not os.path.exists('depmap_to_cellname.json'):
            print("Generating depmap_to_cellname.json from CSV...")
            # 读取 CSV 文件并生成映射字典
            cell_info_df = pd.read_csv('/root/lanyun-tmp/Project/SynergyX/cell_info_used.csv')
            self.depmap_to_cellname = dict(zip(cell_info_df['depmap_id'], cell_info_df['cell_line_name']))
            # 保存为 JSON 文件
            with open('depmap_to_cellname.json', 'w') as f:
                json.dump(self.depmap_to_cellname, f, indent=4)
        else:
            # 加载现有的映射文件
            with open('depmap_to_cellname.json') as f:
                self.depmap_to_cellname = json.load(f)

        # 生成 cellname_to_depmap 映射文件（如果没有找到）
        self.cellname_to_depmap = {}
        if not os.path.exists('cellname_to_depmap.json'):
            print("Generating cellname_to_depmap.json from CSV...")
            # 读取 CSV 文件并生成反向映射字典
            cell_info_df = pd.read_csv('/root/lanyun-tmp/Project/SynergyX/cell_info_used.csv')
            self.cellname_to_depmap = dict(zip(cell_info_df['cell_line_name'], cell_info_df['depmap_id']))
            # 保存为 JSON 文件
            with open('cellname_to_depmap.json', 'w') as f:
                json.dump(self.cellname_to_depmap, f, indent=4)
        else:
            # 加载现有的反向映射文件
            with open('cellname_to_depmap.json') as f:
                self.cellname_to_depmap = json.load(f)

        if force_reprocess or not os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data not found or force_reprocess=True. Doing pre-processing...')
            self.process()
        else:
            print(f'Pre-processed data found: {self.processed_paths[0]}, loading ...')

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def process(self):
        data_list = []
        data_len = len(self.data_items)
        print(f"Starting to process {data_len} items...")
        process = psutil.Process()

        failed_count = 0

        for i in tqdm(range(data_len)):
            if i % 10000 == 0:
                print(f"Processing item {i}, Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            try:
                item = self.data_items[i]
                
                # 以下所有的 print 语句已注释
                # print(f"Processing item {i}, Step: Extracting drugs and cell data")

                if self.include_dose:
                    if len(item) == 6:
                        drugA, drugB, c1, doseA, doseB, label = item
                    elif len(item) == 4:
                        drugA, drugB, c1, label = item
                        doseA, doseB = 1.0, 1.0
                    else:
                        raise ValueError(f"Unexpected data format at index {i}: {item}")
                else:
                    drugA, drugB, c1, label = item
                    doseA, doseB = 1.0, 1.0

                # 以下 print 语句已注释
                # print(f"Step: Cleaning and canonicalizing SMILES")
                drugA = clean_and_canonicalize(drugA)
                drugB = clean_and_canonicalize(drugB)

                # 以下 print 语句已注释
                # print(f"Step: Checking drug embeddings")
                if drugA not in self.drugs or drugB not in self.drugs:
                    raise ValueError(f"Missing drug embedding: {drugA} or {drugB}")

                c1 = str(c1).strip()

                # 清理depmap_id中的空格和不可见字符
                c1 = c1.strip()  # 删除前后的空格和换行符等不可见字符

                # 以下 print 语句已注释
                # print(f"Step: Checking if cell features are available")
                if c1 not in self.celllines:
                    # Debug print to check why it's missing
                    # print(f"Debug: self.celllines contains {len(self.celllines)} entries.")
                    # print(f"Debug: celllines keys: {list(self.celllines.keys())[:20]}...")  # 打印前20个key
                    if c1 in self.depmap_to_cellname:
                        # 使用 depmap_id 获取 cell_line_name
                        c1 = self.depmap_to_cellname[c1]
                    elif c1 in self.cellname_to_depmap:
                        # 使用 cell_line_name 获取 depmap_id
                        c1 = self.cellname_to_depmap[c1]
                    else:
                        #print(f"Missing cell features for item {i}: {c1}")
                        failed_count += 1
                        continue  # Skip this item, but log the failure

                # 以下 print 语句已注释
                # print(f"Step: Extracting cell features for {c1}")
                cell_features = self.celllines[c1]
                dgiA = self.dgi.get(drugA, np.ones(cell_features.shape[0]))
                dgiB = self.dgi.get(drugB, np.ones(cell_features.shape[0]))
                drugA_features = self.drugs[drugA]
                drugB_features = self.drugs[drugB]

                # 以下 print 语句已注释
                # print(f"Step: Creating data object")
                data = Data(
                    drugA=torch.tensor([drugA_features], dtype=torch.float32),
                    drugB=torch.tensor([drugB_features], dtype=torch.float32),
                    x_cell=torch.tensor(cell_features, dtype=torch.float32),
                    y=torch.tensor([float(label)], dtype=torch.float32),
                    dgiA=torch.tensor(dgiA, dtype=torch.float32),
                    dgiB=torch.tensor(dgiB, dtype=torch.float32),
                    doseA=torch.tensor([float(doseA)], dtype=torch.float32),
                    doseB=torch.tensor([float(doseB)], dtype=torch.float32)
                )
                data_list.append(data)

            except Exception as e:
                failed_count += 1
                #print(f"Error processing item {i}: {str(e)}")
                continue

        # 保留的输出：
        print(f"Successfully processed {len(data_list)} items out of {data_len}")
        #print(f"Failed to process {failed_count} items.")
        print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f'Graph construction done. {len(data_list)} samples processed. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Dataset construction done.')
