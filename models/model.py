import torch
import torch.nn as nn
from .head_dose import FusionHead
from .model_utlis import Embeddings, Encoder, EncoderCA, EncoderCell, EncoderCellCA, EncoderD2C, EncoderSSA, EncoderCellSSA
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .scgpt_encoder import ScGPTEncoder



class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]

        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]         

        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)


    def forward(self, x):

        # print('x_cell_embed.shape:',x_cell_embed.shape)
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  # [batch, out_channel, 53]
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) # [batch,53,64] or [batch,53,128]
        
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


def drug_feat(drug_subs_codes, device, patch, length):
    v = drug_subs_codes
    subs = v[:, 0].long().to(device)
    subs_mask = v[:, 1].long().to(device)

    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), 1)
        subs_mask = torch.cat((subs_mask, padding), 1)

    expanded_subs_mask = subs_mask.unsqueeze(1).unsqueeze(2)
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()



class GeneformerEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(GeneformerEncoder, self).__init__()
        # 加载预训练模型
        self.geneformer = AutoModel.from_pretrained("/root/lanyun-tmp/Project/SynergyX/models/Geneformer")
        # 冻结预训练模型参数
        for param in self.geneformer.parameters():
            param.requires_grad = False
            
        # 添加输入投影层，将输入特征维度转换为模型期望的维度
        self.input_projection = nn.Linear(6, self.geneformer.config.hidden_size)
        
        # 添加输出转换层
        self.transform = nn.Linear(self.geneformer.config.hidden_size, hidden_size)
        
    def forward(self, gene_data):
        # gene_data shape: [batch_size, genes_nums, feature_dim]
        batch_size, seq_length, feature_dim = gene_data.size()
        
        # 将数据转换为 Geneformer 期望的格式
        gene_data = gene_data.view(batch_size, seq_length, feature_dim)
        
        # 投影输入特征到模型期望的维度
        gene_data = self.input_projection(gene_data)
        
        # 添加位置编码
        position_ids = torch.arange(seq_length, device=gene_data.device).unsqueeze(0).expand(batch_size, -1)
        
        # 创建注意力掩码
        attention_mask = torch.ones((batch_size, seq_length), device=gene_data.device)
        
        # 使用预训练模型提取特征
        with torch.no_grad():
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 分批处理以减少内存使用
            chunk_size = 100  # 每次处理100个基因
            features_list = []
            
            for i in range(0, seq_length, chunk_size):
                end_idx = min(i + chunk_size, seq_length)
                chunk_data = gene_data[:, i:end_idx, :]
                chunk_positions = position_ids[:, i:end_idx]
                chunk_mask = attention_mask[:, i:end_idx]
                
                outputs = self.geneformer(
                        inputs_embeds=chunk_data,
                        position_ids=chunk_positions,
                        attention_mask=chunk_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                features_list.append(outputs.last_hidden_state)
                
                # 清理中间结果
            del outputs
            torch.cuda.empty_cache()
            
            # 合并所有特征
            features = torch.cat(features_list, dim=1)
            del features_list
            torch.cuda.empty_cache()
        
        # 转换到所需维度
        features = self.transform(features)
        return features


class SynergyxNet(torch.nn.Module):

    def __init__(self,
                 num_attention_heads = 8,
                 attention_probs_dropout_prob = 0.1, 
                 hidden_dropout_prob = 0.1,
                 max_length = 50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 args=None):
        super(SynergyxNet, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length
        
        # 根据数据集调整基因数量
        if args.celldataset == 0:
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        # 如果使用 Geneformer 或 scGPT，调整基因数量以减少显存使用
        if self.args.use_geneformer or self.args.use_scgpt:
            print(f"Using {'Geneformer' if self.args.use_geneformer else 'scGPT'} with {self.genes_nums} genes")

        # 初始化 hidden_size
        if self.args.cellencoder == 'cellTrans':
            self.patch = 50
            if self.in_channel == 3:
                feat_dim = 243
                hidden_size = 256
            elif self.in_channel == 6:
                feat_dim = 243*2
                hidden_size = 512
            self.cell_linear = nn.Linear(feat_dim, hidden_size)

        elif self.args.cellencoder == 'cellCNNTrans':
            self.patch = 165
            if self.in_channel == 3:
                hidden_size = 64
            elif self.in_channel == 6:
                hidden_size = 128 
            self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size, args=args)

        # 添加 Geneformer 编码器
        if self.args.use_geneformer:
            self.geneformer_encoder = GeneformerEncoder(hidden_size=self.args.geneformer_hidden_size)
            # 添加特征融合层
            self.cell_linear = nn.Linear(self.in_channel, self.args.geneformer_hidden_size)
            
        # 添加 scGPT 编码器
        if self.args.use_scgpt:
            self.scgpt_encoder = ScGPTEncoder(hidden_size=self.args.scgpt_hidden_size)
            # 添加特征融合层
            self.scgpt_linear = nn.Linear(self.in_channel, self.args.scgpt_hidden_size)
        
        intermediate_size = hidden_size*2
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)        
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SA = EncoderCell(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_CA = EncoderCA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_CA = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_cell_CA = EncoderD2C(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_SSA = EncoderSSA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SSA = EncoderCellSSA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)

        self.head = FusionHead()

        self.cell_fc = nn.Sequential(
                        nn.Linear(self.patch * hidden_size, output_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                    )

        self.drug_fc = nn.Sequential(
                        nn.Linear(self.patch * hidden_size, output_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                    )


    def forward(self, drugA, drugB, x_cell, doseA, doseB):
        batch_size = drugA.size(0)

        # 药物编码
        drugA, drugA_attention_mask = drug_feat(drugA, self.args.device, self.patch, self.max_length)
        drugB, drugB_attention_mask = drug_feat(drugB, self.args.device, self.patch, self.max_length)
        drugA = self.drug_emb(drugA).float()
        drugB = self.drug_emb(drugB).float()

        # 细胞编码
        x_cell = x_cell.type(torch.float32)
        x_cell = x_cell[:, [self.omic_dict[i] for i in self.include_omic]]
        
        if self.args.use_geneformer or self.args.use_scgpt:
            total_features = x_cell.size(0)
            features_per_sample = total_features // batch_size
            genes_per_sample = features_per_sample // self.in_channel
            expected_size = batch_size * genes_per_sample * self.in_channel
            if expected_size != total_features:
                genes_per_sample = total_features // (batch_size * self.in_channel)
            cellA = x_cell.view(batch_size, -1, self.in_channel)
            cellB = cellA.clone()
        else:
            cellA = x_cell.view(batch_size, self.genes_nums, -1)
            cellB = cellA.clone()

        # Geneformer / scGPT
        if self.args.use_geneformer:
            cellA = cellA.to(self.args.device)
            cellB = cellB.to(self.args.device)
            geneformer_features = self.geneformer_encoder(cellA)
            geneformer_features = geneformer_features.view(batch_size, -1, geneformer_features.size(-1))
            cellA_proj = self.cell_linear(cellA)
            cellB_proj = self.cell_linear(cellB)
            cellA = cellA_proj + geneformer_features
            cellB = cellB_proj + geneformer_features
        elif self.args.use_scgpt:
            cellA = cellA.to(self.args.device)
            cellB = cellB.to(self.args.device)
            scgpt_features = self.scgpt_encoder(cellA)
            scgpt_features = scgpt_features.view(batch_size, -1, scgpt_features.size(-1))
            cellA_proj = self.scgpt_linear(cellA)
            cellB_proj = self.scgpt_linear(cellB)
            cellA = cellA_proj + scgpt_features
            cellB = cellB_proj + scgpt_features
        else:
            cellA = x_cell.view(batch_size, self.genes_nums, -1)
            cellB = cellA.clone()

        # cellCNNTrans
        if self.args.cellencoder == 'cellTrans': 
            gene_length = 4050
            cellA = cellA[:, :gene_length, :]
            cellA = cellA.view(batch_size, self.patch, -1, x_cell.size(-1))
            cellA = cellA.view(batch_size, self.patch, -1)
            cellA = self.cell_linear(cellA)
            cellB = cellB[:, :gene_length, :]
            cellB = cellB.view(batch_size, self.patch, -1, x_cell.size(-1))
            cellB = cellB.view(batch_size, self.patch, -1)
            cellB = self.cell_linear(cellB)
        elif self.args.cellencoder == 'cellCNNTrans':
            if self.args.use_geneformer or self.args.use_scgpt:
                seq_length = cellA.size(1)
                if cellA.size(-1) != self.in_channel:
                    cellA = cellA.view(batch_size, seq_length, -1)
                    cellB = cellB.view(batch_size, seq_length, -1)
                    if not hasattr(self, 'dim_adjust'):
                        self.dim_adjust = nn.Linear(cellA.size(-1), self.in_channel).to(self.args.device)
                    cellA = self.dim_adjust(cellA)
                    cellB = self.dim_adjust(cellB)
            cellA = self.cell_conv(cellA)
            cellB = self.cell_conv(cellB)
        else:
            raise ValueError('Wrong cellencoder type!!!')

        # 多层交互
        cellA0 = cellA
        cellA, drugA, _, _ = self.drug_cell_CA(cellA, drugA, drugA_attention_mask, None)
        cellB, drugB, _, _ = self.drug_cell_CA(cellB, drugB, drugB_attention_mask, None)
        cellA, _ = self.cell_SA(cellA, None)
        cellB, _ = self.cell_SA(cellB, None)
        drugA, _ = self.drug_SA(drugA, drugA_attention_mask)
        drugB, _ = self.drug_SA(drugB, drugB_attention_mask)
        cellA, cellB, _, _ = self.cell_CA(cellA, cellB, None, None)
        drugA, drugB, _, _ = self.drug_CA(drugA, drugB, drugA_attention_mask, drugB_attention_mask)

        # 拼接特征
        drugA_embed = self.drug_fc(drugA.view(batch_size, -1))
        drugB_embed = self.drug_fc(drugB.view(batch_size, -1))
        cellA_embed = self.cell_fc(cellA.view(batch_size, -1))
        cellB_embed = self.cell_fc(cellB.view(batch_size, -1))
        cell_embed = torch.cat((cellA_embed, cellB_embed), dim=1)
        drug_embed = torch.cat((drugA_embed, drugB_embed), dim=1)

        # 加入剂量（doseA, doseB）
        doseA = doseA.view(-1, 1)  # [batch_size, 1]
        doseB = doseB.view(-1, 1)  # [batch_size, 1]
        fusion_input = torch.cat([cell_embed, drug_embed, doseA, doseB], dim=1)  # 如果你想直接拼接剂量

        # 最终预测
        output = self.head(cell_embed, drug_embed, doseA, doseB)


        return output, cell_embed, None



    def init_weights(self):

        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        if self.args.cellencoder == 'cellCNNTrans':
            self.cell_conv.init_weights()
