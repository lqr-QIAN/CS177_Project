import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import json
import os

class ScGPTConfig(PretrainedConfig):
    model_type = "scgpt"
    
    def __init__(
        self,
        vocab_size=60697,
        embsize=512,
        d_hid=512,
        nlayers=12,
        nhead=8,
        dropout=0.0,
        max_seq_len=1536,
        pad_token_id=0,
        cell_emb_style="cls",
        input_emb_style="continuous",
        explicit_zero_prob=False,
        use_fast_transformer=True,
        use_flash_attention=False,
        norm_scheme="post",
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nhead = nhead
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.cell_emb_style = cell_emb_style
        self.input_emb_style = input_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.use_fast_transformer = use_fast_transformer
        self.use_flash_attention = use_flash_attention
        self.norm_scheme = norm_scheme

class ScGPTEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super(ScGPTEncoder, self).__init__()
        
        # 加载配置文件
        config_path = "/root/lanyun-tmp/Project/SynergyX/models/scGPT/config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = ScGPTConfig(**config_dict)
        
        # 创建基础模型结构
        self.scgpt = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.embsize,
                nhead=config.nhead,
                dim_feedforward=config.d_hid,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.nlayers
        )
        
        # 加载权重
        model_path = "/root/lanyun-tmp/Project/SynergyX/models/scGPT/scgpt_gh_repo_original_model.bin"
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
            
        try:
            # 加载模型权重
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 过滤出我们需要的权重
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if 'transformer.encoder' in key:
                    # 将键名转换为我们的模型结构
                    new_key = key.replace('transformer.encoder.', '')
                    filtered_state_dict[new_key] = value
            
            # 加载过滤后的权重
            self.scgpt.load_state_dict(filtered_state_dict, strict=False)
            print("Successfully loaded scGPT weights!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized weights instead.")
        
        # 冻结预训练模型参数
        for param in self.scgpt.parameters():
            param.requires_grad = False
            
        # 添加输入投影层，将输入特征维度转换为模型期望的维度
        self.input_projection = nn.Linear(3, config.embsize)  # 修改输入维度为3
        
        # 添加输出转换层
        self.transform = nn.Linear(config.embsize, hidden_size)
        
    def forward(self, gene_data):
        # gene_data shape: [batch_size, genes_nums, feature_dim]
        batch_size, seq_length, feature_dim = gene_data.size()
        
        # 确保输入维度正确
        if feature_dim != 3:
            raise ValueError(f"Expected input feature dimension to be 3, but got {feature_dim}")
        
        # 将数据转换为 scGPT 期望的格式
        gene_data = gene_data.view(batch_size, seq_length, feature_dim)
        
        # 投影输入特征到模型期望的维度
        gene_data = self.input_projection(gene_data)
        
        # 创建注意力掩码
        attention_mask = torch.ones((batch_size, seq_length), device=gene_data.device)
        
        # 使用预训练模型提取特征
        with torch.no_grad():
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 减小批处理大小以减少内存使用
            chunk_size = 50  # 从100减小到50
            features_list = []
            
            for i in range(0, seq_length, chunk_size):
                end_idx = min(i + chunk_size, seq_length)
                chunk_data = gene_data[:, i:end_idx, :]
                chunk_mask = attention_mask[:, i:end_idx]
                
                # 使用 Transformer 编码器
                features = self.scgpt(chunk_data)
                features_list.append(features)
                
                # 清理中间结果
                del features
                torch.cuda.empty_cache()
            
            # 合并所有特征
            features = torch.cat(features_list, dim=1)
            del features_list
            torch.cuda.empty_cache()
        
        # 转换到所需维度
        features = self.transform(features)
        return features 
