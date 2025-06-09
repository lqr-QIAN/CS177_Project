import torch
import torch.nn as nn
from transformers import AutoModel

class GeneformerEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(GeneformerEncoder, self).__init__()
        self.geneformer = AutoModel.from_pretrained("/root/lanyun-tmp/Project/SynergyX/models/Geneformer")
        for param in self.geneformer.parameters():
            param.requires_grad = False
        self.input_projection = nn.Linear(6, self.geneformer.config.hidden_size)
        self.transform = nn.Linear(self.geneformer.config.hidden_size, hidden_size)

    def forward(self, gene_data):
        batch_size, seq_length, feature_dim = gene_data.size()
        gene_data = self.input_projection(gene_data)
        position_ids = torch.arange(seq_length, device=gene_data.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones((batch_size, seq_length), device=gene_data.device)

        with torch.no_grad():
            chunk_size = 100
            features_list = []
            for i in range(0, seq_length, chunk_size):
                end_idx = min(i + chunk_size, seq_length)
                outputs = self.geneformer(
                    inputs_embeds=gene_data[:, i:end_idx, :],
                    position_ids=position_ids[:, i:end_idx],
                    attention_mask=attention_mask[:, i:end_idx],
                    output_hidden_states=True,
                    return_dict=True
                )
                features_list.append(outputs.last_hidden_state)

            features = torch.cat(features_list, dim=1)

        features = features.mean(dim=1)
        features = self.transform(features)
        return features

class SynergyxNet(nn.Module):
    def __init__(self, args):
        super(SynergyxNet, self).__init__()
        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp': 0, 'mut': 1, 'cn': 2, 'eff': 3, 'dep': 4, 'met': 5}
        self.in_channel = len(self.include_omic)

        self.genes_nums = {0: 697, 1: 18498, 2: 4079}[args.celldataset]
        self.patch = 50
        self.hidden_size = args.geneformer_hidden_size

        self.geneformer_encoder = GeneformerEncoder(hidden_size=self.hidden_size)
        self.cell_linear = None
        self.cell_fc = nn.Sequential(
            nn.Linear(self.patch * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)  # for regression output

    def forward(self, data):
        x_cell = data.x_cell.float()
        x_cell = x_cell[:, [self.omic_dict[i] for i in self.include_omic]]

        total_cells = x_cell.size(0)
        genes_per_sample = self.genes_nums
        in_channel = self.in_channel
        batch_size = total_cells // genes_per_sample

        assert total_cells % genes_per_sample == 0, "x_cell shape不合法，不能均分为每个样本的基因矩阵"
        cell_raw = x_cell.view(batch_size, genes_per_sample, in_channel)

        with torch.no_grad():
            geneformer_embedding = self.geneformer_encoder(cell_raw)
        geneformer_embedding = geneformer_embedding.unsqueeze(1).expand(-1, self.patch, -1)

        patch_input = cell_raw[:, :self.patch * self.in_channel, :].reshape(batch_size, self.patch, -1)
        input_dim = patch_input.size(-1)

        if self.cell_linear is None:
            self.cell_linear = nn.Linear(input_dim, self.hidden_size).to(patch_input.device)

        cell_trans_embedding = self.cell_linear(patch_input)

        cell_embedding = cell_trans_embedding + geneformer_embedding
        cell_flattened = cell_embedding.view(batch_size, -1)
        cell_output = self.cell_fc(cell_flattened)
        prediction = self.output_layer(cell_output)

        return prediction, None, None
