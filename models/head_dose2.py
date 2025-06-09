import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, out_channels=256):
        super(FusionHead, self).__init__()

        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        dose_dim = 64 * 2
        fusion_input_dim = out_channels * 40 + dose_dim

        self.fc = nn.Sequential(
            nn.Linear(fusion_input_dim, out_channels * 20),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels * 20, out_channels * 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels * 10, 1)
        )

    def forward(self, x_cell_embed, drug_embed, doseA, doseB):
        doseA_embed = self.dose_encoder(doseA)
        doseB_embed = self.dose_encoder(doseB)
        dose_embed = torch.cat([doseA_embed, doseB_embed], dim=1)

        fusion_input = torch.cat([x_cell_embed, drug_embed, dose_embed], dim=1)
        output = self.fc(fusion_input)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
