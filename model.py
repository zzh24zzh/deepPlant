import torch.nn as nn
from typing import Optional
import torch
from typing import Literal
# from src.seed import set_seed

# set_seed()


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))  # 960*960

    def forward(self, x: torch.Tensor):
        attn_logits = torch.einsum(
            "b n d, d e -> b n e", x, self.to_attn_logits
        )  # 64*16*960 , 960*960
        attn = attn_logits.softmax(dim=-2)  # 64*1*16*960 => 64*1*16*960
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class finetuneblock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        feedforward_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.project = nn.Conv1d(input_channels, embed_dim, kernel_size=1)

        self.gru=nn.GRU(embed_dim, embed_dim//2,
                        batch_first=True,
                        bidirectional=True,
                        num_layers=2,
                        dropout=0.2,
                        )
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(feedforward_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.prediction_head = nn.Linear(embed_dim, output_dim)


    def forward(self, input,TSSbinIdx=None,pool: Literal['mean','bin'] ='bin'):
        assert pool in ['mean','bin']
        x = self.project(input)
        x = x.permute(0, 2, 1)
        x_rnn, (h_n,c_n)=self.gru(x)
        x_rnn = self.linear(x_rnn)
        x=x+x_rnn
        if TSSbinIdx is None:
            TSSbinIdx = x_rnn.shape[1] // 2+1
        if pool=='mean':
            x = self.prediction_head(x.mean(1))
        else:
            x = self.prediction_head(x[:,TSSbinIdx,:])
        return x


class ConvNet(nn.Module):
    def __init__(self, n_features: int):
        super(ConvNet, self).__init__()
        self.n_features = n_features
        self.lconv_network = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=11, padding=5),
            nn.Conv1d(480, 480, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(480, 640, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(640),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(640, 640, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(640, 720, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(720),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(720, 720, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(720, 960, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(960),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.dconv1 = nn.Sequential(
            nn.Conv1d(960, 480, kernel_size=5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(480),
            nn.GELU(),
            nn.Conv1d(480, 960, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv1d(960, 480, kernel_size=5, dilation=4, padding=8, bias=False),
            nn.BatchNorm1d(480),
            nn.GELU(),
            nn.Conv1d(480, 960, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )
        # self.dconv3 = nn.Sequential(
        #     nn.Conv1d(960, 480, kernel_size=5, dilation=8, padding=16, bias=False),
        #     nn.BatchNorm1d(480),
        #     nn.GELU(),
        #     nn.Conv1d(480, 960, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm1d(960),
        #     nn.Dropout(p=0.2),
        # )
        # self.dconv4 = nn.Sequential(
        #     nn.Conv1d(960, 480, kernel_size=5, dilation=16, padding=32, bias=False),
        #     nn.BatchNorm1d(480),
        #     nn.GELU(),
        #     nn.Conv1d(480, 960, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm1d(960),
        #     nn.Dropout(p=0.2),
        # )
        self.gelu = nn.GELU()
        self.classifier = finetuneblock(
            input_channels=960,
            embed_dim=1280,
            feedforward_dim=2048,
            output_dim=self.n_features,
        )

    def forward(self, input: torch.Tensor):
        output1 = self.lconv_network(input)
        output2 = self.dconv1(output1)
        output3 = self.dconv2(self.gelu(output2 + output1))
        output = self.classifier(output3)

        # output4 = self.dconv3(self.gelu(output3 + output2))
        # output5 = self.dconv4(self.gelu(output4 + output3))
        # output = self.classifier(self.gelu(output5 + output4))
        return output


def build_ConvNet(
    n_features: int,
    model_path: Optional[str] = None,
):
    net = ConvNet(n_features=n_features)
    if model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return net
