import torch.nn as nn
from typing import Optional
import torch
from typing import Literal, Dict
import json
import os
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
        x_rnn, hn=self.gru(x)
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
    def __init__(self, pre_trained_model_config: Dict[str, int],output_dim):
        super(ConvNet, self).__init__()
        self.n_filters = pre_trained_model_config["n_filters"]
        self.embed_dim = pre_trained_model_config["embed_dim"]
        self.kernel_size = pre_trained_model_config["kernel_size"]
        self.feedforward_dim = pre_trained_model_config["feedforward_dim"]
        self.lconv_network = nn.Sequential(
            nn.Conv1d(4, 3 * self.n_filters, kernel_size=11, padding=5),
            nn.Conv1d(
                3 * self.n_filters, 3 * self.n_filters, kernel_size=11, padding=5
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(
                3 * self.n_filters,
                4 * self.n_filters,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(4 * self.n_filters),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(4 * self.n_filters, 4 * self.n_filters, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(
                4 * self.n_filters,
                5 * self.n_filters,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(5 * self.n_filters),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(5 * self.n_filters, 5 * self.n_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                5 * self.n_filters,
                6 * self.n_filters,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(6 * self.n_filters),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(6 * self.n_filters, 6 * self.n_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.dconv1 = nn.Sequential(
            nn.Conv1d(
                6 * self.n_filters,
                4 * self.n_filters,
                kernel_size=5,
                dilation=2,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(4 * self.n_filters),
            nn.GELU(),
            nn.Conv1d(
                4 * self.n_filters,
                6 * self.n_filters,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(6 * self.n_filters),
            nn.Dropout(p=0.1),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv1d(
                6 * self.n_filters,
                4 * self.n_filters,
                kernel_size=5,
                dilation=4,
                padding=8,
                bias=False,
            ),
            nn.BatchNorm1d(4 * self.n_filters),
            nn.GELU(),
            nn.Conv1d(
                4 * self.n_filters,
                6 * self.n_filters,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(6 * self.n_filters),
            nn.Dropout(p=0.2),
        )

        self.gelu = nn.GELU()
        self.head = finetuneblock(
            input_channels=6 * self.n_filters,
            embed_dim=self.embed_dim,
            feedforward_dim=self.feedforward_dim,
            output_dim=output_dim,
        )

    def forward(self, input: torch.Tensor):
        output1 = self.lconv_network(input)
        output2 = self.dconv1(output1)
        output3 = self.dconv2(self.gelu(output2 + output1))
        output = self.head(self.gelu(output2 + output3))
        return output



def build_ConvNet(args,output_dim):
    assert args.config_path!=None

    if not os.path.exists(args.config_path):
        args.config_path=os.path.join(args.model_cache_dir, args.config_path)

    with open(args.config_path, 'r') as file:
        config= json.load(file)

    net = ConvNet(config['model_architecture'],output_dim)
    if args.pretrained_model_path != None:
        if not os.path.exists(args.pretrained_model_path):
            args.pretrained_model_path=os.path.join(args.model_cache_dir, args.pretrained_model_path)
        print("="*10+"Loading pretrained checkpoint"+"="*10)
        net.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device("cpu")),strict=False)
    return net
