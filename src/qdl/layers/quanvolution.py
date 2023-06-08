import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import rearrange, reduce
import torchquantum as tq
from omegaconf import OmegaConf

from qdl.layers.rx_cnot import RxCNOT

logger = logging.getLogger(__name__)

class Quanvolution(tq.QuantumModule):
    """
    Quanvolutional filter https://arxiv.org/abs/1904.04767

    For now it only supports a kernel size of 2x2.
    Uses BarrenLayer0 or Random layers as ansatz.
    """
    def __init__(self, cfg):
        super().__init__()
        self.kernel_size = 2 # Forced for now
        self.n_wires = self.kernel_size ** 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.kernel_size)
            ]
        )
        if cfg.quanvolution_type == "barren":
            self.q_layers = torch.nn.ModuleList([
                tq.layers.BarrenLayer0(
                    arch={
                        "n_wires": self.n_wires,
                        "n_blocks": cfg.n_blocks
                    }
                )
                for _ in range(cfg.out_channels)
            ])
        elif cfg.quanvolution_type == "random":
            self.q_layers = torch.nn.ModuleList([
                tq.layers.RandomLayer(n_ops=cfg.n_ops, wires=list(range(self.n_wires)))
                for _ in range(cfg.out_channels)
            ])
        elif cfg.quanvolution_type == "u3cu3":
            self.q_layers = torch.nn.ModuleList([
                tq.layers.U3CU3Layer0(
                    arch={
                        "n_wires": self.n_wires,
                        "n_blocks": cfg.n_blocks
                    }
                )
                for _ in range(cfg.out_channels)
            ])
        elif cfg.quanvolution_type == "rx-cnot":
            self.q_layers = torch.nn.ModuleList([
                RxCNOT(
                    cfg=OmegaConf.create({
                        "n_wires": self.n_wires,
                        "init_params": cfg.rx_cnot.init_params
                    })
                )
                for _ in range(cfg.out_channels)
            ])
        else:
            raise ValueError(f"Quanvolution type '{cfg.type}' is not in ['barren', 'random']")

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.stride = cfg.stride
        self.out_channels = cfg.out_channels
        self.in_proj = torch.nn.Linear(in_features=self.n_wires, out_features=self.n_wires, bias=False)

    def forward(self, x, use_qiskit=False):
        """
        x must have shape (b 1 h w), one channel for now
        """
        x = rearrange(x, "b 1 h w -> b h w")
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        channel_list = []
        for channel in range(self.out_channels):
            data_list = []
            for c in range(0, x.shape[1], self.stride):
                for r in range(0, x.shape[2], self.stride):
                    data = rearrange(
                        torch.cat(
                            (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                        ),
                        "(k b) -> b k",
                        k=self.n_wires
                    )
                    data = self.in_proj(data)
                    if use_qiskit:
                        data = self.qiskit_processor.process_parameterized(
                            qdev, self.encoder, self.q_layer, self.measure, data
                        )
                    else:
                        self.encoder(qdev, data)
                        self.q_layers[channel](qdev)
                        data = self.measure(qdev)

                    data_list.append(data)
            
            result_channel = rearrange(
                data_list,
                "(h w) b k -> b k h w",
                w=x.shape[1] // self.stride,
                h=x.shape[2] // self.stride,
                k=self.n_wires
            )
            channel_list.append(result_channel)

        result = rearrange(
            channel_list,
            "c b k h w -> b (c k) h w",
            w=x.shape[1] // self.stride,
            h=x.shape[2] // self.stride,
            c=self.out_channels,
            k=self.n_wires
        )

        return result