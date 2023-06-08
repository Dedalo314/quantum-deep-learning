import logging

import torch
from einops import rearrange

from qdl.models.quantum.qlstm import QLSTM
from qdl.layers.quanvolution import Quanvolution

logger = logging.getLogger(__name__)

class CNNQLSTMClassifier(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if cfg.is_cnn_quantum:
            logger.info("Quantum CNN used")
            self.conv2d_1 = Quanvolution(cfg=cfg.quanv)
        else:
            logger.info("Classical CNN used")
            self.conv2d_1 = torch.nn.Conv2d(**cfg.conv2d)

        if cfg.is_lstm_quantum:
            logger.info("Quantum LSTM used")
            self.lstm = QLSTM(**cfg.lstm)
        else:
            logger.info("Classical LSTM used")
            self.lstm = torch.nn.LSTM(
                input_size=cfg.lstm.input_size,
                hidden_size=cfg.lstm.hidden_size,
                batch_first=cfg.lstm.batch_first
            )

        self.relu1 = torch.nn.ReLU()
        self.hidden2vocab = torch.nn.Linear(**cfg.last_linear_layer)

    def forward(self, x):
        x = rearrange(x, "b h w -> b 1 h w")
        conv1_out = self.conv2d_1(x)
        act1_out = self.relu1(conv1_out)
        pool1_out = torch.nn.functional.max_pool2d(
            act1_out,
            kernel_size=2
        )
        lstm_out, _ = self.lstm(
            rearrange(pool1_out, "b c h w -> b (h w) c")
        )
        lstm_out = lstm_out[:, -1, :]
        logits = self.hidden2vocab(lstm_out)
        
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )