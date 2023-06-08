import torch

from qdl.models.quantum.qlstm import QLSTM

class QLSTMGenerator(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.input_embeddings = torch.nn.Embedding(cfg.input.input_vocab_size, cfg.lstm.embedding_dim)

        if cfg.is_quantum:
            print("Tagger will use Quantum LSTM")
            self.lstm = QLSTM(cfg.lstm.embedding_dim, cfg.lstm.hidden_dim)
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = torch.nn.LSTM(cfg.lstm.embedding_dim, cfg.lstm.hidden_dim, batch_first=True)

        self.hidden2vocab = torch.nn.Linear(cfg.lstm.hidden_dim, cfg.last_linear_layer.output_vocab_size)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            embeds = self.input_embeddings(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2vocab(lstm_out)
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )