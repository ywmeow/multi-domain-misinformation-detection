"""One-layer BiLSTM with masked mean pooling for binary classification."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=128, hidden_size=128, dropout=0.3,
        pad_id=0, embedding_dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, 2)

    def forward(self, input_ids, lengths):
        embedded = self.embedding_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed, batch_first=True)
        lengths = lengths.to(output.device)
        mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]
        pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
        return self.classifier(self.dropout(pooled))
