from matplotlib.pyplot import axis
import torch
import torch.nn as nn

from load_data import START_TAG, STOP_TAG


device = "cuda" if torch.cuda.is_available() else "cpu"


def log_sum_exp(smat):
    vmax = smat.max(dim=1, keepdim=True).values
    return (smat - vmax).exp().sum(axis=1, keepdim=True).log() + vmax


class BiLSTM_CRF(nn.Module):
    def __init__(self, char_size, label2idx, embedding_dim, hidden_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_size = char_size
        self.label2idx = label2idx
        self.label_size = len(label2idx)

        self.char_embeds = nn.Embedding(char_size, embedding_dim)

        # 定义双向的lstm
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.hidden2label = nn.Linear(hidden_dim, self.label_size)

        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))
        self.transitions.data[label2idx[START_TAG], :] = -10000
        self.transitions.data[:, label2idx[STOP_TAG]] = -10000

    def init_hidden(self, bs):
        return (
            torch.randn(2, bs, self.hidden_dim // 2).to(device),
            torch.randn(2, bs, self.hidden_dim // 2).to(device),
        )

    def get_lstm_features(self, x):
        # x: [batch_size, seq_len]
        hidden = self.init_hidden(len(x))
        embeds = self.char_embeds(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, hidden_out = self.lstm(
            embeds, hidden
        )  # lstm_out: [batch_size, seq_len, hidden_dim]
        lstm_feats = self.hidden2label(lstm_out)  # [batch_size, seq_len, label_size]
        return lstm_feats

    def get_total_scores(self, frames, real_lengths):
        """
        得到所有可能路径的总和
        """
        # frames: [batch_size, seq_len, label_size]
        # real_lengths: [batch_size]
        alpha = torch.full((frames.shape[0], self.label_size), -10000.0).to(
            device
        )  # [batch_size, label_size]
        alpha[:, self.label2idx[START_TAG]] = 0.0
        alpha_ = torch.zeros((frames.shape[0], self.label_size)).to(device)
        frames = frames.transpose(0, 1)
        index = 0

        for frame in frames:
            index += 1
            alpha = log_sum_exp(
                alpha.unsqueeze(-1) + frame.unsqueeze(1) + self.transitions.T
            ).squeeze(1)

            for idx, length in enumerate(real_lengths):
                if length == index:
                    alpha_[idx] = alpha[idx]

        alpha_ = log_sum_exp(
            alpha_.unsqueeze(-1) + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T
        ).flatten()

        return alpha_

    def get_gloden_scores(self, frames, labels_idx_batch, real_lengths):
        score = torch.zeros(labels_idx_batch.shape[0]).to(device)
        score_ = torch.zeros(labels_idx_batch.shape[0]).to(device)
        labels = torch.cat(
            [
                torch.full(
                    [labels_idx_batch.shape[0], 1],
                    self.label2idx[START_TAG],
                    dtype=torch.long,
                ).to(device),
                labels_idx_batch,
            ],
            dim=1,
        )
        index = 0

        for i in range(frames.shape[1]):
            index += 1
            frame = frames[:, i, :]
            score += (
                self.transitions[labels[:, i + 1], labels[:, i]]
                + frame[range(frame.shape[0]), labels[:, i + 1]]
            )

            for idx, length in enumerate(real_lengths):
                if length == index:
                    score_[idx] = score[idx]

        score_ = score_ + self.transitions[self.label2idx[STOP_TAG], labels[:, -1]]
        return score_

    def viterbi_decode(self, frames):
        backtrace = []
        alpha = torch.fill((1, self.label_size), -10000.0).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0
        for frame in frames:
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)

        smat = alpha.T + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1]

    def neg_log_likelihood(self, inputs_idx_batch, labels_idx_batch, real_lengths):
        feats = self.get_lstm_features(inputs_idx_batch)
        total_scores = self.get_total_scores(feats, real_lengths)
        gold_score = self.get_gloden_scores(feats, labels_idx_batch, real_lengths)
        return torch.mean(total_scores - gold_score)

    def forward(self, inputs_idx_batch):
        lstm_feats = self.get_lstm_features(inputs_idx_batch)
        lstm_feats = lstm_feats.squeeze(0)
        result = self.viterbi_decode(lstm_feats)
        return result
