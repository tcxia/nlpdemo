import torch
import torch.nn as nn

from load_data import START_TAG, STOP_TAG


device = "cuda" if torch.cuda.is_available() else "cpu"


class CharLSTM(nn.Module):
    def __init__(self, char_size, embed_dim, hidden_dim) -> None:
        super().__init__()
        self.char_size = char_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.char_embeds = nn.Embedding(self.char_size, self.embed_dim)

        self.wf = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.wi = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.wo = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.wc = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)

    def forward(self, x, h_pre, c_pre, words_cell_states=[]):
        
        '''
            x: [1, 1]
            h_pre: [1, hidden_dim]
            c_pre: [1, hidden_dim]
            words_cell_states: [[cell_state, weight], [...]]
        '''

        char_embedding = self.char_embeds(x) #[1, 1, embed_dim]
        char_embedding = char_embedding.squeeze(0) # [1, embed_dim]

        f = torch.sigmoid(self.wf(torch.cat([char_embedding, h_pre], dim=1))) # [1, hidden_dim]
        i = torch.sigmoid(self.wi(torch.cat([char_embedding, h_pre], dim=1))) # [1, hidden_dim]
        c_ = torch.tanh(self.wc(torch.cat([char_embedding, h_pre], dim=1))) # [1, hidden_dim]

        if not words_cell_states:
            c_cur = f * c_pre + i * c_ # [1, hidden_dim]

        else:
            cell_states = [c_]
            weights = [i]
            for cell_state, weight in words_cell_states:
                cell_states.append(cell_state)
                weights.append(weight)

            weights = torch.cat(weights, dim=0)
            weights = torch.softmax(weights, dim=0)
            cell_states = torch.cat(cell_states, dim=0)

            c_cur = torch.sum(weights * cell_states, dim=0).unsqueeze(0) # [1, hidden_dim]

        o = torch.sigmoid(self.wo(torch.cat([char_embedding, h_pre], dim=1))) # [1, hidden_dim]
        h_cur = o * torch.tanh(c_cur) # [1, hidden_dim]
        return h_cur, c_cur


class WordLSTM(nn.Module):
    def __init__(self, word_size, embed_dim, hidden_dim) -> None:
        super().__init__()
        self.word_size = word_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(self.word_size, self.embed_dim)

        self.w_f = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.w_i = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.w_c = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
        self.w_l = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)

    def forward(self, x, h, c):
        word_embedding = self.word_embeds(x)
        word_embedding = word_embedding.squeeze(0)

        i = torch.sigmoid(self.w_i(torch.cat([word_embedding, h], dim=1)))
        f = torch.sigmoid(self.w_f(torch.cat([word_embedding, h], dim=1)))
        c_ = torch.tanh(self.w_c(torch.cat([word_embedding, h], dim=1)))

        word_cell_state = f * c + i * c_

        return word_cell_state

    def get_weight(self, x_embed, c):
        word_weight = torch.sigmoid(self.w_l(torch.cat([x_embed, c], dim=1)))
        return word_weight

def log_sum_exp(smat):
    vmax = smat.max(dim=0, keepdim=True).values
    return (smat - vmax).exp().sum(axis=0, keepdim=True).log() + vmax



class LatticeLSTM(nn.Module):
    def __init__(self, char_size, word_size, label2idx, embed_dim, hidden_dim) -> None:
        super().__init__()

        self.label2idx = label2idx
        self.hidden_dim = hidden_dim
        self.label_size = len(label2idx)

        self.hidden2label = nn.Linear(hidden_dim, self.label_size)

        self.charlstm = CharLSTM(char_size, embed_dim, hidden_dim)
        self.wordlstm = WordLSTM(word_size, embed_dim, hidden_dim)

        self.transition = nn.Parameter(torch.randn(self.label_size, self.label_size))
        self.transition.data[label2idx[START_TAG], :] = -10000
        self.transition.data[:, label2idx[STOP_TAG]] = -10000

    def init_hidden(self):
        return (
            torch.randn(1, self.hidden_dim).to(device),
            torch.randn(1, self.hidden_dim).to(device),
        )

    def get_lstm_features(self, input_ids, input_words):
        char_h, char_c = self.init_hidden()

        length = len(input_ids)
        words_cell_states = [[]] * length

        hidden_states = []
        for idx, charid in enumerate(input_ids):
            charid = torch.tensor([[charid]]).to(device)
            char_h, char_c = self.charlstm(charid, char_h, char_c, words_cell_states[idx])
            hidden_states.append(char_h)

            if input_words[idx]:
                for word_id, word_length in input_words[idx]:
                    word_id = torch.tensor([[word_id]]).to(device)
                    word_cell_state = self.wordlstm(word_id, char_h, char_c)

                    end_char_id = input_ids[idx + word_length - 1]
                    end_char_id = torch.tensor([[end_char_id]]).to(device)
                    end_char_embed = self.charlstm.char_embeds(end_char_id).squeeze(0)
                    word_weight = self.wordlstm.get_weight(end_char_embed, word_cell_state)

                    words_cell_states[idx + word_length - 1].append([word_cell_state, word_weight])

        hidden_states = torch.cat(hidden_states, dim=0)
        lstm_feats = self.hidden2label(hidden_states)

        return lstm_feats


    def get_golden_score(self, lstm_feats, labels_idx):
        labels_idx.insert(0, self.label2idx[START_TAG])
        labels_tensor = torch.tensor(labels_idx).to(device)
        score = torch.zeros(1).to(device)

        for i, frame in enumerate(lstm_feats):
            score += self.transition[labels_tensor[i+1], labels_tensor[i]] + frame[labels_tensor[i+1]]

        return score + self.transition[self.label2idx[STOP_TAG], labels_tensor[-1]]

    def get_total_score(self, lstm_feats):
        alpha = torch.full((1, self.label_size), -10000.0).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0
        for frame in lstm_feats:
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transition.T)
        return log_sum_exp(alpha.T + 0 + self.transition[[self.label2idx[STOP_TAG]], :].T).flatten()


    def viterbi_decode(self, lstm_feats):
        backtrace = []
        alpha = torch.full((1, self.label_size), -10000.).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0
        for frame in lstm_feats:
            smat = alpha.T + frame.unsqueeze(0) + self.transition.T

            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)

        smat = alpha.T + 0 + self.transition[[self.label2idx[STOP_TAG]],:].T

        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1]

    def neg_log_likelihood(self, input_ids, input_words, labels_idx):
        lstm_feats = self.get_lstm_features(input_ids, input_words)
        total_score = self.get_total_score(lstm_feats)
        gold_score = self.get_golden_score(lstm_feats, labels_idx)
        return total_score - gold_score

    def forward(self, input_ids, input_words):
        lstm_feats = self.get_lstm_features(input_ids, input_words)
        result = self.viterbi_decode(lstm_feats)
        return result;