import random

import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hid_dim, dec_hid_dim, dropout) -> None:
        super().__init__()
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.rnn = nn.GRU(embed_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1,:,:]), dim=1)))

        return outputs, hidden

# 注意力权重生成
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim) -> None:
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hid_dim, dec_hid_dim, dropout, attention) -> None:
        super().__init__()
        self.output_dim = output_dim
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embed_dim)
        # self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.rnn = nn.GRU((enc_hid_dim * 2) + embed_dim, dec_hid_dim, batch_first=True)
        # self.fc_out = nn.Linear(hid_dim, output_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.attention = attention

    def forward(self, inputs, hidden, encoder_outputs):
        inputs = inputs.unsqueeze(1)
        embedded = self.dropout(self.embedding(inputs))

        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embeded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        pred = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return pred, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.8):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # hidden, cell = self.encoder(src)
        encoder_outputs, hidden = self.encoder(src)

        inputs = trg[:, 0]
        
        for t in range(1, trg_len):
            # output, hidden, cell = self.decoder(inputs, hidden, cell)
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            inputs = trg[:, t] if teacher_force else top1

        return outputs



    