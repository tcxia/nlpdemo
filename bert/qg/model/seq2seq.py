# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 11:02
# seq2seq.py
# @author: tcxia
'''


import random
import torch
import torch.nn as nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device, encoder_trained) -> None:
        super().__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_trained = encoder_trained

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        input_ids, token_type_ids, attention_mask = src
        # input_ids: [8, 512]

        if self.encoder_trained:
            bert_hs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                bert_hs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


        bert_encodings = bert_hs[0]
        # print(bert_encodings.shape)  # [8, 512, 1024]

        batch_size = trg.shape[0]
        max_len = trg.shape[1]

        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        output = trg[:, 0]

        hidden = torch.zeros(self.decoder.num_layers, output.shape[0], self.decoder.dec_hid_dim).to(self.device)

        for t in range(1, max_len):
            output, hidden = self.decoder(output, bert_encodings, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[:, t] if teacher_force else top1)
        
        return outputs