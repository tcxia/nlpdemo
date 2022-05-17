# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 16:07
# @filename: beam_search.py
# @author: tcxia
'''



import torch
import torch.nn.functional as F

import math

from data.process import nopeak_mask


def init_vars(src, model, src_field, trg_field, top_k, max_len, device):

    init_tok = trg_field.stoi['<sos>'] # <sos>对应的id [target]

    src_mask = (src != src_field.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)


    outputs = torch.LongTensor([[init_tok]])
    outputs = outputs.to(device)

    trg_mask = nopeak_mask(1, device) # [1, 1, 1]

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(top_k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(top_k, max_len).long()
    outputs = outputs.to(device)

    e_output = torch.zeros(top_k, e_output.size(-2), e_output.size(-1))
    e_output = e_output.to(device)

    e_output[:, :] = e_output[0]

    return outputs, e_output, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)

    k_prob, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_prob.unsqueeze(0)

    return outputs, log_scores



def beamSearch(src, model, src_field, trg_field, top_k, max_len, device):

    output, e_output, log_scores = init_vars(src, model, src_field, trg_field,
                                             top_k, max_len, device)

    eos_tok = trg_field.vocab.stoi['<eos>']
    src_mask = (src != src_field.vocab.stoi['<pad>']).unsqueeze(-2)

    ind = None
    for i in range(2, max_len):
        trg_mask = nopeak_mask(i, device)

        out = model.out(model.decoder(output[:, :i], e_output, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)

        output, log_scores = k_best_outputs(output, out, log_scores, i, top_k)

        ones = (output == eos_tok).nonzero()

        sentence_lengths = torch.zeros(len(output), dtype=torch.long).to(device)

        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:
                sentence_lengths[i] = vec[1]

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == top_k:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is not None:
        length = (output[0] == eos_tok).nonzero()[0]
        return ' '.join([trg_field.vocab.itos[tok] for tok in output[0][1:length]])
    else:
        length = (output[ind] == eos_tok).nonzero()[0]
        return ' '.join([trg_field.vocab.itos[tok] for tok in output[ind][1:length]])
