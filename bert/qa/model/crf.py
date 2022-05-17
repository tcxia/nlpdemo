# -*- coding: utf-8 -*-
'''
# Created on 11 月-19-20 16:58
# crf.py
# @author: tcxia
'''

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, target_size, device) -> None:
        super(CRF, self).__init__()
        self.target_size = target_size
        self.device = device

        self.START_TAG, self.END_TAG = -2, -1
        init_transitions = torch.zeros(self.target_size + 2,
                                       self.target_size + 2,
                                       device=self.device)
        init_transitions[:, self.START_TAG] = -10000.0
        init_transitions[self.END_TAG, :] = -10000.0

        self.transitions = nn.Parameter(init_transitions)

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def log_sum_exp(self, vec, m_size):
        _, idx = torch.max(vec, 1)
        max_score = torch.gather(vec, 1, idx.view(-1, 1,
                                                  m_size)).view(-1, 1, m_size)
        return max_score.view(-1, m_size) + torch.log(
            torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(
                -1, m_size)

    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def _viterbi_decode(self, feats, mask):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        back_points = []
        partition_history = []

        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)

        partition = inivalues[:, self.START_TAG, :].clone().view(
            batch_size, tag_size)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)

            cur_bp.masked_fill_(mask[idx].view(batch_size,
                                               1).expand(batch_size, tag_size).bool(),
                                0)  #
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history,
                                      0).view(seq_len, batch_size,
                                              -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(
            batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1,
                                      last_position).view(
                                          batch_size, tag_size, 1)

        last_values = last_partition.expand(
            batch_size, tag_size, tag_size) + self.transitions.view(
                1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)

        pad_zero = torch.zeros(batch_size,
                               tag_size,
                               device=self.device,
                               requires_grad=True).long()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size,
                                                  tag_size)

        pointer = last_bp[:, self.END_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(
            batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = torch.empty(seq_len,
                                 batch_size,
                                 device=self.device,
                                 requires_grad=True).long()
        decode_idx[-1] = pointer.detach()

        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1,
                                   pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)

        patch_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return patch_score, decode_idx

    def _forward_alg(self, feats, mask):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)

        partition = inivalues[:, self.START_TAG, :].clone().view(
            batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = self.log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size,
                                         1).expand(batch_size, tag_size)

            masked_cur_partition = cur_partition.masked_select(
                mask_idx)  #
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            partition.masked_scatter_(mask_idx, masked_cur_partition)  #

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        cur_partition = self.log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG]
        return final_partition.sum(), scores

    def _score_sentence(self, scores, mask, tags):
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        tags = tags.view(batch_size, seq_len)

        new_tags = torch.empty(batch_size,
                               seq_len,
                               device=self.device,
                               requires_grad=True).long()

        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        end_ids = torch.gather(tags, 1, length_mask - 1)

        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).contiguous().view(
            seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2,
                                 new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))  #

        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score
