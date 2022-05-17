# -*- coding: utf-8 -*-
'''
# Created on 11 月-10-20 14:18
# abstract_gen.py
# @author: tcxia
'''
import os
import logging
import numpy as np
from copy import deepcopy

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from ernie.modeling_ernie import ErnieModel, ErnieModelForPretraining, ErnieModelForGeneration
from ernie.modeling_ernie import _build_linear, _build_ln, append_name
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.optimization import AdamW, LinearDecay

from propeller import log
import propeller.paddle as propeller

from decode import beam_search_infilling, post_process


# logging.getLogger().handlers[0] = log.handlers[0]
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger()

data_dir = '/data/cmrc2018'
save_dir = '/data/cmrc2018/save'
predict_output_dir = '/data/cmrc2018/predict'
pretrained_weight = 'ernie-1.0'
attn_token = '[MASK]'
use_random_noice = True
noise_prob = 0.7
max_encode_len = 640
max_decode_len = 120
tgt_type_id = 3
eval_bsz = 20
bsz = 8
lr = 5e-5
warmup_proportion = 0.1
max_steps = 5000
wd = 0.01
label_smooth = 0.1
skip_eval_steps = 1
eval_steps = 5000

beam_width = 5
length_penalty = 1.0


@np.vectorize
def rev_lookup(i):
    return rev_dict[i]

def evaluate(model, datasets, step):
    did = D.parallel.Env().dev_id
    place = F.CUDAPlace(D.parallel.Env().dev_id)
    with open(os.path.join(predict_output_dir, 'pred.step%d.%d' % (step, did)), 'w') as outf:
        for step, data in enumerate(datasets.start(place)):
            (example_id, src_ids, src_sids, src_pids,
             _, _, _,
             _,
             _, _, _, _) = data # never use target when infer
            output_ids = beam_search_infilling(model, src_ids, src_sids,
                    eos_id=tokenizer.sep_id,
                    sos_id=tokenizer.cls_id,
                    attn_id=tokenizer.vocab[attn_token],
                    max_decode_len=max_decode_len,
                    max_encode_len=max_encode_len,
                    beam_width=beam_width,
                    length_penalty=length_penalty,
                    tgt_type_id=tgt_type_id,)
            output_str = rev_lookup(output_ids.numpy())
            for eid, ostr in zip(example_id.numpy().tolist(), output_str.tolist()):
                if '[SEP]' in ostr:
                    ostr = ostr[: ostr.index('[SEP]')]
                ostr = ''.join(map(post_process, ostr))
                print('%d\t%s' % (eid, ostr), file=outf)

    model.train()

def seq2seq(model, tokenizer):
    log.info('Training starts')
    attn_id = tokenizer.vocab[attn_token]
    def gen_mask(batch_ids, mask_type='bidi', query_len=None, pad_value=0):
        if query_len is None:
            query_len = batch_ids.shape[1]
        if mask_type != 'empty':
            mask = (batch_ids != pad_value).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
            if mask_type == 'causal':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask)
            elif mask_type == 'causal_without_diag':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask, -1)
            elif mask_type == 'diag':
                assert query_len == batch_ids.shape[1]
                mask = np.stack([np.diag(np.diag(m)) for m in mask], 0)
        else:
            mask_type == 'empty'
            mask = np.zeros_like(batch_ids).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
        return mask


    def make_some_noice(ids):
        if use_random_noice:
            noice_ids = np.random.randint(1, len(tokenizer.vocab), size=ids.shape)
        else:
            noice_ids = np.ones_like(ids) * tokenizer.vocab['[NOISE]']
        pos, = np.where(np.ones_like(ids))
        np.random.shuffle(pos)
        pos = pos[:int(noise_prob * len(pos))]
        ids[pos, ] = noice_ids[pos, ]
        return ids

    def map_fn(example_id, src_ids, tgt_ids):
        src_ids = src_ids[:max_encode_len]
        tgt_ids = tgt_ids[:max_decode_len]
        src_ids, src_sids = tokenizer.build_for_ernie(src_ids)
        src_pids = np.arange(len(src_ids))

        tgt_ids, tgt_sids = tokenizer.build_for_ernie(tgt_ids)
        tgt_pids = np.arange(len(tgt_ids)) + len(src_ids)  # continues position
        tgt_sids = np.ones_like(tgt_sids) * tgt_type_id

        attn_ids = np.ones_like(tgt_ids) * attn_id
        if noise_prob > 0.:
            tgt_labels = deepcopy(tgt_ids)
            tgt_ids = make_some_noice(tgt_ids)  #corrupted
        else:
            tgt_labels = tgt_ids

        return (example_id, src_ids, src_pids, src_sids, tgt_ids, tgt_pids,
                tgt_sids, attn_ids, tgt_labels)


    def after_padding(example_id, src_ids, src_pids, src_sids, tgt_ids, tgt_pids,
                    tgt_sids, attn_ids, tgt_labels):
        '''
            attention mask:
            ***  src,  tgt, attn
            src  00,   01,   11
            tgt  10,   11,   12
            attn 20,   21,   22
            ***   s1, s2 | t1 t2 t3| attn1 attn2 attn3
            s1    1,  1  | 0, 0, 0,| 0,    0,    0,
            s2    1,  1  | 0, 0, 0,| 0,    0,    0,
            -
            t1    1,  1, | 1, 0, 0,| 0,    0,    0,
            t2    1,  1, | 1, 1, 0,| 0,    0,    0,
            t3    1,  1, | 1, 1, 1,| 0,    0,    0,
            -
            attn1 1,  1, | 0, 0, 0,| 1,    0,    0,
            attn2 1,  1, | 1, 0, 0,| 0,    1,    0,
            attn3 1,  1, | 1, 1, 0,| 0,    0,    1,
            for details, see Fig3. https://arxiv.org/abs/2001.11314
            '''

        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1]
        mask_00 = gen_mask(src_ids, 'bidi', query_len=src_len)
        mask_01 = gen_mask(tgt_ids, 'empty', query_len=src_len)
        mask_02 = gen_mask(attn_ids, 'empty', query_len=src_len)

        mask_10 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_11 = gen_mask(tgt_ids, 'causal', query_len=tgt_len)
        mask_12 = gen_mask(attn_ids, 'empty', query_len=tgt_len)

        mask_20 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_21 = gen_mask(tgt_ids, 'causal_without_diag', query_len=tgt_len)
        mask_22 = gen_mask(attn_ids, 'diag', query_len=tgt_len)
        '''
            mask = np.concatenate([
                np.concatenate([mask_00, mask_01, mask_02], 2),
                np.concatenate([mask_10, mask_11, mask_12], 2),
                np.concatenate([mask_20, mask_21, mask_22], 2),
            ], 1)
            ids = np.concatenate([src_ids, tgt_ids, attn_ids], 1)
            pids = np.concatenate([src_pids, tgt_pids, tgt_pids], 1)
            sids = np.concatenate([src_sids, tgt_sids, tgt_sids], 1)
            '''

        mask_src_2_src = mask_00
        mask_tgt_2_srctgt = np.concatenate([mask_10, mask_11], 2)
        mask_attn_2_srctgtattn = np.concatenate([mask_20, mask_21, mask_22], 2)

        tgt_labels = tgt_labels[np.where(tgt_labels != 0)]
        return (example_id, src_ids, src_sids, src_pids, tgt_ids, tgt_sids,
                tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                mask_attn_2_srctgtattn, tgt_labels)
    bytes_vocab = {k.encode('utf8'): v for k, v in tokenizer.vocab.items()}
    feature_column = propeller.data.FeatureColumns([
        propeller.data.LabelColumn('id'),
        propeller.data.TextColumn('src', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
        propeller.data.TextColumn('tgt', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
    ])

    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(data_dir, 'train'), shuffle=False, repeat=True, use_gz=False) \
                                   .map(map_fn)

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(eval_bsz) \
                                   .map(after_padding)

    log.debug('shard %d of %d'%(D.parallel.Env().dev_id, D.parallel.Env().nranks))
    train_ds = train_ds.shard(D.parallel.Env().nranks, D.parallel.Env().dev_id).shuffle(10000).padded_batch(bsz).map(after_padding)
    dev_ds = dev_ds.shard(D.parallel.Env().nranks, D.parallel.Env().dev_id)

    shapes = [[None, None]] * 7 + [[None, None, None]] * 3 +[[None]]
    types = ['int64'] * 11

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types

    vocab_size, _ = model.word_emb.weight.shape
    ctx = D.parallel.prepare_context()
    model = D.parallel.DataParallel(model, ctx)
    g_clip = F.clip.GradientClipByGlobalNorm(1.0)
    opt = AdamW(learning_rate=LinearDecay(lr, int(warmup_proportion * max_steps), max_steps), parameter_list=model.parameters(), weight_decay=wd, grad_clip=g_clip)
    attn_id = tokenizer.vocab[attn_token]
    for step, data in enumerate(train_ds.start(place)):
        (example_id, src_ids, src_sids, src_pids,
         tgt_ids, tgt_sids, tgt_pids,
         attn_ids,
         mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn, tgt_labels) = data

        _, __, info = model(src_ids, sent_ids=src_sids, pos_ids=src_pids, attn_bias=mask_src_2_src, encode_only=True)
        cached_k, cached_v = info['caches']
        _, __, info = model(tgt_ids, sent_ids=tgt_sids, pos_ids=tgt_pids, attn_bias=mask_tgt_2_srctgt, past_cache=(cached_k, cached_v), encode_only=True)
        cached_k2, cached_v2 = info['caches']
        past_cache_k = [L.concat([k, k2], 1) for k, k2 in zip(cached_k, cached_k2)]
        past_cache_v = [L.concat([v, v2], 1) for v, v2 in zip(cached_v, cached_v2)]
        if label_smooth > 0.:
            tgt_labels = L.label_smooth(F.one_hot(tgt_labels, vocab_size), epsilon=label_smooth)
        loss, _, __ = model(attn_ids, sent_ids=tgt_sids, pos_ids=tgt_pids, attn_bias=mask_attn_2_srctgtattn,
                past_cache=(past_cache_k, past_cache_v),
                tgt_labels=tgt_labels,
                tgt_pos=L.where(attn_ids==attn_id))

        scaled_loss = model.scale_loss(loss)
        scaled_loss.backward()
        model.apply_collective_grads()
        opt.minimize(scaled_loss)
        model.clear_gradients()
        if step % 10 == 0:
            loss = loss.numpy()
            ppl = np.exp(loss)
            log.debug('[step %d]train loss %.5f, ppl %.5f, lr %.3e' % (step, loss, ppl, opt.current_step_lr()))
        if save_dir is not None and step % 1000 == 0 and D.parallel.Env().dev_id == 0:
            F.save_dygraph(model.state_dict(), save_dir)
        if predict_output_dir is not None and step > skip_eval_steps and step % eval_steps == 0:
            assert os.path.exists(predict_output_dir), 'predict_output_dir not found: %s' % predict_output_dir
            log.debug('doing predict on gpu %d...' % D.parallel.Env().dev_id)
            evaluate(model, dev_ds, step)
        if step > max_steps:
            break
        evaluate(model, dev_ds, step)

    if save_dir is not None:
        F.save_dygraph(model.state_dict(), save_dir)


if __name__ == "__main__":

    place = F.CUDAPlace(D.parallel.Env().dev_id)
    D.guard(place).__enter__()
    # place = F.CUDAPlace(2)
    # D.guard(place)

    ernie = ErnieModelForGeneration.from_pretrained(pretrained_weight)
    tokenizer = ErnieTokenizer.from_pretrained(pretrained_weight, mask_token=None)

    rev_dict = {v:k for k, v in tokenizer.vocab.items()}
    rev_dict[tokenizer.pad_id] = ''
    rev_dict[tokenizer.unk_id] = ''
    # print(rev_dict)
    # print(tokenizer)
    seq2seq(ernie, tokenizer)