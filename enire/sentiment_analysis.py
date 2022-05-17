import re
import os
import numpy as np
from tqdm import tqdm
import logging

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L
from tqdm.std import tqdm

#export PYTHONPATH=$PWD:$PYTHONPATH
from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger()

from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay

train_flags = False
epochs = 10
data_dir = '/data/chnsenticorp'
bsz = 32
max_seqlen = 128
lr = 5e-5
warmup_proportion = 0.1
wd = 0.01
max_steps = 5000
save_dir = '/data/chnsenticorp/save' # 保存成save.pdparams
# pretrained_weight = '/data/model-ernie1.0.1.tar.gz'
pretrained_weight = 'ernie-1.0'
tokenizer = ErnieTokenizer.from_pretrained(pretrained_weight)

place = F.CUDAPlace(2)
with FD.guard(place):
    model = ErnieModelForSequenceClassification.from_pretrained(pretrained_weight, num_labels=3, name='')

    if train_flags:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.TextColumn('seg_a', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
            propeller.data.LabelColumn('label'),
        ])

        def map_fn(seg_a, label):
            seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=max_seqlen)
            sentence, segments = tokenizer.build_for_ernie(seg_a, [])
            return sentence, segments, label

        train_ds = feature_column.build_dataset('train', data_dir=os.path.join(data_dir, 'train'), shuffle=True, repeat=False, use_gz=False).map(map_fn).padded_batch(bsz)
        dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False).map(map_fn).padded_batch(bsz)

        shapes = ([-1, max_seqlen], [-1, max_seqlen], [-1])
        types = ('int64', 'int64', 'int64')

        train_ds.data_shapes = shapes
        train_ds.data_types = types
        dev_ds.data_shapes = shapes
        dev_ds.data_types = types

        g_clip = F.clip.GradientClipByGlobalNorm(1.0)
        opt = AdamW(learning_rate=LinearDecay(
            lr,
            int(warmup_proportion * max_steps), max_steps),
            parameter_list=model.parameters(),
            weight_decay=wd,
            grad_clip=g_clip
        )

        for epoch in range(epochs):
            for step, d in enumerate(tqdm(train_ds.start(place), desc='training')):
                ids, sids, label = d
                loss, _ = model(ids, sids, labels=label)
                loss.backward()
                if step % 10 == 0:
                    log.debug('train loss %.5f lr %.3e' %(loss.numpy(), opt.current_step_lr()))
                opt.minimize(loss)
                model.clear_gradients()
                if step % 100 == 0:
                    acc = []
                    with FD.base._switch_tracer_mode_guard_(is_train=False):
                        model.eval()
                        for step, d in enumerate(tqdm(dev_ds.start(place), desc='evaluating %d' % epoch)):
                            ids, sids, label = d
                            loss, logits = model(ids, sids, labels=label)
                            a = L.argmax(logits, -1) == label
                            acc.append(a.numpy())
                        model.train()
                    log.debug('acc %.5f' % np.concatenate(acc).mean())
        if save_dir is not None:
            F.save_dygraph(model.state_dict(), save_dir)

    else:
        feature_column = propeller.data.FeatureColumns([
            propeller.data.TextColumn('seg_a',unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
        ])

    
        assert save_dir is not None
        sd, _ = FD.load_dygraph(save_dir)
        model.set_dict(sd)
        model.eval()

        def map_fn(seg_a):
            seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=max_seqlen)
            sentence, segments = tokenizer.build_for_ernie(seg_a, [])
            return sentence, segments

        #predict_ds = feature_column.build_dataset_from_stdin('predict').map(map_fn).padded_batch(bsz)
        test_ds = feature_column.build_dataset('test', data_dir=os.path.join(data_dir, 'test'), shuffle=False, repeat=False, use_gz=False).map(map_fn).padded_batch(bsz)
        
        shapes = ([-1, max_seqlen], [-1, max_seqlen])
        types = ('int64', 'int64')
        test_ds.data_shapes = shapes
        test_ds.data_types = types

        for step, (ids, sids) in enumerate(test_ds.start(place)):
            _, logits = model(ids, sids)
            pred = logits.numpy().argmax(-1)
            print('\n'.join(map(str, pred.tolist())))