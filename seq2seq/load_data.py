from logging.config import valid_ident
import torch
import jieba
from torchtext import data

device = "cuda" if torch.cuda.is_available() else "cpu"

def tokenizer(text):
    token = [tok for tok in jieba.cur(text)]
    return token

TEXT = data.Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

train, val = data.TabularDataset(
    path='',
    train='train.tsv',
    validation='dev.tsv',
    format='tsv',
    skip_header=True,
    fields=[('tag', TEXT), ('src', TEXT)]
)

TEXT.build_vocab(train, min_freq=2)
id2vocab = TEXT.vocab.itos
vocab2id = TEXT.vocab.stoi

PAD_IDX = vocab2id[TEXT.pad_token]
UNK_IDX = vocab2id[TEXT.unk_token]
SOS_IDX = vocab2id[TEXT.init_token]
EOS_IDX = vocab2id[TEXT.eos_token]

train_iter, val_iter = data.BucketIterator.splits(
    (train, val),
    batch_sizes=(256, 256),
    sort_key=lambda x: len(x.src),
    device=device
)