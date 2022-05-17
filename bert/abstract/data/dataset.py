# -*- coding: utf-8 -*-
'''
# Created on 11 月-23-20 11:23
# dataset.py
# @author: tcxia
'''
import glob
from  torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

class AbstractData(Dataset):
    def __init__(self, dir_path, model_path) -> None:
        super(AbstractData, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.txts = glob.glob(dir_path + '/*/*.txt')


    def __getitem__(self, index: int):
        each_text_path = self.txts[index]
        with open(each_text_path, 'r', encoding='utf-8') as fr:
            text = fr.read()

        text = text.split('\n')
        if len(text) > 1:
            title = text[0]
            content = '\n'.join(text[1])
            tokens = self.tokenizer.tokenize('[CLS]' + content + '[SEP]' + title)

            token_sep_id = tokens.index('[SEP]')
            title_l = tokens[token_sep_id:]
            content_l = tokens[:token_sep_id]

            if len(tokens) > 512:
                # tokens = tokens[:512]
                tokens = content_l[:-(len(tokens) - 512)] + title_l


            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [
                0 if i <= token_ids.index(102) else 1
                for i in range(len(token_ids))
            ]

            return tokens, token_ids, token_type_ids

    def __len__(self) -> int:
        return len(self.txts)


if __name__ == "__main__":
    ds = AbstractData(
        '/data/nlp_dataset/THUCNews',
        '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext')
    dl = DataLoader(ds, batch_size=10, num_workers=0, shuffle=True, pin_memory=True)
    tokens, token_ids, token_type_ids = next(iter(dl))
    print(token_ids)