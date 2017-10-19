import collections as col
import json
import numpy as np
import pandas as pd

import torch.utils.data
from torch.autograd import Variable


START_TOKEN, START_IDX = "<S>", 0
END_TOKEN, END_IDX = "<E>", 1
NOTHING_TOKEN, NOTHING_IDX = "<>", 2

_PRESET_TOKENS = [START_TOKEN, END_TOKEN, NOTHING_TOKEN]


class LangDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, token_list, sep):
        self.data_list = data_list
        self.token_list = token_list
        self.sep = sep

        self.t2i, self.i2t = self.get_t2i_and_i2t(token_list)
        self.data_index_list = list(map(self.to_index, data_list))

    def __getitem__(self, index):
        return self.data_index_list[index]

    def __len__(self):
        return len(self.data_index_list)

    def to_index(self, string):
        if self.sep == "":
            raw_tokens = string
        else:
            raw_tokens = string.split(self.sep)
        return np.array(
            [START_IDX] + list(map(self.t2i.get, raw_tokens)) + [END_IDX])

    def to_string(self, index_list):
        return self.sep.join(map(self.i2t.get, index_list))

    def batch_tensor_to_string(self, batch_tensor):
        if isinstance(batch_tensor, Variable):
            batch_tensor = batch_tensor.data

        return [
            self.to_string(tensor_line)
            for tensor_line in batch_tensor.cpu().numpy()
        ]

    @classmethod
    def get_t2i_and_i2t(cls, token_list):
        token_list = _PRESET_TOKENS + token_list[:]
        t2i = col.OrderedDict(zip(token_list, range(len(token_list))))
        i2t = col.OrderedDict(zip(range(len(token_list)), token_list))
        return t2i, i2t


class DoubleLangDataset(torch.utils.data.Dataset):
    def __init__(self, input_lang, output_lang):
        assert len(input_lang) == len(output_lang)
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __getitem__(self, index):
        return (
            self.input_lang[index],
            self.output_lang[index],
        )

    def __len__(self):
        return len(self.input_lang)


def load_data(data_path, tokens_path, word_to_num, char_by_char):
    # Load files
    data_df = pd.read_csv(data_path, header=None, dtype={1: str})
    with open(tokens_path, "r") as f:
        word2num_tokens = json.loads(f.read())

    # Form Langs
    (_, word_column), (_, num_column) = data_df.iteritems()
    word_column = word_column.str.replace("-", " ").str.replace(",", "")
    if char_by_char:
        word_lang = LangDataset(
            data_list=word_column,
            token_list=word2num_tokens["char_tokens"],
            sep="",
        )
    else:
        word_lang = LangDataset(
            data_list=word_column,
            token_list=word2num_tokens["word_tokens"],
            sep=" ",
        )
    num_lang = LangDataset(
        data_list=num_column,
        token_list=word2num_tokens["num_tokens"],
        sep="",
    )

    if word_to_num:
        return DoubleLangDataset(
            input_lang=word_lang,
            output_lang=num_lang,
        )
    else:
        return DoubleLangDataset(
            input_lang=num_lang,
            output_lang=word_lang,
        )


def collate_fn(batch):
    padded_x_ls = []
    padded_y_ls = []
    ordered_x_len_ls = []
    ordered_y_len_ls = []

    x_len_ls = [len(_[0]) for _ in batch]
    y_len_ls = [len(_[1]) for _ in batch]
    max_x_len = max(x_len_ls)
    max_y_len = max(y_len_ls)

    ordered_x = sorted(zip(x_len_ls, range(len(x_len_ls))),
                       reverse=True)

    for _, i in ordered_x:
        (x, y) = batch[i]
        padded_x_ls.append(np.pad(
            x, (0, max_x_len - len(x)),
            'constant', constant_values=NOTHING_IDX,
        ))
        padded_y_ls.append(np.pad(
            y, (0, max_y_len - len(y)),
            'constant', constant_values=NOTHING_IDX,
        ))
        ordered_x_len_ls.append(len(x))
        ordered_y_len_ls.append(len(y))

    padded_x = np.vstack(padded_x_ls)
    padded_y = np.vstack(padded_y_ls)

    if padded_y.dtype != np.dtype("int64"):
        # TODO: Better UNK handling
        raise Exception("Unknown token encountered")

    x_tensor = torch.LongTensor(padded_x)
    y_tensor = torch.LongTensor(padded_y)

    return (
        (x_tensor, ordered_x_len_ls),
        (y_tensor, ordered_y_len_ls),
    )


def get_data_loader(data_path, tokens_path, config, update_config=True):
    dataset = load_data(
        data_path=data_path,
        tokens_path=tokens_path,
        word_to_num=config.word_to_num,
        char_by_char=config.char_by_char,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.cuda,
        collate_fn=collate_fn,
    )
    if update_config:
        input_vocab_size = len(dataset.input_lang.t2i)
        output_vocab_size = len(dataset.output_lang.t2i)

        if config.input_vocab_size is not None \
                and config.input_vocab_size != input_vocab_size:
            raise RuntimeError(
                f"config.input_vocab_size ({config.input_vocab_size})"
                f"!= input_vocab_size ({input_vocab_size})"
            )
        else:
            config.input_vocab_size = input_vocab_size

        if config.output_vocab_size is not None \
                and config.output_vocab_size != output_vocab_size:
            raise RuntimeError(
                f"config.output_vocab_size ({config.output_vocab_size})"
                f"!= output_vocab_size ({output_vocab_size})"
            )
        else:
            config.output_vocab_size = output_vocab_size

    return dataloader
