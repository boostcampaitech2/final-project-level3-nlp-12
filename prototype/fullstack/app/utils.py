import re
import json
import emoji
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from soynlp.normalizer import repeat_normalize

import torch

def make_samples(results):
    sample_results = []
    for result in results[:5]:
        sample_results.append({
            "comment": result['comment'],
            "label": result['label']
        })
    return sample_results

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def preprocess(sents):
    """
    kcELECTRA-base preprocess procedure + modification
    """
    preprocessed_sents = []
    
    emojis = set()
    for k in emoji.UNICODE_EMOJI.keys():
        emojis.update(emoji.UNICODE_EMOJI[k].keys())
        
    punc_bracket_pattern = re.compile(f'[\'\"\[\]\(\)]')
    base_pattern = re.compile(f'[^.,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'(http|ftp|https)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )

    for sent in sents:
        sent = punc_bracket_pattern.sub(' ', sent)
        sent = base_pattern.sub(' ', sent)
        sent = url_pattern.sub('', sent)
        sent = sent.strip()
        sent = repeat_normalize(sent, num_repeats=2)
        preprocessed_sents.append(sent)

    return preprocessed_sents


class Preprocess:
    """A class for preprocessing contexts from train and wikipedia
    Args:
        sents (list): context list
        langs (list): language list should be removed from sentence
    """

    PERMIT_REMOVE_LANGS = [
        "arabic",
        "russian",
    ]

    def __init__(self, sents: list):
        self.sents = sents

    def proc_preprocessing(self):
        """
        A function for doing preprocess
        """
        self.remove_hashtag()
        self.remove_user_mention()
        self.remove_bad_char()
        self.clean_punc()
        self.remove_useless_char()
        self.remove_linesign()
        self.remove_repeated_spacing()

        return self.sents

    def remove_hashtag(self):
        """
        A function for removing hashtag
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"#\S+", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_user_mention(self):
        """
        A function for removing mention tag
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"@\w+", "", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_bad_char(self):
        """
        A function for removing raw unicode including unk
        """
        bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
        preprcessed_sents = []
        for sent in self.sents:
            for bad_char in bad_chars:
                sent = sent.replace(bad_char, bad_chars[bad_char])
            sent = re.sub(r"[\+á?\xc3\xa1]", "", sent)
            if sent:
                preprcessed_sents.append(sent)
        self.sents = preprcessed_sents

    def clean_punc(self):
        """
        A function for removing useless punctuation
        """
        punct_mapping = {
            "‘": "'",
            "₹": "e",
            "´": "'",
            "°": "",
            "€": "e",
            "™": "tm",
            "√": " sqrt ",
            "×": "x",
            "²": "2",
            "—": "-",
            "–": "-",
            "’": "'",
            "_": "-",
            "`": "'",
            "“": '"',
            "”": '"',
            "“": '"',
            "£": "e",
            "∞": "infinity",
            "θ": "theta",
            "÷": "/",
            "α": "alpha",
            "•": ".",
            "à": "a",
            "−": "-",
            "β": "beta",
            "∅": "",
            "³": "3",
            "π": "pi",
            "ㅂㅅ": "병신",
            "ㄲㅈ": "꺼져",
            "ㅂㄷ": "부들",
            "ㅆㄹㄱ": "쓰레기",
            "ㅆㅂ": "씨발",
            "ㅈㅅ": "죄송",
            "ㅈㄹ": "지랄",
            "ㅈㄴ": "정말",
        }

        preprocessed_sents = []
        for sent in self.sents:
            for p in punct_mapping:
                sent = sent.replace(p, punct_mapping[p])
            sent = sent.strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_useless_char(self):
        preprocessed_sents = []
        re_obj = re.compile("[^가-힣a-z0-9\x20]+")

        for sent in self.sents:
            temp = re_obj.findall(sent)
            if temp != []:
                for ch in temp:
                    sent = sent.replace(ch, " ")
            sent = sent.strip()
            if sent:
                preprocessed_sents.append(sent)

        self.sents = preprocessed_sents

    def remove_repeated_spacing(self):
        """
        A function for reducing whitespaces into one
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"\s+", " ", sent).strip()
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def spacing_sent(self):
        """
        A function for spacing properly
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = self.spacing(sent)
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents

    def remove_linesign(self):
        """
        A function for removing line sings like \n
        """
        preprocessed_sents = []
        for sent in self.sents:
            sent = re.sub(r"[\n\t\r\v\f\\\\n\\t\\r\\v\\f]", "", sent)
            if sent:
                preprocessed_sents.append(sent)
        self.sents = preprocessed_sents
