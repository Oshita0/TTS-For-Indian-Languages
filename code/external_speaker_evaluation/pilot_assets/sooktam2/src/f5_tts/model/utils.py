from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files
from typing import List, Union

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence
import requests
import json
import socket
import subprocess
from datetime import datetime


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

# Get tokenizer

def _get_tokenizer(dataset_name, tokenizer, extra_vocab_path: str = None):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    extra_vocab_path - path to txt file containing additional characters (one per line) to expand vocabulary
    """
    
    if tokenizer in ["pinyin", "char", "cls"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"
        
    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256
        
    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
    
    # Load and merge extra vocabulary from txt file
    if extra_vocab_path is not None and os.path.exists(extra_vocab_path):
        if vocab_char_map is not None:  # Only extend if not byte tokenizer
            current_vocab_size = len(vocab_char_map)
            with open(extra_vocab_path, "r", encoding="utf-8") as f:
                for char in f:
                    char = char.strip()
                    if char and char not in vocab_char_map:
                        vocab_char_map[char] = len(vocab_char_map)
            vocab_size = len(vocab_char_map)
            print(f"Extended vocabulary with {vocab_size - current_vocab_size} new tokens from {extra_vocab_path}")
    
    return vocab_char_map, vocab_size



def get_tokenizer(dataset_name, tokenizer: str | List[str], extra_vocab_path: str = None):
    """
    Return a dictionary of tokenizers if tokenizer is a list, each key is the tokenizer name and value is a tuple of (vocab_char_map, vocab_size)
    Otherwise, return a dictionary with single tokenizer entry
    """
    
    if isinstance(tokenizer, list):
        tokenizers_dict = {}
        for t_name in tokenizer:
            vocab_char_map, vocab_size = _get_tokenizer(dataset_name, t_name, extra_vocab_path)
            tokenizers_dict[t_name] = (vocab_char_map, vocab_size)  # Fixed: was using 't' instead of 't_name'
        return tokenizers_dict
    else:
        vocab_char_map, vocab_size = _get_tokenizer(dataset_name, tokenizer, extra_vocab_path)
        return  vocab_char_map, vocab_size

def save_vocab(vocab_char_map, save_path):
    """Save vocabulary to file"""
    if vocab_char_map is None:
        return
        
    print(f"\nSaving vocabulary to: {save_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Sort by index to maintain order
    vocab_items = sorted(vocab_char_map.items(), key=lambda x: x[1])
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for char, idx in vocab_items:
            f.write(f"{char}\n")
    
    print(f"✓ Saved {len(vocab_char_map)} tokens to vocab.txt")

def send_slack_notification(message, webhook_url=None, title="Training Notification"):
    """Send a notification to a Slack channel via webhook."""
    if webhook_url is None:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    if not webhook_url:
        return False
    
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    payload = {
        "text": f"*{title}*\n*Time:* {timestamp}\n*Host:* {hostname}\n*Message:* {message}"
    }
    
    try:
        response = requests.post(
            webhook_url, 
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")
        return False

def track_with_dvc(path):
    """Track a file or directory with DVC."""
    try:
        # Check if dvc is initialized
        if not os.path.exists(".dvc"):
            print("DVC not initialized. Skipping tracking.")
            return False
            
        print(f"Tracking with DVC: {path}")
        subprocess.run(["dvc", "add", path], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"DVC add failed for {path}: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"Failed to track with DVC: {e}")
        return False

# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
