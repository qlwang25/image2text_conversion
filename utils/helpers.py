import os
import time
import math
import random
import numpy as np
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# SET SEED
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def cal_max_len(train_df, args):
    # calculate content length
    for text_col in ['tweet_content']:
        content_lengths = []
        tk0 = tqdm(train_df[text_col].fillna("").values, total=len(train_df))
        for text in tk0:
            length = len(args.tokenizer(text, add_special_tokens=False)['input_ids'])
            content_lengths.append(length)
        print(f'{text_col} max(content_lengths): {max(content_lengths)}')
    # calculate target length
    for text_col in ['target']:
        target_lengths = []
        tk0 = tqdm(train_df[text_col].fillna("").values, total=len(train_df))
        for text in tk0:
            length = len(args.tokenizer(text, add_special_tokens=False)['input_ids'])
            target_lengths.append(length)
        print(f'{text_col} max(target_lengths): {max(target_lengths)}')
    # max length: cls & sep & . & reservation & add
    max_length = max(np.array(content_lengths) + np.array(target_lengths)) + 3 + args.max_caption_len + 1
    print(f'Total max length: {max_length}')
    return max_length


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

# Helper funcitons
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))