import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Caption Generation
def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:,0] = start_token
    mask_template[:,0] = False
    return caption_template, mask_template


class TwitterDataset(Dataset):
    def __init__(self, cfg, df, image_captions, transform, create_nested_tensor):
        self.cfg = cfg
        self.tweets = df['tweet_content'].values
        self.targets = df['target'].values
        self.labels = df['sentiment'].values
        self.image_ids = df['image_id'].values
        self.transform = transform
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len
        self.create_nested_tensor = create_nested_tensor

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        target = self.targets[item]
        image_id = self.image_ids[item]
        image = Image.open(os.path.join(self.cfg.data_dir,self.cfg.dataset+"_images",image_id))
        # image process
        image = self.transform(image)
        image = self.create_nested_tensor(image.unsqueeze(0))
        # get default caption and capmask
        start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        caption, cap_mask = create_caption_and_mask(start_token, 128)
        caption = caption.reshape(-1,)
        cap_mask = cap_mask.reshape(-1,)
        # text-aspect encoding (with special tokens)
        encoding = self.tokenizer.encode_plus(
            tweet,
            text_pair = target,
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = False,
            padding="max_length",
            return_attention_mask = True,
            return_tensors = "pt",
            truncation = True,
        )
        # text-only encoding
        tweet_remove = tweet.replace('$T$','')
        if tweet_remove == '':
            tweet_remove = target
        encoding_text = self.tokenizer.encode_plus(
            tweet_remove,
            max_length = 60,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )
        # aspect-only encoding
        encoding_aspect = self.tokenizer.encode_plus(
            target,
            max_length = 12,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )
        # remove the last sep
        # encoding = remove_last_sep(encoding)
        return {
            "review_text": tweet,
            "targets": target,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image": image.tensors.squeeze(0),
            "image_mask": image.mask.squeeze(0),
            "caption": caption,
            "caption_mask": cap_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "input_ids_tt": encoding_text["input_ids"].flatten(),
            "attention_mask_tt": encoding_text["attention_mask"].flatten(),
            "input_ids_at": encoding_aspect["input_ids"].flatten(),
            "attention_mask_at": encoding_aspect["attention_mask"].flatten(),
        }
