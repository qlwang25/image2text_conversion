import os
import pandas as pd
import emoji

def load_data(args, replace_emoji=True):
    # data loading
    train_tsv = os.path.join(args.data_dir, args.dataset, "train.tsv")
    dev_tsv = os.path.join(args.data_dir, args.dataset, "dev.tsv")
    test_tsv = os.path.join(args.data_dir, args.dataset, "test.tsv")

    test_df = pd.read_csv(test_tsv, sep="\t")
    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(dev_tsv, sep="\t")

    test_df = test_df.rename(
        {
            "index": "sentiment",
            "#1 ImageID": "image_id",
            "#2 String": "tweet_content",
            "#2 String.1": "target",
        },
        axis=1,
    )
    train_df = train_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#3 String.1": "target",
        },
        axis=1,
    ).drop(["index"], axis=1)
    val_df = val_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#3 String.1": "target",
        },
        axis=1,
    ).drop(["index"], axis=1)
    
    if replace_emoji:
        train_df['tweet_content'] = train_df['tweet_content'].apply(emoji.replace_emoji)
        val_df['tweet_content'] = val_df['tweet_content'].apply(emoji.replace_emoji)
        test_df['tweet_content'] = test_df['tweet_content'].apply(emoji.replace_emoji)
        
    return train_df, val_df, test_df