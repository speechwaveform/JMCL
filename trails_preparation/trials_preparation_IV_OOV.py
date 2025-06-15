
import pandas as pd
import torch
import os
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import random
import csv
import os
from collections import Counter
import os
import pandas as pd
import itertools
# --- CONFIG ---


IV_TRIAL_CSV = "word_discrimination_trials_IV.csv"
OOV_TRIAL_CSV = "word_discrimination_trials_OOV.csv"

FEATURE_DIR = "test_clean_feats/"



def load_keywords_from_text(text_path, min_len=3):
    keyword_counter = Counter()
    with open(text_path, 'r') as f:
        for line in f:
            utt_id, transcript = line.strip().split(maxsplit=1)
            words = transcript.upper().split()
            filtered_words = [w for w in words if len(w) >= min_len]
            keyword_counter.update(filtered_words)
    return keyword_counter

# Load counts

train_keyword_counts = load_keywords_from_text("data/train/text")
test_keyword_counts = load_keywords_from_text("data/test/text")

train_vocab = set(train_keyword_counts.keys())
test_vocab = set(test_keyword_counts.keys())

iv_keywords = test_vocab.intersection(train_vocab)
oov_keywords = test_vocab.difference(train_vocab)

filenames = os.listdir(FEATURE_DIR)


# Group by word
IV_word2files = []
OOV_word2files = []

for filename in filenames:
    word = filename.split("_")[0].upper()
    if word in iv_keywords:
        IV_word2files.append(filename)
    else:
        OOV_word2files.append(filename)

def get_word_and_filename(filename):
    base = os.path.basename(filename)
    word = base.split('_')[0]
    return word, base

def generate_trials(file_list, output_csv):
    trials = []

    # Generate all unique pairs (i, j) with i <= j to avoid duplicates
    for f1, f2 in itertools.combinations(file_list, 2):
        word1, name1 = get_word_and_filename(f1)
        word2, name2 = get_word_and_filename(f2)
        gt = 1 if word1 == word2 else 0

        trials.append({
            'file1': name1,
            'file2': name2,
            'word1': word1,
            'word2': word2,
            'gt_decision': gt
        })

    df = pd.DataFrame(trials)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} trials to {output_csv}")

generate_trials(OOV_word2files, OOV_TRIAL_CSV)
   
generate_trials(IV_word2files, IV_TRIAL_CSV)

