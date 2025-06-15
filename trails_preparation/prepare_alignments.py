import csv
from collections import defaultdict
import os
import pickle
from tqdm import tqdm
# Adjust this to your CSV file path
csv_path = 'librispeech_clean_train_100h_all_utt.csv'

word_dict = defaultdict(lambda: defaultdict(list))



with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
    
        if len(row) < 4:
            continue
        wav_path, start, end, word = row
        word = word.strip()
        if word == '<unk>':
            continue
        if word == '':
            continue  # Skip empty words (silences)

        #word_dict[word][wav_path].append([float(start), float(end)])
        if (float(end) - float(start)) >=0.5 and (float(end) - float(start))<=2 :
        #    word_dict[word][wav_path] = [start, end]
            word_dict[word][wav_path].append([float(start), float(end)])
# Optional: Convert defaultdict to dict for serialization or inspection
final_dict = {word: dict(files) for word, files in word_dict.items()}


words = (final_dict.keys())
instances = []
for word in words:
    for file, times in word_dict[word].items():
        for time in times:
            instances.append((word, file, time))
            
# Path to save the pickle file
pickle_path = 'words_alignment_train-clean-100.pkl'


# Save dictionary to pickle file
with open(pickle_path, 'wb') as f:
    pickle.dump(instances, f)

print(f"Saved dictionary to {pickle_path}")




csv_path = 'librispeech_clean_test_all_utt.csv'

word_dict = defaultdict(lambda: defaultdict(list))



with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
    
        if len(row) < 4:
            continue
        wav_path, start, end, word = row
        word = word.strip()
        if word == '<unk>':
            continue
        if word == '':
            continue  # Skip empty words (silences)

        #word_dict[word][wav_path].append([float(start), float(end)])
        if (float(end) - float(start)) >=0.5 and (float(end) - float(start))<=2 :
        #    word_dict[word][wav_path] = [start, end]
            word_dict[word][wav_path].append([float(start), float(end)])
# Optional: Convert defaultdict to dict for serialization or inspection
final_dict = {word: dict(files) for word, files in word_dict.items()}


words = (final_dict.keys())
instances = []
for word in words:
    for file, times in word_dict[word].items():
        for time in times:
            instances.append((word, file, time))
            
# Path to save the pickle file

pickle_path = 'words_alignment_test-clean.pkl'

# Save dictionary to pickle file
with open(pickle_path, 'wb') as f:
    pickle.dump(instances, f)

print(f"Saved dictionary to {pickle_path}")

## Save melspectrogram:

import torchaudio
import torch
def mel_spec_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=128,
        mel_scale="htk",
    )
transform = torch.nn.Sequential(
    mel_spec_transform()
)
    

feat_dir = 'test_clean_feats/'
wav_dir = '/data/user/asr_data/LibriSpeech/'
for i in tqdm(range(len(instances))):
    path = instances[i][1]
    waveform, sr = torchaudio.load(wav_dir+path)
    word = instances[i][0]
    file_name =  path.split('/')[-1].split('.')[0]
    start = int(instances[i][2][0]*16000)
    end = int(instances[i][2][1]*16000)

    segment = waveform[:, start:end]

    segment = transform(segment).squeeze(0).transpose(0,1)
    torch.save(segment, feat_dir+"%s_%s_%d.pt"%(word,file_name,i))

