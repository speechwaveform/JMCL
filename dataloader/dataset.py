import random
import numpy as np
import torch
import librosa
import pickle
import os
import torchaudio
from torch.utils.data import Dataset
from g2p_en import G2p
import random
g2p=G2p()


class TripletSpeechDataset(Dataset):
    def __init__(self, config, transform=None, type="train"):
        self.config = config
        self.type = type
        self.word_dict = pickle.load(open(self.config.data_info_dict[type]['alignments_path'], "rb"))

        self.words = list(self.word_dict.keys())

        self.transform = transform
        
        self.phones_dict={}
        phones=open('data/lang/phones.txt').read().splitlines()
        for i in range(len(phones)):
            key=phones[i].split()[0]
            value=int(phones[i].split()[1])
            self.phones_dict[key]=value

        # Build all (word, file, [start, end]) triplets
        self.instances = []
        for word in self.words:
            if len(self.word_dict[word].keys())>=self.config.number_examples_wer_word:
                for file, times in self.word_dict[word].items():
                    self.instances.append((word, file, times))

    def __len__(self):
        return len(self.instances)

    def _load_segment(self, wav_path, word, start, end):
        start = float(start)
        end = float(end)
        wav_path =  os.path.join(self.config.data_info_dict[self.type]['wav_dir'], wav_path)

        waveform, sr = torchaudio.load(wav_path)

            
        # Crop segment
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        segment = waveform[:, start_frame:end_frame]
        if segment.size(-1) < 1024:
            print(wav_path,start,end,word)
        # Apply transform (e.g., MFCC, MelSpectrogram)
        if self.transform:
            segment = self.transform(segment)
        

        return segment.squeeze(0).transpose(0, 1)  # [T, D]

    def __getitem__(self, idx):
        anchor_word, anchor_file, anchor_time = self.instances[idx]
        anchor = self._load_segment(anchor_file, anchor_word, *anchor_time)
        
        
        # Positive sample (different occurrence of same word)
        pos_choices = [
            (f, ts) for f, ts in self.word_dict[anchor_word].items()
            if not (f == anchor_file and ts == anchor_time)
        ]

        
        choices = random.sample(pos_choices,self.config.number_examples_wer_word-1)
        mels = []
        mels_lengths=[]
        mels.append(anchor)
        mels_lengths.append(anchor.shape[0])
        for choice in choices:
            pos_file, pos_time = choice
            positive = self._load_segment(pos_file,anchor_word, *pos_time)
            mels.append(positive)
            mels_lengths.append(positive.shape[0])
            
        
        anc_phones = g2p(anchor_word.replace("'",""))
        anc_seq=[[self.phones_dict[x] for x in anc_phones]]*self.config.number_examples_wer_word
        anc_seq = np.array(anc_seq).astype('uint8')
        anc_seq = torch.IntTensor(anc_seq)

        return {
                'mels' : mels,
                'anchor_word' : anchor_word,
                'mel_lengths' :  mels_lengths,
                'anc_seq' : anc_seq,
            }

            
