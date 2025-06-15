import torch
import numpy as np
import math
from torch.utils.data import DataLoader


## collate function
def data_processing(data, verification = False):
    #print(data)
    data = list(filter(lambda x: x is not None, data))
    
    spectrograms = []
    spectrograms_lengths = []
    keywords = []
    anc_sequences = []
    anc_sequences_lenghts = []
    # for verifications
    if verification:
        signal_filenames = []
        keywords = []
        query_utt_ids = []

    for sample in data:

        # signal spectrograms

        spectrograms.append(sample['mels'])
        spectrograms_lengths.append(sample['mel_lengths'])
        
        keywords.append(sample['anchor_word'])
        anc_sequences.append(sample['anc_seq'])
        anc_sequences_lenghts.append(sample['anc_seq'].shape[1])
        

    _, idx = np.unique(keywords, return_index=True)
    unique_indices = idx.tolist()
    
    selected_spectrograms = [spectrograms[i] for i in unique_indices]
    selected_lengths = [spectrograms_lengths[i] for i in unique_indices]
    selected_anc_sequences = [anc_sequences[i] for i in unique_indices]
    selected_anc_sequences_lenghts = [anc_sequences_lenghts[i] for i in unique_indices]
    
    
    flat_spectrograms = [seq for sublist in selected_spectrograms for seq in sublist]
    flat_lenghts = [seq for sublist in selected_lengths for seq in sublist]    
    flat_anc_sequences = [seq for sublist in selected_anc_sequences for seq in sublist]
    flat_anc_sequences_lenghts = [x for x in selected_anc_sequences_lenghts for _ in range(4)]
    
    spectrograms = torch.nn.utils.rnn.pad_sequence(flat_spectrograms, batch_first=True)
    
    spectrograms_lengths = torch.tensor(flat_lenghts, dtype=torch.int)
    
    anc_sequences = torch.nn.utils.rnn.pad_sequence(flat_anc_sequences, batch_first=True, padding_value=0)    
    anc_sequences_lenghts = torch.tensor(flat_anc_sequences_lenghts, dtype=torch.int)
    
    B = spectrograms.shape[0]
    word_labels = torch.tensor([i // 4 for i in range(B)]) # 8 word classes, 4 examples each
   
    
    return spectrograms, anc_sequences, spectrograms_lengths, word_labels , anc_sequences_lenghts

# DataLoader initialization
def create_data_loader(dataset, batch_size, shuffle=True, num_workers=1, verification=True):

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: data_processing(x, verification=verification),
        num_workers=num_workers
    )
