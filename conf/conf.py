class Config: 
    
    def __init__(self):
       
        self.sample_rate = 16000
        self.n_fft = 1024 
        self.win_length = int(self.sample_rate * 0.025) 
        self.hop_length = int(self.sample_rate * 0.010) 
        self.n_mels = 128
        self.f_max = 8000
        self.f_min = 0
        self.data_info_dict={'train':{'wav_dir': '/home/user/asr_data/LibriSpeech/','alignments_path':'data/train/words_alignment_train-clean-100.pkl'},
                             'test':{'wav_dir': '/home/user/asr_data/LibriSpeech/','alignments_path':'data/test/words_alignment_test-clean.pkl'}} 
        self.batch_size = 128
        self.hidden_dim = 256
        self.embedding_dim = 512
        self.no_of_tokens = 87
        self.input_dim_text = 256
        self.num_epochs = 30
        self.number_examples_wer_word = 4
        self.wt_clap_loss = 0.1
        self.wt_dwd_loss = 1
        self.num_layers = 3
