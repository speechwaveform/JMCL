# Joint Multimodal Contrastive Learning for Robust Spoken Term Detection and Keyword Spotting


This repository contains resources, models, or scripts for training the models


##  Download 

```bash
git clone https://github.com/speechwaveform/JMCL.git
cd JMCL
```

##  Create alignments and trails preparation 

```bash
cd trails_preparation
python prepare_alignments.py
python trials_preparation_IV_OOV.py
cd ../
```

##  Training

Run the following script to train the NAWE

```
python train.py --device 'cuda:0' --ckpt_dir_name exp
```
