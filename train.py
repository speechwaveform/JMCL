import sys
import os
import copy
import torch
import argparse
from timeit import default_timer as timer
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from conf.conf import Config
from dataloader.dataloader import create_data_loader
from dataloader.dataset import TripletSpeechDataset
from model.NAW_models import NAW_LSTM_multi_view_model
from utils.steps import train
from utils.transforms import mel_spec_transform
from utils.utils import ensure_dir_exists
#from utils.loss_functions import MultiObjectiveContrastiveLoss
# Argument parsing
parser = argparse.ArgumentParser(description='Train an Neural acoustic word embedding model.')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to train the model on (default: cuda)')#, choices=['cuda', 'cpu']
parser.add_argument('--ckpt_dir_name', type=str, default='test',
                    help='checkpoint directory name ')

parser.add_argument('--log_dir_name', type=str, default='test',
                    help='tensorboard log directory name')

args = parser.parse_args()

# Set device
device = args.device #if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
print(f"Using device: {device}")

# Loading configuration
print("Loading configuration...")
conf = Config()
print("Configuration loaded.")

# Feature extractor
audio_transforms = torch.nn.Sequential(
    mel_spec_transform(config=conf)
)

# Example usage
print("Loading datasets...")
NAWTrainData = TripletSpeechDataset(conf, transform=audio_transforms, type="train")
NAWValData = TripletSpeechDataset(conf, transform=audio_transforms, type="test")
print("Datasets loaded.")

# Dataloaders
print("Creating dataloaders...")
train_loader = create_data_loader(
    dataset=NAWTrainData,
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=10, 
    verification=True
)

test_loader = create_data_loader(
    dataset=NAWValData,
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=10,
    verification=True
)
print("Dataloaders created.")

# Model
print("Initializing model...")
model = NAW_LSTM_multi_view_model( input_dim_acoustic = conf.n_mels, input_dim_text = conf.input_dim_text , no_of_tokens = conf.no_of_tokens, hidden_dim = conf.hidden_dim, embedding_dim = conf.embedding_dim, num_layers=conf.num_layers, bidirectional=True).to(device)

print("Model initialized.")



# Loss function and optimizer
print("Setting up loss function and optimizer...")
loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)
#loss_fn = MultiObjectiveContrastiveLoss(margin=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Set number of epochs
NUM_EPOCHS = 30
print(f"Number of epochs set to: {NUM_EPOCHS}")

# Define the scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-3,             
    steps_per_epoch=len(train_loader),  
    epochs=NUM_EPOCHS,
    pct_start=0.15,         
    anneal_strategy='cos',   
    div_factor=10.0,          
    final_div_factor=100.0   
)

print("Loss function and optimizer setup complete.")

# Ensure directories exist
print("Setting up directories...")
cwd = os.getcwd()
# CHECKPOINT_DIR = "exp/ckpt/complete_frame_50_3/"
# LOG_DIR = "exp/logs/complete_frame_50_3/"
CHECKPOINT_DIR = f"exp/ckpt/{args.ckpt_dir_name}/"
LOG_DIR = f"exp/logs/{args.ckpt_dir_name}/"


ensure_dir_exists(os.path.join(cwd, CHECKPOINT_DIR))
ensure_dir_exists(os.path.join(cwd, LOG_DIR))
print("Directories are set up.")

# Start the timer
print("Starting training...")
start_time = timer()

# print(f"conf : {conf}")

# Train the model
train(
    model=model, 
    train_dataloader=train_loader,
    validation_dataloader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn, 
    epochs=NUM_EPOCHS,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    device=device, 
    config = conf
)

# Save last checkpoint
print("Saving the last checkpoint...")
model_save_path = os.path.join(CHECKPOINT_DIR, 'model_last.pt')
torch.save(copy.deepcopy(model).state_dict(), model_save_path)
print(f"Last checkpoint saved at: {model_save_path}")

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")

# Indicate end of script
print("Script execution completed.")


