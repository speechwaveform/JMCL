import torch
import torchaudio
import os
import copy
from conf.conf import Config
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import CLAP_loss,DeepWordDiscriminationLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
        
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        writer: SummaryWriter,  # Add writer to log metrics
        device: torch.device,
        epoch: int, 
        config: Config
    ) -> float:
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        writer (SummaryWriter): TensorBoard writer.
        device (torch.device): Device to perform computations on.
        epoch (int): Current epoch number for logging.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    
    train_loss = 0.0
    
    n_batches = len(dataloader)
    
    # Initialize progress bar
    progress_bar = tqdm(dataloader, desc="Training", total=n_batches)

    total_clap_loss = 0.0
    total_dwd_loss = 0.0
    loss_fn1 = CLAP_loss()
    loss_fn2 = DeepWordDiscriminationLoss()
    for batch, data in enumerate(progress_bar):
    # Send data to target device
        melspecgram = data[0].to(device)
        anc_sequences = data[1].to(device)        
        melspecgram_lengths = data[2].to('cpu')
        word_labels = data[3].to(device)
        anc_sequences_lengths =data[4].to('cpu')
        
        pos_acoustic_embeddings, pos_text_embeddings, _ = model(melspecgram,melspecgram_lengths, anc_sequences,anc_sequences_lengths)
       

        N=melspecgram.size(0)//config.number_examples_wer_word
        
        P = config.number_examples_wer_word
        
        x_labels = torch.arange(N).repeat_interleave(P)
        y_labels = torch.arange(N).repeat_interleave(P)

        # Reshape to (32, 4, D)
        x_grouped = pos_acoustic_embeddings.view(N, P, -1)  # [num_classes, num_views, dim]
        y_grouped = pos_text_embeddings.view(N, P, -1)

        # Slice each view (total 4 views)
        x_views = [x_grouped[:, i, :] for i in range(P)]  # List of 4 tensors, each [32, D]
        y_views = [y_grouped[:, i, :] for i in range(P)]


        view_loss=[]
        for i in range(P):
            logits = model.logit_scale.exp() * (y_views[i]@x_views[i].T)
            view_loss.append(loss_fn1(x_views[i],logits ))
        clap_loss = torch.stack(view_loss, dim=0).mean(dim=0)


        
        dwd_loss = loss_fn2(pos_acoustic_embeddings,word_labels)
        
        total_loss = clap_loss*config.wt_clap_loss + dwd_loss*config.wt_dwd_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # Optimization step
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step()  # Update the scheduler to adjust the learning rate
        
        train_loss += total_loss.item()
       
        # Log learning rate to TensorBoard
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            writer.add_scalar('Learning Rate', lr, epoch * n_batches + batch)
        
        # Update progress bar with loss
        total_clap_loss += clap_loss.item()
        total_dwd_loss += dwd_loss.item()
        # Update progress bar with loss
        progress_bar.set_postfix({
            'total_loss': total_loss.item(),
            'dwd_loss': dwd_loss.item()*config.wt_dwd_loss,
            'clap_loss': clap_loss.item()*config.wt_clap_loss
        })
        
    return train_loss/n_batches, total_clap_loss/n_batches , total_dwd_loss/n_batches


def val_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device, 
        config: Config
    ) -> float:
    """
    Perform a single validation step.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    
    val_loss = 0.0

    n_batches = len(dataloader)
    
    # Initialize progress bar
    progress_bar = tqdm(dataloader, desc="Validation", total=n_batches)
    loss_fn1 = CLAP_loss()
    loss_fn2 = DeepWordDiscriminationLoss()
    total_clap_loss = 0.0
    total_dwd_loss = 0.0
    with torch.no_grad():
        for batch, data in enumerate(progress_bar):
        # Send data to target device
            melspecgram = data[0].to(device)
            anc_sequences = data[1].to(device)        
            melspecgram_lengths = data[2].to('cpu')
            word_labels = data[3].to(device)
            anc_sequences_lengths =data[4].to('cpu')
            
            pos_acoustic_embeddings, pos_text_embeddings, _ = model(melspecgram,melspecgram_lengths, anc_sequences,anc_sequences_lengths)
           
            N=melspecgram.size(0)//config.number_examples_wer_word
            
            P = config.number_examples_wer_word
            
            # Reshape to (32, 4, D)
            x_grouped = pos_acoustic_embeddings.view(N, P, -1)  # [num_classes, num_views, dim]
            y_grouped = pos_text_embeddings.view(N, P, -1)

            # Slice each view (total 4 views)
            x_views = [x_grouped[:, i, :] for i in range(P)]  # List of 4 tensors, each [32, D]
            y_views = [y_grouped[:, i, :] for i in range(P)]



            view_loss=[]
            for i in range(P):
                logits = model.logit_scale.exp() * (y_views[i]@x_views[i].T)
                view_loss.append(loss_fn1(x_views[i],logits ))
            clap_loss = torch.stack(view_loss, dim=0).mean(dim=0)


            
            dwd_loss = loss_fn2(pos_acoustic_embeddings,word_labels)
            
            total_loss = clap_loss*config.wt_clap_loss + dwd_loss*config.wt_dwd_loss
            
            val_loss += total_loss.item()
            total_clap_loss += clap_loss.item()
            total_dwd_loss += dwd_loss.item()
            # Update progress bar with loss
            progress_bar.set_postfix({
                'total_loss': total_loss.item(),
                'dwd_loss': dwd_loss.item()*config.wt_dwd_loss,
                'clap_loss': clap_loss.item()*config.wt_clap_loss
            })
        
        return val_loss / n_batches, total_clap_loss/n_batches , total_dwd_loss/n_batches
def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_dir="saved_models",
        loss_fn: torch.nn.Module = nn.TripletMarginLoss(margin=0.5, p=2),
        epochs: int = 5,
        log_dir="logs",
        device: torch.device = torch.device('cpu'),
        config: Config = Config() 
    ):
    """
    Train the model with TensorBoard logging.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        validation_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        checkpoint_dir (str): Directory to save model checkpoints.
        loss_fn (torch.nn.Module): Loss function.
        epochs (int): Number of epochs to train.
        log_dir (str): Directory to save TensorBoard logs.
        device (torch.device): Device to perform computations on.
    """
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    
    min_loss = float('inf')
    
    # Loop through training and validation steps for a number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        train_loss, train_clap_loss, train_dwd_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,  # Pass the writer
            device=device,
            epoch=epoch,  # Pass the current epoch number
            config = config
        )
        
        # Validation
        val_loss, val_clap_loss, val_dwd_loss = val_step(
            model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            device=device, 
            config = config
        )
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_clap_loss: {train_clap_loss:.4f} | "
            f"train_dwd_loss: {train_dwd_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_clap_loss: {val_clap_loss:.4f} | "
            f"val_dwd_loss: {val_dwd_loss:.4f} | "
        )

        # Log metrics to TensorBoard

        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        writer.add_scalar('Loss/train_clap_loss', train_clap_loss, epoch)
        writer.add_scalar('Loss/train_dwd_loss', train_dwd_loss, epoch)
        writer.add_scalar('Loss/train_clap_loss_wt', train_clap_loss*config.wt_clap_loss, epoch)
        writer.add_scalar('Loss/train_dwd_loss_wt', train_dwd_loss*config.wt_dwd_loss, epoch)
        writer.add_scalar('Loss/val_clap_loss', val_clap_loss, epoch)
        writer.add_scalar('Loss/val_dwd_loss', val_dwd_loss, epoch)
        writer.add_scalar('Loss/val_clap_loss_wt', val_clap_loss*config.wt_clap_loss, epoch)
        writer.add_scalar('Loss/val_dwd_loss_wt', val_dwd_loss*config.wt_dwd_loss, epoch)


        # Save model checkpoint
        if min_loss > val_loss:
            min_loss = val_loss
            model_save_path = os.path.join(checkpoint_dir, f'NAW_{epoch}.pt')
            torch.save(copy.deepcopy(model).state_dict(), model_save_path)
        
    # Close the TensorBoard writer
    writer.close()
