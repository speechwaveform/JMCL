import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_dist(a, b):
    return 1 - (a * b).sum(dim=1)

class MultiObjectiveContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self,  fx_pos, fx_neg, gc_pos, gc_neg):

        d_pos = cosine_dist(fx_pos, gc_pos)
        d_neg_text = cosine_dist(gc_pos, gc_neg)
        d_neg_audio = cosine_dist(fx_neg, gc_pos)
        d_audio_audio = cosine_dist(fx_pos, fx_neg)

        obj0 = F.relu(self.margin + d_pos - cosine_dist(fx_pos, gc_neg))
        obj1 = F.relu(self.margin + d_pos - d_neg_text)
        obj2 = F.relu(self.margin + d_pos - d_neg_audio)
        obj3 = F.relu(self.margin + d_pos - d_audio_audio)

        if self.reduction == 'mean':
            return {
                'obj0': obj0.mean(),
                'obj1': obj1.mean(),
                'obj2': obj2.mean(),
                'obj3': obj3.mean()
            }
        elif self.reduction == 'sum':
            return {
                'obj0': obj0.sum(),
                'obj1': obj1.sum(),
                'obj2': obj2.sum(),
                'obj3': obj3.sum()
            }
        return {
            'obj0': obj0,
            'obj1': obj1,
            'obj2': obj2,
            'obj3': obj3
        }




class CLAP_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, logits ):
        
        labels = torch.arange(embeddings.size(0), device=embeddings.device)
           
        loss_t2a = F.cross_entropy(logits, labels)
        loss_a2t = F.cross_entropy(logits.T, labels)
        clap_loss =  0.5 * (loss_t2a + loss_a2t)
        
        return clap_loss

class DeepWordDiscriminationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, word_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Deep Word Discrimination Loss (DWD) from embeddings and labels.

        Args:
            embeddings: (B, D) tensor of normalized acoustic word embeddings
            word_labels: (B,) tensor of class indices (0 to N_word - 1)

        Returns:
            Scalar tensor: DWD loss (L_sm + L_cc)
        """
        device = embeddings.device
        B, D = embeddings.size()

        # Normalize input embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # ---- Intra-class centroids (excluding self) ----
        same_label = word_labels.unsqueeze(0) == word_labels.unsqueeze(1)  # (B, B)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)          # (B, B)
        mask = same_label & eye_mask                                       # (B, B)

        class_sizes = mask.sum(dim=1, keepdim=True).clamp(min=1)
        centroids = (mask.float() @ embeddings) / class_sizes              # (B, D)
        centroids = F.normalize(centroids, dim=1)

        # ---- Softmax-based Loss (L_sm) ----
        unique_labels, label_to_class = torch.unique(word_labels, return_inverse=True)
        N_word = unique_labels.size(0)

        # Compute centroids for each class
        class_mask = word_labels.unsqueeze(1) == unique_labels.unsqueeze(0)  # (B, N_word)
        class_counts = class_mask.sum(dim=0, keepdim=True).clamp(min=1)
        class_centroids = (class_mask.float().T @ embeddings) / class_counts.T  # (N_word, D)
        class_centroids = F.normalize(class_centroids, dim=1)

        # Cosine similarity matrix (B, N_word)
        sim_matrix = embeddings @ class_centroids.T

        # Cross-entropy-like softmax loss
        log_probs = F.log_softmax(sim_matrix, dim=1)
        L_sm = -log_probs[torch.arange(B, device=device), label_to_class].mean()

        # ---- Contrastive Centroid Loss (L_cc) ----
        sim_pos = F.cosine_similarity(embeddings, centroids, dim=1)  # (B,)
        sim_matrix_neg = sim_matrix.clone()
        sim_matrix_neg[torch.arange(B), label_to_class] = float('-inf')
        sim_neg = sim_matrix_neg.max(dim=1).values  # max similarity to other classes

        L_cc = ((1 - sim_pos) + sim_neg).mean()

        return L_sm + L_cc

