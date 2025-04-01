import torch
import torch.nn.functional as F


def loss_func(batch, similarity_threshold=0.65, margin=0.0):
    batch_size = batch.shape[0]
    batch = F.normalize(batch, p=2, dim=1)  
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(batch, batch.T)  
    
    # Mask out self-similarity
    mask = torch.eye(batch_size, device=batch.device).bool()
    similarity_matrix.masked_fill_(mask, -1)  # Set diagonal to -1 so it's not considered a neighbor
    
    # Find nearest neighbors based on similarity threshold
    positive_mask = similarity_matrix >= similarity_threshold  # Positive pairs
    negative_mask = similarity_matrix < similarity_threshold  # Negative pairs
    
    # Loss computation
    if positive_mask.any():
        positive_loss = (1 - similarity_matrix)[positive_mask].mean() 
    else: 
        positive_loss = torch.tensor(0.0, device=batch.device)

    if negative_mask.any():
        negative_loss = F.relu(similarity_matrix[negative_mask] - margin).mean() 
    else:
        negative_loss = torch.tensor(0.0, device=batch.device)

    loss = positive_loss + negative_loss
    return loss