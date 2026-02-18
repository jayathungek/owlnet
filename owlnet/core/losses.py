import torch
import torch.nn.functional as F


def basic_loss(batch, similarity_threshold=0.65, margin=0.0):
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

    
class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features):
        """Compute loss for model. It is essentially an
        implementation of SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=features.dtype).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isfinite(log_prob)):
        #     print("Inf or NaN after logprob")
        #     print(log_prob)
        #     print(exp_logits)
        #     print(logits_mask)
        #     exit()

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        # if torch.any(torch.isnan(mean_log_prob_pos)):
        #     print("NaN after logprob")
        #     exit()

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss   