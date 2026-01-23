import torch
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Result is in kilometers.
    """
    R = 6371  # Earth radius in km

    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    
    a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c

def accuracy_at_k(predictions, ground_truth, k=10):
    """
    Calculate Accuracy@K.
    predictions: [batch_size, num_pois] - Scores or probabilities
    ground_truth: [batch_size] - Indices of target POIs
    """
    # Get top k indices: [batch_size, k]
    _, topk_indices = torch.topk(predictions, k, dim=1)
    
    # Expand ground_truth to match topk shape: [batch_size, 1]
    ground_truth_expanded = ground_truth.unsqueeze(1)
    
    # Check if ground truth is in top k
    hits = (topk_indices == ground_truth_expanded).sum().item()
    
    return hits / ground_truth.size(0)

def ndcg_at_k(predictions, ground_truth, k=10):
    """
    Calculate nDCG@K (Normalized Discounted Cumulative Gain).
    Assumes single ground truth per user (IDCG=1).
    predictions: [batch_size, num_pois]
    ground_truth: [batch_size]
    """
    # Get top k indices: [batch_size, k]
    _, topk_indices = torch.topk(predictions, k, dim=1)
    
    # ground_truth: [batch_size, 1]
    ground_truth_expanded = ground_truth.unsqueeze(1)
    
    # [batch_size, k] boolean mask
    hits = (topk_indices == ground_truth_expanded)
    
    # nonzero returns [hit_index, col_index] where col_index is rank (0-based)
    hits_indices = hits.nonzero(as_tuple=False)
    
    if hits_indices.size(0) == 0:
        return 0.0
        
    # col_index is the 0-based rank in the top-k list
    ranks = hits_indices[:, 1].float()
    
    # DCG = sum(1 / log2(rank + 1)) where rank is 1-based
    # So here rank (1-based) = ranks (0-based) + 1
    # denominator = log2((ranks + 1) + 1) = log2(ranks + 2)
    # Since IDCG=1 for single target, nDCG = DCG
    
    scores = 1.0 / torch.log2(ranks + 2.0)
    
    return scores.sum().item() / ground_truth.size(0)

def mrr(predictions, ground_truth):
    """
    Calculate Mean Reciprocal Rank (MRR).
    """
    # Sort predictions (descending) to get ranks
    # We want the rank of the ground truth
    # optimization: only sort up to a reasonable number if K is small, but for MRR we usually need full sort or at least until the hit
    # For full MRR, we can use argsort or topk(num_pois)
    
    # Using sort for simplicity on standard datasets
    _, sorted_indices = torch.sort(predictions, descending=True, dim=1)
    
    # Find where the ground truth is
    # ground_truth: [batch_size]
    # sorted_indices: [batch_size, num_pois]
    
    hits = (sorted_indices == ground_truth.unsqueeze(1)).nonzero(as_tuple=False)
    
    # hits[:, 1] contains the index (rank-1) in the sorted list
    ranks = hits[:, 1].float() + 1
    reciprocal_ranks = 1.0 / ranks
    
    return reciprocal_ranks.mean().item()
