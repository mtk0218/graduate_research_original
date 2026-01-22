import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import TransLayer, UserGraphLayer, SpatialTemporalSelfAttention

class TWTransNet(nn.Module):
    def __init__(self, num_users, num_pois, embed_dim, num_times, num_now_weathers, num_day_weathers, num_month_weathers, dropout=0.1):
        super(TWTransNet, self).__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.embed_dim = embed_dim

        # 1. Embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.poi_embedding = nn.Embedding(num_pois, embed_dim)
        self.time_embedding = nn.Embedding(num_times, embed_dim)
        self.now_weather_embedding = nn.Embedding(num_now_weathers, embed_dim)
        self.day_weather_embedding = nn.Embedding(num_day_weathers, embed_dim) # Daily weather (Sunny, Cloudy, Rainy, Snowy)
        self.month_weather_embedding = nn.Embedding(num_month_weathers, embed_dim) # Monthly weather (Sunny, Cloudy, Rainy, Snowy)
        self.season_embedding = nn.Embedding(4, embed_dim) # Spring, Summer, Autumn, Winter
        
        # Multi-granularity Weather Aggregation (Eq 9, 10)
        # W_d: 3d -> d
        self.W_day_agg = nn.Linear(3 * embed_dim, embed_dim) 
        self.b_d = nn.Parameter(torch.zeros(embed_dim))
        
        # W_w: 3d -> d (Eq 10: [e_now; e_day; e_season] -> e_W)
        self.W_weather_agg = nn.Linear(3 * embed_dim, embed_dim)
        self.b_w = nn.Parameter(torch.zeros(embed_dim))
        
        # Eq (8) POI Aggregation Parameters
        self.W_poi_agg = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()

        # 2. Modules
        self.trans_layer = TransLayer()
        self.gnn_layer = UserGraphLayer(embed_dim) # Can be stacked
        self.st_attention = SpatialTemporalSelfAttention(embed_dim, dropout=dropout)

        # 3. Prediction Layer Parameters
        self.W_2 = nn.Linear(embed_dim, 1, bias=False) # Used in Eq 13? Or just a vector? 
        # Eq 13: ... W_2. W_2 is (n x 1)? No, usually it projects to scalar.
        # "W2 is a parameter matrix... size (nx1)" -> This implies weighting the n items in trajectory?
        # Actually standard attention prediction usually is: Score(Target) = Attention(History, Target).
        # Let's interpret Eq 13: 
        # p_li = Softmax( (QK^T + ds + dt) W2 )
        # It looks like it computes a score for a candidate l_i by attending to history.
        
        self.init_weights()

    def init_weights(self): # Xavier initialization(一様分布の一つ)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)
        nn.init.xavier_uniform_(self.now_weather_embedding.weight)
        nn.init.xavier_uniform_(self.day_weather_embedding.weight)
        nn.init.xavier_uniform_(self.month_weather_embedding.weight)
        nn.init.xavier_uniform_(self.season_embedding.weight)
        nn.init.xavier_uniform_(self.W_day_agg.weight)
        nn.init.xavier_uniform_(self.W_weather_agg.weight)

    def forward_gnn(self, edges):
        """
        Execute GNN updates to refine embeddings based on interactions.
        """
        u_emb = self.user_embedding.weight
        p_emb = self.poi_embedding.weight
        t_emb = self.time_embedding.weight
        w_now_emb = self.now_weather_embedding.weight
        w_day_emb = self.day_weather_embedding.weight
        w_month_emb = self.month_weather_embedding.weight
        s_emb = self.season_embedding.weight

        # Prepare Multi-granular Weather & Category Context
        # Need: weather_day_seq (idx), season (idx), category (idx) in 'edges'
        # edges should contain:
        # 'weather_day_seq': [num_edges, 3] (d-1, d, d+1)
        # 'season_idxs': [num_edges]
        # 'category_idxs': [num_edges] (Category of POI l)
        
        e_now = w_now_emb[edges['weather_now_idxs']] # [num_edges, dim]
        
        # Eq 9: e_day
        w_day_seq_idx = edges['weather_day_idxs'] # [num_edges, 3]
        w_yesterday = w_day_emb[w_day_seq_idx[:, 0]]
        w_today = w_day_emb[w_day_seq_idx[:, 1]]
        w_tomorrow = w_day_emb[w_day_seq_idx[:, 2]]
        
        # [num_edges, 3*dim]
        concat_day = torch.cat([w_yesterday, w_today, w_tomorrow], dim=1)
        e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
        
        # Eq 10: e_W
        e_month = w_month_emb[edges['weather_month_idxs']]
        concat_weather = torch.cat([e_now, e_day, e_month], dim=1)
        e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w

        season_idx = edges['season_idxs']        # [num_edges]
        
        # Pass constructed embeddings to GNN
        new_u, new_p, new_t, new_w, new_s, new_day, new_month = self.gnn_layer(
            u_emb, p_emb, t_emb, w_now_emb, s_emb, w_day_emb, w_month_emb,
            edges, e_W_per_edge=e_W
        )
        
        return new_u, new_p, new_t, new_w, new_s, new_day, new_month

    def calc_translation_loss(self, head_idx, r_time_idx, r_weather_idx, tail_idx, neg_tail_idx, 
                              r_season_idx, r_day_seq_idx, r_month_idx,
                              u_emb_all=None, p_emb_all=None, t_emb_all=None, w_emb_all=None,
                              s_emb_all=None, d_emb_all=None, m_emb_all=None):
        """
        Calculate Translation Loss (max-margin).
        L_T = sum( max(0, g(pos) + gamma - g(neg)) )
        
        Args:
           ..._emb_all: Refined embeddings from GNN. If None, use raw lookups.
        """
        if p_emb_all is not None:
            # Look up from refined matrices
            h = p_emb_all[head_idx]
            t = p_emb_all[tail_idx]
            n_t = p_emb_all[neg_tail_idx]
            t_time = t_emb_all[r_time_idx]
            
            # Refined components for Weather Context construction
            r_w_now = w_emb_all[r_weather_idx]
            
            # Use refined day weather embedding if available
            if d_emb_all is not None:
                r_w_minus = d_emb_all[r_day_seq_idx[:, 0]]
                r_w_curr = d_emb_all[r_day_seq_idx[:, 1]]
                r_w_plus = d_emb_all[r_day_seq_idx[:, 2]]
            else:
                r_w_minus = self.day_weather_embedding(r_day_seq_idx[:, 0])
                r_w_curr = self.day_weather_embedding(r_day_seq_idx[:, 1])
                r_w_plus = self.day_weather_embedding(r_day_seq_idx[:, 2])
            
            # Recalculate e_day, e_W using refined parts
            concat_day = torch.cat([r_w_minus, r_w_curr, r_w_plus], dim=1)
            e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
            
            if s_emb_all is not None:
                e_season = s_emb_all[r_season_idx]
            else:
                e_season = self.season_embedding(r_season_idx) # Static season embedding
            
            # Month embedding
            if m_emb_all is not None:
                e_month = m_emb_all[r_month_idx]
            else:
                e_month = self.month_weather_embedding(r_month_idx)

            # Eq 10: e_W includes month now
            concat_weather = torch.cat([r_w_now, e_day, e_month], dim=1)
            e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w

        else:
            # Fallback to raw embeddings
            h = self.poi_embedding(head_idx)
            t = self.poi_embedding(tail_idx)
            n_t = self.poi_embedding(neg_tail_idx)
            
            t_time = self.time_embedding(r_time_idx)
            
            # Compute Raw e_W
            r_w_now = self.weather_embedding(r_weather_idx)
            r_w_minus = self.day_weather_embedding(r_day_seq_idx[:, 0])
            r_w_curr = self.day_weather_embedding(r_day_seq_idx[:, 1])
            r_w_plus = self.day_weather_embedding(r_day_seq_idx[:, 2])
            
            concat_day = torch.cat([r_w_minus, r_w_curr, r_w_plus], dim=1)
            e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
            
            e_season = self.season_embedding(r_season_idx)
            e_month = self.month_weather_embedding(r_month_idx)

            concat_weather = torch.cat([r_w_now, e_day, e_month], dim=1)
            e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w
            
        # Calculate distances
        # Context is t + W
        # Simplified TransLayer forward(h, relation, dummy, t)
        relation_emb = t_time + e_W
        pos_dist = self.trans_layer(h, relation_emb, None, t)
        neg_dist = self.trans_layer(h, relation_emb, None, n_t)
        
        gamma = 1.0 # Margin, hyperparameter
        loss = torch.relu(pos_dist + gamma - neg_dist)
        return loss.mean()

    def enhance_poi_rep(self, traj_emb):
        """
        Eq (8): Aggregate User History POIs (Trajectory) -> Enhanced Representation.
        e = sigma(W * Mean(e_p))
        """
        # Mean pooling over sequence dimension
        # traj_emb: [batch, seq, dim]
        avg_emb = traj_emb.mean(dim=1) # [batch, dim]
        
        # Weighted Aggregation + Activation
        enhanced_rep = self.activation(self.W_poi_agg(avg_emb))
        
        return enhanced_rep
        
    def predict(self, user_idx, traj_poi_idx, traj_time_vec, traj_lat, traj_lon, candidate_poi_indices, 
                cand_lat, cand_lon, current_time, mask=None, p_emb_all=None, enhanced_history=None):
        """
        ...
        Args:
            p_emb_all: Refined POI embeddings.
            enhanced_history: [batch, dim] (Eq 8 Result)
        """
        batch_size, seq_len = traj_poi_idx.shape
        num_cands = candidate_poi_indices.shape[1]
        
        # 1. Get Trajectory Embeddings
        if p_emb_all is not None:
             traj_emb = p_emb_all[traj_poi_idx] # [batch, seq, dim]
        else:
             traj_emb = self.poi_embedding(traj_poi_idx) # [batch, seq, dim]
        
        # 2. Self-Attention to get 'Updated Trajectory Embedding' R(u)
        # Eq 12: R(u) = Softmax(A') V_i
        # We use our module
        history_rep = self.st_attention(traj_emb, traj_time_vec, traj_lat, traj_lon, mask=mask) 
        # history_rep: [batch, seq, dim]
        
        # Incorporate Eq 8 (Global Trajectory Aggregation) if provided
        if enhanced_history is not None:
            # Add global context to each sequence step
            history_rep = history_rep + enhanced_history.unsqueeze(1)
        
        # 3. Prediction for Candidates
        # Eq 13: p = Softmax( (Q K^T + d_s + d_t) W2 )
        # Q: Candidate embedding [batch, num_cands, dim]
        # K: History rep [batch, seq, dim]
        
        cand_emb = self.poi_embedding(candidate_poi_indices) # [batch, n_cand, dim]
        
        # Attention scores between Candidates (Q) and History (K)
        # [batch, n_cand, seq]
        raw_scores = torch.matmul(cand_emb, history_rep.transpose(1, 2)) / (self.embed_dim ** 0.5)
        
        # Add Spatial/Temporal Distances between Candidates and History
        # We need to broadcast
        # cand: [batch, n_cand, 1] vs hist: [batch, 1, seq]
        
        # Time distance
        # current_time: [batch] -> dist to history times? Or candidate time?
        # Usually candidate time is typically "next time step". 
        # Let's assume current_time is the target time.
        t_target = current_time.unsqueeze(1).unsqueeze(2) # [batch, 1, 1]
        t_hist = traj_time_vec.unsqueeze(1) # [batch, 1, seq]
        d_t = torch.abs(t_target - t_hist) # [batch, 1, seq] (Broadcasted)
        
        # Spatial distance
        # cand: [batch, n_cand] -> lat/lon
        c_lat = cand_lat.unsqueeze(2) # [batch, n_cand, 1]
        c_lon = cand_lon.unsqueeze(2)
        h_lat = traj_lat.unsqueeze(1) # [batch, 1, seq]
        h_lon = traj_lon.unsqueeze(1)
        
        d_s = self.st_attention.tensor_haversine(c_lat, c_lon, h_lat, h_lon)
        
        # Combine
        combined_scores = raw_scores + d_s + d_t # [batch, n_cand, seq]
        
        # Apply W2 parameter?
        # Eq 13 says "... W2". If W2 is (n x 1), it sums over the sequence dimension?
        # Yes, usually we aggregate the history attention into a single score for the candidate.
        # "W2 is a parameter matrix (n x 1)" -> It learns how to weight the sequence items?
        # That requires fixed sequence length 'n'.
        # Alternatively, it could be a mean or sum. 
        # Let's assume weighted sum over sequence dim.
        
        # For variable length, usually we use an attention vector or just sum. 
        # If we stick to fixed length n=100 (from paper settings), we can use Linear(seq_len, 1).
        # But handling padding is tricky.
        # Let's try simple sum or mean for now to be robust.
        
        # [batch, n_cand, seq] -> [batch, n_cand]
        final_scores = combined_scores.mean(dim=2) 
        
        return final_scores
