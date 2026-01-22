import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransLayer(nn.Module):
    """
    Temporal-Weather-Aware Translation Layer.
    Calculates the distance g(l_i, r_i, l_j) = || e_li + e_ti + e_wi - e_lj ||^2
    """
    def __init__(self):
        super(TransLayer, self).__init__()

    def forward(self, poi_emb_head, relation_emb, dummy_weather, poi_emb_tail):
        """
        Args:
            poi_emb_head: [batch_size, embed_dim]
            relation_emb: [batch_size, embed_dim] (Combined Time + Weather + Season)
            dummy_weather: Ignored (kept for compatibility if needed, or passed as None)
            poi_emb_tail: [batch_size, embed_dim]
        Returns:
            distance: [batch_size]
        """
        # relation r_i is passed fully formed
        
        # diff = h + r - t
        diff = poi_emb_head + relation_emb - poi_emb_tail
        
        # Distance (L2 norm squared)
        distance = torch.sum(diff ** 2, dim=1)
        return distance

    def predict_likelihood(self, poi_emb_head, time_emb, weather_emb, poi_emb_tail):
        """
        Equation (3): s(...) = exp(-g(...))
        """
        distance = self.forward(poi_emb_head, time_emb, weather_emb, poi_emb_tail)
        return torch.exp(-distance)

    def get_top_k_neighbors(self, poi_emb_head, time_emb, weather_emb, all_poi_embs, k=5):
        """
        Select top k POIs as adjacent nodes based on likelihood.
        """
        # relation r_i = t_i + w_i
        relation_emb = time_emb + weather_emb
        
        # query = h + r
        query = poi_emb_head + relation_emb # [batch_size, dim]
        
        # Calculate distance to ALL POIs
        # cdist calculates L2 distance. Ranking is same as Squared L2.
        # [batch_size, num_pois]
        dists = torch.cdist(query, all_poi_embs, p=2)
        
        # Top K (Smallest distance = Highest likelihood)
        _, indices = dists.topk(k, largest=False)
        
        return indices


class UserGraphLayer(nn.Module):
    """
    One layer of Message Passing for User-POI Interaction Graph (Eq 4-7).
    Updates Embeddings for User, POI, Time, Weather based on their interactions.
    """
    def __init__(self, embed_dim):
        super(UserGraphLayer, self).__init__()
        self.embed_dim = embed_dim
        
        # ユーザー集約はk-NNを使用するため、パラメータは不要
        # self.W_user = nn.Linear(embed_dim, embed_dim)
        # self.activation = nn.ReLU()

    def forward(self, 
                user_emb, poi_emb, time_emb, weather_emb, season_emb,
                day_weather_emb, month_weather_emb,
                edges, e_W_per_edge):
        """
        Args:
            user_emb: [num_users, dim]
            poi_emb:  [num_pois, dim]
            ...
            edges: Dict containing interaction tensors.
            e_W_per_edge: [num_edges, dim] (Pre-computed Multi-granular Weather Context)
        """
        u_idx = edges['user_idxs']
        p_idx = edges['poi_idxs']
        t_idx = edges['time_idxs']
        s_idx = edges['season_idxs']
        w_now_idx = edges['weather_now_idxs'] # Primary weather index
        w_day_idx = edges['weather_day_idxs']
        w_yesterday_idx = w_day_idx[:, 0]
        w_today_idx = w_day_idx[:, 1]
        w_tomorrow_idx = w_day_idx[:, 2]
        w_month_idx = edges['weather_month_idxs']
        
        # 1. Update User Embeddings (Eq 12)
        # e_u = Agg(e_p + e_t + e_Wm)
        # Message:
        msg_for_user = poi_emb[p_idx] + time_emb[t_idx] + e_W_per_edge
        
        new_user_emb = torch.zeros_like(user_emb)
        user_counts = torch.zeros(user_emb.size(0), 1, device=user_emb.device)
        
        new_user_emb.index_add_(0, u_idx, msg_for_user)
        user_counts.index_add_(0, u_idx, torch.ones((u_idx.size(0), 1), device=user_counts.device))
        
        new_user_emb = new_user_emb / (user_counts + 1e-9)
        
        # 2. Update POI Embeddings (Eq 13)
        # e_lj = Agg(e_u + e_t + e_Wm)
        msg_for_poi = user_emb[u_idx] + time_emb[t_idx] + e_W_per_edge
        new_poi_emb = torch.zeros_like(poi_emb)
        poi_counts = torch.zeros(poi_emb.size(0), 1, device=poi_emb.device)
        
        new_poi_emb.index_add_(0, p_idx, msg_for_poi)
        poi_counts.index_add_(0, p_idx, torch.ones((p_idx.size(0), 1), device=poi_counts.device))
        new_poi_emb = new_poi_emb / (poi_counts + 1e-9)

        # 3. Update Time Embeddings (Eq 14)
        # e_tk = Agg(e_u + e_lj) (Note: original Eq 7 was just u+l, Eq 14 is same)
        msg_for_time = user_emb[u_idx] + poi_emb[p_idx]
        new_time_emb = torch.zeros_like(time_emb)
        time_counts = torch.zeros(time_emb.size(0), 1, device=time_emb.device)
        
        new_time_emb.index_add_(0, t_idx, msg_for_time)
        time_counts.index_add_(0, t_idx, torch.ones((t_idx.size(0), 1), device=time_counts.device))
        new_time_emb = new_time_emb / (time_counts + 1e-9)

        # 4. Update Season Embeddings (Eq 15)
        msg_for_season = user_emb[u_idx] + poi_emb[p_idx]
        new_season_emb = torch.zeros_like(season_emb)
        season_counts = torch.zeros(season_emb.size(0), 1, device=season_emb.device)
        
        new_season_emb.index_add_(0, s_idx, msg_for_season)
        season_counts.index_add_(0, s_idx, torch.ones((s_idx.size(0), 1), device=season_counts.device))
        new_season_emb = new_season_emb / (season_counts + 1e-9)

        # 5. Update Weather Embeddings (Eq 16)
        # e_Wm = Agg(e_u + e_lj). 
        # Wait, Eq 16 is for e_Wm. Is it the Consolidated Weather or the Raw Weather?
        # Usually we update the raw components so the next iteration's consolidated weather is better.
        # Or do we update the underlying w_now index?
        # The equation says "e_Wm = ...". m usually indexes the weather type (Light Rain).
        # So we update the base `weather_embedding`.
        msg_for_weather = user_emb[u_idx] + poi_emb[p_idx]
        new_weather_emb = torch.zeros_like(weather_emb)
        weather_counts = torch.zeros(weather_emb.size(0), 1, device=weather_emb.device)
        
        new_weather_emb.index_add_(0, w_now_idx, msg_for_weather)
        weather_counts.index_add_(0, w_now_idx, torch.ones((w_now_idx.size(0), 1), device=weather_counts.device))
        new_weather_emb = new_weather_emb / (weather_counts + 1e-9)
        
        # Apply Aggregation (Neighborhood Smoothing)
        new_user_emb = self.aggregation(new_user_emb, k=5)

        # 6. Update Day Weather Embeddings
        # w_day_idx: [num_edges, 3]. Flatten to update.
        msg_for_day = user_emb[u_idx] + poi_emb[p_idx] # Context
        
        # Repeat context for each of the 3 days
        msg_expanded = msg_for_day.repeat_interleave(3, dim=0) # [3*num_edges, dim]
        day_idx_expanded = w_day_idx.view(-1) # [3*num_edges]
        
        new_day_emb = torch.zeros_like(day_weather_emb)
        day_counts = torch.zeros(day_weather_emb.size(0), 1, device=day_weather_emb.device)
        
        new_day_emb.index_add_(0, day_idx_expanded, msg_expanded)
        day_counts.index_add_(0, day_idx_expanded, torch.ones((day_idx_expanded.size(0), 1), device=day_counts.device))
        new_day_emb = new_day_emb / (day_counts + 1e-9)

        # 7. Update Month Weather Embeddings
        msg_for_month = user_emb[u_idx] + poi_emb[p_idx]
        new_month_emb = torch.zeros_like(month_weather_emb)
        month_counts = torch.zeros(month_weather_emb.size(0), 1, device=month_weather_emb.device)
        
        new_month_emb.index_add_(0, w_month_idx, msg_for_month)
        month_counts.index_add_(0, w_month_idx, torch.ones((w_month_idx.size(0), 1), device=month_counts.device))
        new_month_emb = new_month_emb / (month_counts + 1e-9)

        return new_user_emb, new_poi_emb, new_time_emb, new_weather_emb, new_season_emb, new_day_emb, new_month_emb

    def aggregation(self, user_emb, k=5):
        dist = torch.cdist(user_emb, user_emb, p=2)
        _, indices = dist.topk(k=k+1, largest=False)
        neighbor_embs = user_emb[indices]
        avg_emb = neighbor_embs.mean(dim=1)
        
        # Simple Average (Simple方式)
        return avg_emb

    def aggregate_messages(self, target_node_emb, neighbor_emb_list): #近傍集約
        # neighbor_emb_list: list of tensors to sum/mean
        # Logic: e_new = Mean(sum(neighbors))
        total = sum(neighbor_emb_list)
        return total # In real GNN, this is normalized by degree


class SpatialTemporalSelfAttention(nn.Module):
    """
    Self-Attention with Spatial and Temporal distance bias.
    Eq (9)-(12)
    """
    def __init__(self, embed_dim, dropout=0.1):
        super(SpatialTemporalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, trajectory_emb, time_seq, lat_seq, lon_seq, mask=None):
        """
        trajectory_emb: [batch, seq_len, dim] (Input E(u))
        time_seq:       [batch, seq_len] (Raw timestamps or time indices representing scalar time)
        lat_seq, lon_seq: [batch, seq_len] (GPS coordinates)
        mask:           [batch, seq_len] (1 for valid, 0 for padding)
        """
        Q = self.W_Q(trajectory_emb)
        K = self.W_K(trajectory_emb)
        V = self.W_V(trajectory_emb)

        # Basic Attention Score: QK^T / sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Calculate Pairwise Distances for Bias
        # Time distance: |t_j - t_i|
        # [batch, seq, 1] - [batch, 1, seq] -> [batch, seq, seq]
        t_i = time_seq.unsqueeze(2)
        t_j = time_seq.unsqueeze(1)
        d_t = torch.abs(t_j - t_i).float()

        # Spatial distance: Haversine
        # We need a tensorized haversine.
        lat_i = lat_seq.unsqueeze(2)
        lon_i = lon_seq.unsqueeze(2)
        lat_j = lat_seq.unsqueeze(1)
        lon_j = lon_seq.unsqueeze(1)
        
        # d_s calculation (Haversine approx or simplified for gradient)
        # Using simple Euclidean on coords often sufficient for small area, 
        # but paper specifies Haversine. 
        # For simplicity in 'forward' we implement a differentiable approximation or call the utility if tensorized.
        # Let's implement a quick tensor version here.
        d_s = self.tensor_haversine(lat_i, lon_i, lat_j, lon_j)

        # Normalize/Scale distances if necessary? Paper sums them: a'_ij = a_ij + d_s + d_t
        # Usually distances are large, attention scores are small. Maybe they need scaling or log?
        # Paper Eq 10 just adds them. Assuming d_t and d_s are strictly defined/normalized.
        # We will add them as bias. NOTE: Large distance usually reduces attention? 
        # Paper Eq 10 says: a' = a + d_s + d_t.
        # WAIT: If d_s is large (far away), we usually want *less* attention.
        # But if it's a "transition pattern", maybe we attend to far things?
        # Actually Eq 13 uses QK^T + d_s + d_t. 
        # If the paper implies "distance influences transition", usually short distance = high prob.
        # But in attention, higher score = higher weight.
        # If they just add +distance, it means FURTHER items get MORE attention. This seems counter-intuitive for "Next POI".
        # However, checking Eq 3: exp(-distance). 
        # Let's verify Eq 10 in paper... "a'_{ij} = a_{ij} + d_s + d_t"
        # If they are positive, it increases attention. 
        # Let's stick to the equation but keeps in mind it might need a negative sign if results are weird.
        # For now, implementing strictly as + d_s + d_t.
        
        scores = scores - d_s - d_t

        if mask is not None:
            # mask: [batch, seq] -> [batch, 1, seq]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output

    def tensor_haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c
