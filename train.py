import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from TWTransNet.twtransnet import TWTransNet
from TWTransNet.data_loader import CheckinDataset
from TWTransNet.utils import accuracy_at_k, mrr, ndcg_at_k
from graph_utils import build_interaction_graph
from datetime import datetime, timedelta
import sys
import os


# Global weather list to ensure consistency across functions
REALTIME_WEATHER_LIST = ["Fair", "Partly Cloudy", "Cloudy", "Mostly Cloudy", "Light Rain", "Rain", "Heavy Rain", "Light Snow", "Snow", "Heavy Snow", "Fair / Windy", "Partly Cloudy / Windy", "Cloudy / Windy", "Mostly Cloudy / Windy", "Light Rain / Windy", "Rain / Windy", "Heavy Rain / Windy", "Light Snow / Windy", "Snow / Windy", "Heavy Snow / Windy"]
DAY_WEATHER_LIST = ["Sunny", "Cloudy", "Rainy", "Snowy"]

def generate_mock_data(num_users=20, num_pois=100, min_len=5, max_len=20):
    """
    Generate random trajectories for testing.
    Returns dict: {user_id: [(poi, time, weather, lat, lon), ...]}
    """
    data = {}
    poi_coords = {}
    for u in range(num_users):
        seq_len = np.random.randint(min_len, max_len)
        traj = []
        for _ in range(seq_len):
            poi = np.random.randint(1, num_pois) # 0 is PAD
            time = np.random.rand() * 24
            now_weather_idx = np.random.randint(0, len(REALTIME_WEATHER_LIST))
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            
            # New Attributes
            yesterday_weather_idx = np.random.randint(0, len(DAY_WEATHER_LIST))
            today_weather_idx = np.random.randint(0, len(DAY_WEATHER_LIST))
            tomorrow_weather_idx = np.random.randint(0, len(DAY_WEATHER_LIST))
            month_weather_idx = np.random.randint(0, 9)
            season_idx = np.random.randint(0, 4) #3月~5月が春、6月~8月が夏、9月~11月が秋、12月~2月が冬

            
            traj.append((poi, time, now_weather_idx, lat, lon, yesterday_weather_idx, today_weather_idx, tomorrow_weather_idx, month_weather_idx, season_idx))
            poi_coords[poi] = (float(lat), float(lon))
        data[u] = traj
    return data, poi_coords

def load_real_data(): #新たに作成
    """
    Load real data from file.
    Returns dict: {user_id: [(poi, time, weather, lat, lon), ...]}
    """
    data = {}
    POI_list = []
    user_map = {} # Map raw user_id to 0-based index
    
    # weather_list used from global WEATHER_LIST
    traj = []
    current_user_idx = -1
    last_raw_user_id = None
    
    POI_index = -1
    poi_coords = {} # Map POI Index -> (lat, lon)
    
    with open("boston_1year_checkin_weather.csv", "r", encoding="utf-8-sig") as f:
        for line in f:
            user_id, poi, UTC_time, Timezone, lat, lon, category, country, now_weather, yesterday_weather, today_weather, tomorrow_weather, month_weather_idx, season_idx = line.strip().split(",")
            
            # User ID Remapping
            if user_id not in user_map:
                user_map[user_id] = len(user_map)
            
            u_idx = user_map[user_id]
            
            # Check for user switch
            if last_raw_user_id is None:
                last_raw_user_id = user_id
                current_user_idx = u_idx
            elif last_raw_user_id != user_id:
                # Save previous user's trajectory
                # Use current_user_idx (which corresponds to last_raw_user_id)
                data[current_user_idx] = traj
                traj = []
                last_raw_user_id = user_id
                current_user_idx = u_idx
            
            if poi not in POI_list: # 新しいPOI
                POI_list.append(poi)
                POI_index = len(POI_list) # 1-based index (0 reserved for pad)
            else: # 既存のPOI
                POI_index = POI_list.index(poi) + 1
            
            # Update coords (overwrite is fine as they should be static for same POI)
            poi_coords[POI_index] = (float(lat), float(lon))
            
            try:
                now_weather_index = REALTIME_WEATHER_LIST.index(now_weather)
            except ValueError:  
                print(f"Warning: Unknown weather '{now_weather}', using index 0")
                now_weather_index = 0
            
            try:
                yesterday_weather_index = DAY_WEATHER_LIST.index(yesterday_weather)
            except ValueError:  
                print(f"Warning: Unknown weather '{yesterday_weather}', using index 0")
                yesterday_weather_index = 0
            
            try:
                today_weather_index = DAY_WEATHER_LIST.index(today_weather)
            except ValueError:  
                print(f"Warning: Unknown weather '{today_weather}', using index 0")
                today_weather_index = 0
            
            try:
                tomorrow_weather_index = DAY_WEATHER_LIST.index(tomorrow_weather)
            except ValueError:  
                print(f"Warning: Unknown weather '{tomorrow_weather}', using index 0")
                tomorrow_weather_index = 0
            
            local_time = datetime.strptime(UTC_time, "%a %b %d %H:%M:%S %z %Y") + timedelta(minutes=int(Timezone))
            float_time = local_time.hour + local_time.minute / 60 + local_time.second / 3600
            traj.append((POI_index, float_time, now_weather_index, float(lat), float(lon), yesterday_weather_index, today_weather_index, tomorrow_weather_index, int(month_weather_idx), int(season_idx)))
            
    # Save last user
    if current_user_idx != -1:
        data[current_user_idx] = traj

    return len(data), len(POI_list) + 1, data, poi_coords


def train_epoch(model, dataloader, optimizer, num_pois, edges, poi_coords, lambda_weight=0.1, device='cpu'):
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_trans_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    total_acc10 = 0
    total_mrr = 0
    total_ndcg10 = 0
    
    criterion_rec = nn.CrossEntropyLoss()
    
    # Prepare Coordinate Tensor for Fast Lookup
    # Shape: [num_pois, 2] -> (lat, lon)
    coord_tensor = torch.zeros(num_pois, 2, device=device)
    if poi_coords is not None:
        for idx, (lat, lon) in poi_coords.items():
            if idx < num_pois:
                coord_tensor[idx] = torch.tensor([lat, lon], device=device)
    
    for batch in dataloader:
        # Move keys to device
        user_id = batch['user_id'].to(device)
        traj_poi = batch['traj_poi'].to(device)
        traj_time = batch['traj_time'].to(device)
        traj_now_weather = batch['traj_now_weather'].to(device)
        traj_lat = batch['traj_lat'].to(device)
        traj_lon = batch['traj_lon'].to(device)
        traj_yesterday_weather = batch['traj_yesterday_weather'].to(device)
        traj_today_weather = batch['traj_today_weather'].to(device)
        traj_tomorrow_weather = batch['traj_tomorrow_weather'].to(device)
        traj_month_weather = batch['traj_month_weather'].to(device)
        traj_season = batch['traj_season'].to(device)
        
        target_poi = batch['target_poi'].to(device)
        target_time = batch['target_time'].to(device)
        target_now_weather = batch['target_now_weather'].to(device)
        target_lat = batch['target_lat'].to(device)
        target_lon = batch['target_lon'].to(device)
        target_yesterday_weather = batch['target_yesterday_weather'].to(device)
        target_today_weather = batch['target_today_weather'].to(device)
        target_tomorrow_weather = batch['target_tomorrow_weather'].to(device)
        target_month_weather = batch['target_month_weather'].to(device)
        target_season = batch['target_season'].to(device)
        
        head_poi = batch['head_poi'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # 0. GNN Update (Refine Embeddings)
        # In this implementation, we refine embeddings based on the global graph for each batch.
        # Ideally, this might be done once per epoch if the graph is static, 
        # but since we backprop into embeddings, we do it here (or outside batch loop if full batch).
        # For efficiency with SGD, we do it per batch (though it uses global edges, 
        # gradients will accumulate to used nodes).
        # Or, strictly, we might want to do it outside batch loop? 
        # If we do it inside, we re-compute refinement every batch. 
        # Given "edges" is constant, this is fine, but compute heavy.
        # But this is closest to "training with GNN".
        
        # Forward
        # 4. GNN (Refine Embeddings)
        # Forward
        # 4. GNN (Refine Embeddings)
        # Now returns 5 values (User, POI, Time, Weather, Season)
        # Now returns 7 values (User, POI, Time, Weather, Season, Day, Month)
        refined_u, refined_p, refined_t, refined_w, refined_s, refined_day, refined_month = model.forward_gnn(edges)
        
        # Prepare "refined all" for lookup
        # Since implementation assumes we look up by index from the weight matrix or tensor,
        # we need to pass these refined tensors.
        
        # 1. Recommendation Prediction
        # ...
        
        # Prepare candidates: All POIs [1...num_pois-1]
        cands = torch.arange(1, num_pois).unsqueeze(0).repeat(user_id.size(0), 1).to(device) # [batch, num_pois-1]
        # We need lat/lon for all cands? 
        # For this demo, we generate random lat/lon for cands OR use embedding table if we had static POI data.
        # Since we don't have static POI table in this script, we'll use a placeholder or 
        # just use the embedding (model uses embedding).
        # Wait, model.predict needs cand_lat/lon for Haversine.
        # We'll pass Zeros or Randoms if we don't have static metadata loaded.
        # Ideally: load POI_lat_lon table.
        # Look up coordinates for candidates
        cand_coords = coord_tensor[cands] # [batch, num_pois-1, 2]
        cand_lats = cand_coords[:, :, 0]
        cand_lons = cand_coords[:, :, 1]
        
        # Prediction Scores: [batch, num_cands]
        
        # Calculate Enhanced POI Representation (Eq 8)
        traj_emb_for_agg = refined_p[traj_poi] # Get refined embeddings for history
        enhanced_history = model.enhance_poi_rep(traj_emb_for_agg)
        
        # Pass refined POI embeddings and Enhanced History
        scores = model.predict(user_id, traj_poi, traj_time, traj_lat, traj_lon, 
                               cands, cand_lats, cand_lons, target_time, mask=mask,
                               p_emb_all=refined_p, enhanced_history=enhanced_history)
        
        # Target for CrossEntropy
        # cands are 1-based, target_poi is 1-based. 
        # We need index in 'scores' corresponding to 'target_poi'.
        # If cands = [1, 2, ..., N], then index = target_poi - 1.
        target_idx = target_poi - 1
        
        rec_loss = criterion_rec(scores, target_idx)
        
        # 2. Translation Loss
        # Need Negative Sample for Tail
        # Random neg tail
        neg_tail = torch.randint(1, num_pois, target_poi.shape).to(device)
        
        # Prepare target daily weather seq: [batch, 3]
        target_w_day_seq = torch.stack([target_yesterday_weather, target_today_weather, target_tomorrow_weather], dim=1)
        
        trans_loss = model.calc_translation_loss(head_poi, target_time.long(), target_now_weather, target_poi, neg_tail,
                                                 r_season_idx=target_season,
                                                 r_day_seq_idx=target_w_day_seq,
                                                 r_month_idx=target_month_weather,
                                                 u_emb_all=refined_u, p_emb_all=refined_p, 
                                                 t_emb_all=refined_t, w_emb_all=refined_w,
                                                 s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month)
        
        # Total Loss
        loss = rec_loss + lambda_weight * trans_loss # lambda defined by param
        
        loss.backward()
        optimizer.step()
        
        # --- Metrics Calculation ---
        # Calculate for this batch
        # scores: [batch, num_cands] -> We need to check if num_cands covers all POIs
        # In current implementation, cands is ALL POIs (1..num_pois-1).
        # And target_idx is (target_poi - 1). This aligns correctly.
        
        acc1 = accuracy_at_k(scores, target_idx, k=1)
        acc5 = accuracy_at_k(scores, target_idx, k=5)
        acc10 = accuracy_at_k(scores, target_idx, k=10)
        mrr_val = mrr(scores, target_idx)
        ndcg10_val = ndcg_at_k(scores, target_idx, k=10)
        
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_trans_loss += trans_loss.item()
        total_acc1 += acc1
        total_acc5 += acc5
        total_acc10 += acc10
        total_mrr += mrr_val
        total_ndcg10 += ndcg10_val
        
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    return (total_loss / num_batches, 
            total_rec_loss / num_batches, 
            total_trans_loss / num_batches,
            total_acc1 / num_batches,
            total_acc5 / num_batches,
            total_acc10 / num_batches,
            total_mrr / num_batches,
            total_ndcg10 / num_batches)

def evaluate(model, dataloader, num_pois, edges, poi_coords, lambda_weight=0.1, device='cpu'):
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    total_trans_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    total_acc10 = 0
    total_mrr = 0
    total_ndcg10 = 0
    
    criterion_rec = nn.CrossEntropyLoss()
    
    # Prepare Coordinate Tensor for Fast Lookup
    coord_tensor = torch.zeros(num_pois, 2, device=device)
    if poi_coords is not None:
        for idx, (lat, lon) in poi_coords.items():
            if idx < num_pois:
                coord_tensor[idx] = torch.tensor([lat, lon], device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            user_id = batch['user_id'].to(device)
            traj_poi = batch['traj_poi'].to(device)
            traj_time = batch['traj_time'].to(device)
            traj_now_weather = batch['traj_now_weather'].to(device)
            traj_lat = batch['traj_lat'].to(device)
            traj_lon = batch['traj_lon'].to(device)
            traj_yesterday_weather = batch['traj_yesterday_weather'].to(device)
            traj_today_weather = batch['traj_today_weather'].to(device)
            traj_tomorrow_weather = batch['traj_tomorrow_weather'].to(device)
            traj_month_weather = batch['traj_month_weather'].to(device)
            traj_season = batch['traj_season'].to(device)
            
            target_poi = batch['target_poi'].to(device)
            target_time = batch['target_time'].to(device)
            target_now_weather = batch['target_now_weather'].to(device)
            target_lat = batch['target_lat'].to(device)
            target_lon = batch['target_lon'].to(device)
            target_yesterday_weather = batch['target_yesterday_weather'].to(device)
            target_today_weather = batch['target_today_weather'].to(device)
            target_tomorrow_weather = batch['target_tomorrow_weather'].to(device)
            target_month_weather = batch['target_month_weather'].to(device)
            target_season = batch['target_season'].to(device)
            
            head_poi = batch['head_poi'].to(device)
            mask = batch['mask'].to(device)
            
            # GNN Forward
            # GNN Forward
            refined_u, refined_p, refined_t, refined_w, refined_s, refined_day, refined_month = model.forward_gnn(edges)
            
            # Candidates
            cands = torch.arange(1, num_pois).unsqueeze(0).repeat(user_id.size(0), 1).to(device)
            cand_coords = coord_tensor[cands]
            cand_lats = cand_coords[:, :, 0]
            cand_lons = cand_coords[:, :, 1]
            
            # Prediction
            traj_emb_for_agg = refined_p[traj_poi]
            enhanced_history = model.enhance_poi_rep(traj_emb_for_agg)
            
            scores = model.predict(user_id, traj_poi, traj_time, traj_lat, traj_lon, 
                                   cands, cand_lats, cand_lons, target_time, mask=mask,
                                   p_emb_all=refined_p, enhanced_history=enhanced_history)
            
            target_idx = target_poi - 1
            rec_loss = criterion_rec(scores, target_idx)
            
            # Translation Loss (Validation uses random neg tail for loss calc same as train? 
            # Usually validation loss is just checking. kept same for consistency)
            # Translation Loss
            neg_tail = torch.randint(1, num_pois, target_poi.shape).to(device)
            target_w_day_seq = torch.stack([target_yesterday_weather, target_today_weather, target_tomorrow_weather], dim=1)
            
            trans_loss = model.calc_translation_loss(head_poi, target_time.long(), target_now_weather, target_poi, neg_tail,
                                                     r_season_idx=target_season,
                                                     r_day_seq_idx=target_w_day_seq,
                                                     r_month_idx=target_month_weather,
                                                     u_emb_all=refined_u, p_emb_all=refined_p, 
                                                     t_emb_all=refined_t, w_emb_all=refined_w,
                                                     s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month)
            
            loss = rec_loss + lambda_weight * trans_loss # lambda defined by param
            
            # Metrics
            acc1 = accuracy_at_k(scores, target_idx, k=1)
            acc5 = accuracy_at_k(scores, target_idx, k=5)
            acc10 = accuracy_at_k(scores, target_idx, k=10)
            mrr_val = mrr(scores, target_idx)
            ndcg10_val = ndcg_at_k(scores, target_idx, k=10)
            
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_trans_loss += trans_loss.item()
            total_acc1 += acc1
            total_acc5 += acc5
            total_acc10 += acc10
            total_mrr += mrr_val
            total_ndcg10 += ndcg10_val
            
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
        
    return (total_loss / num_batches, 
            total_rec_loss / num_batches, 
            total_trans_loss / num_batches,
            total_acc1 / num_batches,
            total_acc5 / num_batches,
            total_acc10 / num_batches,
            total_mrr / num_batches,
            total_ndcg10 / num_batches)

def main():
    TIME_SLOT = 24
    # Hyperparameters Config
    Config = {
        'EMBED_DIM': 32,
        'SEQ_LEN': 10,
        'BATCH_SIZE': 16,
        'EPOCHS': 100,
        'LR': 0.001,
        'LAMBDA_WEIGHT': 0.1, # Weight for Translation Loss
        'GAMMA': 1.0,         # Margin for Translation Loss
        'DROPOUT': 0.1,
    }
    
    # デバイスの自動検出
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    

    # 1. Load Real Data
    print("Loading Real Data...")
    num_users, num_pois, raw_data, poi_coords = load_real_data()
    # Note: NUM_POIS includes padding index 0. Actual POIs are 1..NUM_POIS-1
    print(f"Loaded Data: Users={num_users}, POIs={num_pois}")

    # 2. Setup Dataset & DataLoader
    # Create 3 separate datasets based on the usage mode
    train_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='train')
    val_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='validation')
    test_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='test')

    train_loader = DataLoader(train_dataset, batch_size=Config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config['BATCH_SIZE'], shuffle=False)
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Hyperparams: Embed={Config['EMBED_DIM']}, Gamma={Config['GAMMA']}, Lambda={Config['LAMBDA_WEIGHT']}")
    
    # 3. Model
    print("Initializing Model...")
    # Update num_weathers based on data
    num_now_weathers = len(REALTIME_WEATHER_LIST)
    num_day_weathers = len(DAY_WEATHER_LIST)
    num_month_weathers = 9 # 気温傾向(3段階)✖️降水量傾向(3段階)の9段階
    print(f"Num Now Weathers: {num_now_weathers}, Num Day Weathers: {num_day_weathers}, Num Month Weathers: {num_month_weathers}")
    
    # Assuming num_categories=10 for mock data, or 10 if we add real category parsing later.
    model = TWTransNet(num_users, num_pois, Config['EMBED_DIM'], TIME_SLOT, num_now_weathers, num_day_weathers, num_month_weathers, Config['DROPOUT'], Config['GAMMA'])
    model = model.to(device)  # モデルをGPUに転送
    
    optimizer = optim.Adam(model.parameters(), lr=Config['LR'])
    
    print("Starting Training...")
    
    # Build Interaction Graph (Edges) - デバイスを指定
    edges = build_interaction_graph(raw_data, device=device)

    best_val_mrr = -1
    best_val_ndcg10 = -1
    best_model_state = None
    
    for epoch in range(Config['EPOCHS']):

        # Train
        loss, rec_l, trans_l, acc1, acc5, acc10, mrr_val, ndcg10 = train_epoch(model, train_loader, optimizer, num_pois, 
                                                                       edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device)
        print(f"Epoch {epoch+1}/{Config['EPOCHS']} [Train] | Loss: {loss:.4f} | Rec: {rec_l:.4f} | Trans: {trans_l:.4f} "
              f"| Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | Acc@10: {acc10:.4f} | MRR: {mrr_val:.4f} | nDCG@10: {ndcg10:.4f}")
        
        # Validation
        v_loss, v_rec_l, v_trans_l, v_acc1, v_acc5, v_acc10, v_mrr_val, v_ndcg10 = evaluate(model, val_loader, num_pois, 
                                                                                  edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device)
        print(f"Epoch {epoch+1}/{Config['EPOCHS']} [Valid] | Loss: {v_loss:.4f} | Rec: {v_rec_l:.4f} | Trans: {v_trans_l:.4f} "
              f"| Acc@1: {v_acc1:.4f} | Acc@5: {v_acc5:.4f} | Acc@10: {v_acc10:.4f} | MRR: {v_mrr_val:.4f} | nDCG@10: {v_ndcg10:.4f}")
        
        # Save Best Model based on Validation nDCG
        if v_ndcg10 > best_val_ndcg10:
            best_val_ndcg10 = v_ndcg10
            best_model_state = model.state_dict()
            print(f"  >>> Best Valid nDCG@10 updated: {best_val_ndcg10:.4f}")
        else:
            print(f"  >>> Best Valid nDCG@10 not updated: {best_val_ndcg10:.4f}")

    print("Training Finished.")

    
    # Final Test with Best Model
    print("Loading Best Model for Testing...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    t_loss, t_rec_l, t_trans_l, t_acc1, t_acc5, t_acc10, t_mrr_val, t_ndcg10 = evaluate(model, test_loader, num_pois, 
                                                                              edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device)
    print(f"Final Test Result | Loss: {t_loss:.4f} | Rec: {t_rec_l:.4f} | Trans: {t_trans_l:.4f} "
          f"| Acc@1: {t_acc1:.4f} | Acc@5: {t_acc5:.4f} | Acc@10: {t_acc10:.4f} | MRR: {t_mrr_val:.4f} | nDCG@10: {t_ndcg10:.4f}")
    

if __name__ == "__main__":
    main()
