import torch

def build_interaction_graph(raw_data, device='cpu'):
    """
    Convert raw_data {user: [traj]} into edge tensors for GNN.
    Edge = (User, POI, Time, Weather)
    """
    u_list, p_list, t_list, w_now_list, w_day_list, w_month_list, s_list = [], [], [], [], [], [], []
    
    for u_id, traj in raw_data.items():
        for i, item in enumerate(traj):
            # item: (poi, float_time, weather, lat, lon, season, category, w_minus, w_plus)
            poi = item[0]
            time_val = item[1]
            # Map float time to index (0-23)
            time_idx = int(time_val) % 24 
            now_weather = item[2]

            yesterday_weather = item[5]
            today_weather = item[6] # Today (Day granularity)
            tomorrow_weather = item[7]
            month_weather = item[8]
            season = item[9]

            three_day_weather = [yesterday_weather, today_weather, tomorrow_weather]
            
            u_list.append(u_id)
            p_list.append(poi)
            t_list.append(time_idx)
            w_now_list.append(now_weather)
            w_day_list.append(three_day_weather)
            w_month_list.append(month_weather)
            s_list.append(season)
            
    edges = {
        'user_idxs': torch.tensor(u_list, dtype=torch.long).to(device),
        'poi_idxs': torch.tensor(p_list, dtype=torch.long).to(device),
        'time_idxs': torch.tensor(t_list, dtype=torch.long).to(device),
        'weather_now_idxs' : torch.tensor(w_now_list, dtype=torch.long).to(device),
        'weather_day_idxs' : torch.tensor(w_day_list, dtype=torch.long).to(device),
        'weather_month_idxs' : torch.tensor(w_month_list, dtype=torch.long).to(device),
        'season_idxs': torch.tensor(s_list, dtype=torch.long).to(device),
    }
    return edges
