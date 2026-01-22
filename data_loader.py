import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CheckinDataset(Dataset):
    """
    Dataset for TWTransNet.
    Generates items for both:
    1. Translation Training (Triplets: Head, Time, Weather, Tail, NegTail)
       - Actually, for efficiency, we might sample negative tails on the fly or providing them here.
       - But the paper trains everything jointly or iteratively?
       - "We run our model 100 epochs... L = L_T + L_Rec ?"
       - No, Eq 14 L does not mention L_T.
       - But Section 4.1 defines L_T (Eq 2) and Section 4.3 defines L (Eq 14).
       - Usually joint training: Loss = L_rec + lambda * L_trans.
       
    2. Recommendation Training (Trajectory, Time, Weather, Target)
    """
    def __init__(self, user_trajectories, seq_len=20, num_pois=None):
        """
        user_trajectories: dict {user_id: list of (poi_id, time_idx, weather_idx, lat, lon)}
        seq_len: max sequence length
        """
        self.user_trajectories = user_trajectories
        self.seq_len = seq_len
        self.num_pois = num_pois
        
        self.data_samples = []
        self._process_trajectories()

    def _process_trajectories(self):
        """
        Create sliding window sequences.
        Format: (user_id, input_seq_poi, input_seq_time, input_seq_now_weather, input_seq_lat, input_seq_lon, input_seq_yesterday_weather, input_seq_today_weather, input_seq_tomorrow_weather, input_seq_month_weather, input_seq_season,
                 target_poi, target_time, target_now_weather, target_lat, target_lon, target_yesterday_weather, target_today_weather, target_tomorrow_weather, target_month_weather, target_season)
        """
        for user_id, traj in self.user_trajectories.items():
            # traj is list of tuples
            if len(traj) < 2:
                continue
            
            # Simple approach: input=[0...t-1], target=t
            # Sliding window if trajectory is long
            
            # We treat the entire history up to t-1 as input (truncated to seq_len)
            for i in range(1, len(traj)):
                target = traj[i]
                
                start_idx = max(0, i - self.seq_len)
                input_seq = traj[start_idx:i]
                
                if len(input_seq) > self.seq_len:
                    pass # Should be handled by seq_len constraint in loop

                self.data_samples.append({
                    'user_id': user_id,
                    'input_seq': input_seq,
                    'target': target
                })

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        user_id = sample['user_id']
        input_seq = sample['input_seq']
        target = sample['target'] # (poi, time, weather, lat, lon)
        
        # Unpack input sequence
        # (poi, time, weather, lat, lon) per step
        # Unpack input sequence
        # (poi, time, weather, lat, lon, season, category, w_minus, w_plus) per step
        pois = [x[0] for x in input_seq]
        times = [x[1] for x in input_seq]
        now_weathers = [x[2] for x in input_seq]
        lats = [x[3] for x in input_seq]
        lons = [x[4] for x in input_seq]
        yesterday_weathers = [x[5] for x in input_seq]
        today_weathers = [x[6] for x in input_seq]
        tomorrow_weathers = [x[7] for x in input_seq]
        month_weathers = [x[8] for x in input_seq]
        seasons = [x[9] for x in input_seq]
        
        # Force limit to seq_len
        target_len = self.seq_len # Should be 10
        current_len = len(pois)
        
        if current_len < target_len:
            pad_len = target_len - current_len
            pois = [0] * pad_len + pois
            times = [0] * pad_len + times
            now_weathers = [0] * pad_len + now_weathers
            lats = [0.0] * pad_len + lats
            lons = [0.0] * pad_len + lons
            yesterday_weathers = [0] * pad_len + yesterday_weathers
            today_weathers = [0] * pad_len + today_weathers
            tomorrow_weathers = [0] * pad_len + tomorrow_weathers
            month_weathers = [0] * pad_len + month_weathers
            seasons = [0] * pad_len + seasons
            mask = [0] * pad_len + [1] * current_len
        else:
            # Truncate from end? Or Keep recent?
            # input_seq is [start:i]. Most recent is at end.
            # Convert to keep last target_len
            offset = current_len - target_len
            pois = pois[offset:]
            times = times[offset:]
            now_weathers = now_weathers[offset:]
            lats = lats[offset:]
            lons = lons[offset:]
            yesterday_weathers = yesterday_weathers[offset:]
            today_weathers = today_weathers[offset:]
            tomorrow_weathers = tomorrow_weathers[offset:]
            month_weathers = month_weathers[offset:]
            seasons = seasons[offset:]
            mask = [1] * target_len

        target_poi = target[0]
        target_time = target[1]
        target_now_weather = target[2]
        target_lat = target[3]
        target_lon = target[4]
        target_yesterday_weather = target[5]
        target_today_weather = target[6]
        target_tomorrow_weather = target[7]
        target_month_weather = target[8]
        target_season = target[9]
        
        # For Translation Loss, we need the Last Step -> Target transition
        # Head = input_seq[-1], Relation = Target Time/Weather, Tail = Target POI
        # If input_seq is empty (padded fully?), valid len check needed.
        # But we iterated range(1, len), so at least 1 item in input.
        last_item = input_seq[-1]
        head_poi = last_item[0]
        
        # Helper for negative sampling (can be done in training loop or here)
        # We return everything as tensors
        
        # Debug shapes if needed
        # print(f"Sample {idx}: pois {len(pois)}, times {len(times)}")
        if len(pois) != self.seq_len:
             # Should not happen with forced padding
             pass
        
        assert len(pois) == self.seq_len, f"Pois len {len(pois)} != {self.seq_len}"
        assert len(seasons) == self.seq_len, f"Seasons len {len(seasons)} != {self.seq_len}"

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'traj_poi': torch.tensor(pois, dtype=torch.long),
            'traj_time': torch.tensor(times, dtype=torch.float), # Assuming continuous or index
            'traj_now_weather': torch.tensor(now_weathers, dtype=torch.long),
            'traj_lat': torch.tensor(lats, dtype=torch.float),
            'traj_lon': torch.tensor(lons, dtype=torch.float),
            'traj_yesterday_weather': torch.tensor(yesterday_weathers, dtype=torch.long),
            'traj_today_weather': torch.tensor(today_weathers, dtype=torch.long),
            'traj_tomorrow_weather': torch.tensor(tomorrow_weathers, dtype=torch.long),
            'traj_month_weather': torch.tensor(month_weathers, dtype=torch.long),
            'traj_season': torch.tensor(seasons, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            
            'head_poi': torch.tensor(head_poi, dtype=torch.long),
            'target_poi': torch.tensor(target_poi, dtype=torch.long),
            'target_time': torch.tensor(target_time, dtype=torch.float),
            'target_now_weather': torch.tensor(target_now_weather, dtype=torch.long),
            'target_lat': torch.tensor(target_lat, dtype=torch.float),
            'target_lon': torch.tensor(target_lon, dtype=torch.float),
            'target_yesterday_weather': torch.tensor(target_yesterday_weather, dtype=torch.long),
            'target_today_weather': torch.tensor(target_today_weather, dtype=torch.long),
            'target_tomorrow_weather': torch.tensor(target_tomorrow_weather, dtype=torch.long),
            'target_month_weather': torch.tensor(target_month_weather, dtype=torch.long),
            'target_season': torch.tensor(target_season, dtype=torch.long),
        }
