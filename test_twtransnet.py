import torch
from TWTransNet.twtransnet import TWTransNet
from TWTransNet.data_loader import CheckinDataset
from train import load_real_data
from torch.utils.data import DataLoader

def test_twtransnet():
    # ... existing test code ...
    pass 

def test_real_data_integration():
    print("\nTesting Real Data Loading and Integration...")
    try:
        num_data, num_pois, raw_data, poi_coords = load_real_data()
        print(f"Successfully loaded data: {num_data} users, {num_pois} POIs")
        
        # Verify Data Structure
        first_user = list(raw_data.keys())[0]
        traj = raw_data[first_user]
        print(f"Sample trajectory length for user {first_user}: {len(traj)}")
        assert len(traj) > 0
        assert len(traj[0]) == 5 # (poi, time, weather, lat, lon)
        
        # Verify Dataset Creation
        dataset = CheckinDataset(raw_data, seq_len=10, num_pois=num_pois)
        print(f"Dataset created with {len(dataset)} samples")
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print("DataLoader produced a batch successfully.")
        print("Batch Keys:", batch.keys())
        
    except FileNotFoundError:
        print("Skipping Real Data Test: '2012-4~5.csv' not found.")
    except Exception as e:
        print(f"Real Data Test Failed: {e}")
        raise e

if __name__ == "__main__":
    test_twtransnet()
    test_real_data_integration()
