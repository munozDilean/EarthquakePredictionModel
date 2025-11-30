# dependencies
import tools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# check available device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Current device: {device}")

# TODO: Setup Dataset for model
data_set = tools.earthQuakeData(
    file_path='./datasets/earthquake_1995-2023.csv', # or './datasets/earthquake_data.csv'
    feature_columns=['magnitude', 'cdi', 'mmi', 'tsunami', 'dmin', 'gap', 'latitude', 'longitude'],
    target_column='sig' # or 'alert'
)
    
data_loader = DataLoader(
    data_set,
    batch_size=16,
    # shuffle=True,
)

#   - Training Data
#   - Validation Data
#   - Test Data

# TODO: Setup Training loop

# Earthquake Prediction Model
class MLPNet(nn.Module):
    def __init__(self) -> None:
        super(MLPNet, self).__init__()
    
    def forward(self, x):
        pass