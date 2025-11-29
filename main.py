import torch.nn as nn

# TODO: Download data to repo
# TODO: Setup Dataset for model
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