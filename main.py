# dependencies
import tools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# check available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = torch.device("cpu")
print(f"Current device: {device}")

FEATURE_COLUMNS = ['magnitude', 'cdi', 'mmi', 'tsunami', 'dmin', 'gap', 'latitude', 'longitude']
TARGET_COLUMN = 'sig' # or 'alert' 

def load_data(file_path:str) -> tuple[DataLoader, DataLoader]:

    data_frame = pd.read_csv(file_path)
    data_split = int(len(data_frame) * 0.8)

    training_data, test_data = data_frame[:data_split], data_frame[data_split:]
 

    # TODO: Setup Dataset for model
    training_data_set = tools.earthQuakeData(
        data_frame=training_data,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN
    )

    # TODO: Split data into training and test
    test_data_set = tools.earthQuakeData(
        data_frame=test_data,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN   
    )

    data_train_loader = DataLoader(
        training_data_set,
        batch_size=16,
        # shuffle=True,
    )

    data_test_loader = DataLoader(
        test_data_set,
        batch_size=16,
        # shuffle=True,
    )

    # TODO: Return train and test dataloaders
    return data_train_loader, data_test_loader

#   - Training Data
#   - Validation Data
#   - Test Data

# TODO: Setup Training loop

# Earthquake Prediction Model
class EarthquakeModel(nn.Module):
    def __init__(self) -> None:
        super(EarthquakeModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=len(FEATURE_COLUMNS),out_features=128),
            nn.Linear(in_features=128,out_features=1)
        )
    
    def forward(self, x):
        return self.layers(x)
    

def main():
    model = EarthquakeModel()
    train_data_loader, test_data_loader = load_data(file_path='./datasets/earthquake_1995-2023.csv') # or './datasets/earthquake_data.csv')

    # Loss Function
    criterion = nn.L1Loss()
    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    EPOCHS = 250 # Best Training/Test Loss 76% / 67% 

    # Put data on target device
    # Actually have no idea how to do this
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_data_loader:
            # 1. forward pass
            y_pred = model(X)

            # 2. Calculate the loss
            loss = criterion(y_pred, y)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Perform backprogation
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            ### Testing
            model.eval()
            with torch.inference_mode():
                for X_test, y_test in test_data_loader:
                    test_pred = model(X_test)

                    test_loss = criterion(test_pred, y_test)
        # Print out status
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss:{test_loss}")

if __name__ == "__main__":
    main()