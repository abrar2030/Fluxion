import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLiquidityLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=4, dropout=0.2, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2 if bidirectional else hidden_size,
            num_heads=8
        )
        
        # Fully connected layers with residual connections
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_input_size, fc_input_size // 2)
        self.fc2 = nn.Linear(fc_input_size // 2, fc_input_size // 4)
        self.fc3 = nn.Linear(fc_input_size // 4, 1)
        
        # Activation functions
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(fc_input_size // 2)
        self.layer_norm2 = nn.LayerNorm(fc_input_size // 4)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2) if bidirectional
        
        # Apply attention mechanism
        # Reshape for attention: (sequence_length, batch_size, hidden_size * 2)
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out_permuted, lstm_out_permuted, lstm_out_permuted)
        
        # Take the last sequence element
        last_output = attn_out[-1]
        
        # Fully connected layers with residual connections
        fc1_out = self.fc1(last_output)
        fc1_out = self.layer_norm1(fc1_out)
        fc1_out = self.gelu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.layer_norm2(fc2_out)
        fc2_out = self.gelu(fc2_out)
        fc2_out = self.dropout(fc2_out)
        
        output = self.fc3(fc2_out)
        
        return output

class SupplyChainForecaster(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 5)  # 5 outputs for different status predictions
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        
        # Permute for CNN: (batch_size, input_size, sequence_length)
        x_permuted = x.permute(0, 2, 1)
        
        # Apply CNN for feature extraction
        conv1_out = self.relu(self.conv1(x_permuted))
        conv2_out = self.relu(self.conv2(conv1_out))
        
        # Permute back for LSTM: (batch_size, sequence_length, 64)
        lstm_input = conv2_out.permute(0, 2, 1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(lstm_input)
        
        # Apply attention mechanism
        # Reshape for attention: (sequence_length, batch_size, hidden_size)
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out_permuted, lstm_out_permuted, lstm_out_permuted)
        
        # Take the last sequence element
        last_output = attn_out[-1]
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)
        
        return output

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pt")
                logger.info("Model saved as best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss_curve.png')
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        test_loss /= len(test_loader)
        logger.info(f"Test Loss: {test_loss:.6f}")
        
        # Plot predictions vs actuals
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(actuals)), actuals, label='Actual', alpha=0.7, s=10)
        plt.scatter(range(len(predictions)), predictions, label='Predicted', alpha=0.7, s=10)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Predictions vs Actuals')
        plt.legend()
        plt.savefig('predictions_vs_actuals.png')
        
        return test_loss, predictions, actuals

def prepare_data(data_path, sequence_length=10, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare data for training, validation, and testing
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Create synthetic data for demonstration
        logger.info("Creating synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Create time series data
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        features = np.random.randn(n_samples, n_features)
        target = np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1
        
        # Create DataFrame
        data = np.column_stack([features, target])
        columns = [f'feature_{i}' for i in range(n_features)] + ['target']
        df = pd.DataFrame(data, index=timestamps, columns=columns)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Preprocess data
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract features and target
    if 'target' in df.columns:
        X = df.drop(['target', 'timestamp'] if 'timestamp' in df.columns else ['target'], axis=1).values
        y = df['target'].values.reshape(-1, 1)
    else:
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
    
    # Normalize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y_scaled[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split data
    n_samples = len(X_sequences)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    
    X_val = X_sequences[train_size:train_size+val_size]
    y_val = y_sequences[train_size:train_size+val_size]
    
    X_test = X_sequences[train_size+val_size:]
    y_test = y_sequences[train_size+val_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"Data prepared: {len(train_loader)} training batches, {len(val_loader)} validation batches, {len(test_loader)} test batches")
    
    return train_loader, val_loader, test_loader, scaler_X, scaler_y

def train_liquidity_model(data_path=None, epochs=100, save_path="liquidity_predictor.pt"):
    """
    Train the enhanced liquidity prediction model
    """
    logger.info("Training enhanced liquidity prediction model")
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler_X, scaler_y = prepare_data(
        data_path, sequence_length=20, train_ratio=0.7, val_ratio=0.15
    )
    
    # Get input size from data
    for inputs, _ in train_loader:
        input_size = inputs.shape[2]
        break
    
    # Initialize model
    model = EnhancedLiquidityLSTM(input_size=input_size, hidden_size=128, num_layers=4)
    
    # Train model
    trainer = ModelTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=epochs, patience=15)
    
    # Evaluate model
    test_loss, predictions, actuals = trainer.evaluate(test_loader)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 4,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
    
    return model, test_loss

def train_supply_chain_model(data_path=None, epochs=100, save_path="supply_chain_forecaster.pt"):
    """
    Train the supply chain forecasting model
    """
    logger.info("Training supply chain forecasting model")
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler_X, scaler_y = prepare_data(
        data_path, sequence_length=30, train_ratio=0.7, val_ratio=0.15
    )
    
    # Get input size from data
    for inputs, _ in train_loader:
        input_size = inputs.shape[2]
        break
    
    # Initialize model
    model = SupplyChainForecaster(input_size=input_size, hidden_size=128, num_layers=3)
    
    # Train model
    trainer = ModelTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=epochs, patience=15)
    
    # Evaluate model
    test_loss, predictions, actuals = trainer.evaluate(test_loader)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 3,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
    
    return model, test_loss

if __name__ == "__main__":
    # Train liquidity prediction model
    liquidity_model, liquidity_loss = train_liquidity_model(
        data_path="historical_trades.csv",
        epochs=50,
        save_path="liquidity_predictor.pt"
    )
    
    # Train supply chain forecasting model
    supply_chain_model, supply_chain_loss = train_supply_chain_model(
        data_path="supply_chain_data.csv",
        epochs=50,
        save_path="supply_chain_forecaster.pt"
    )
    
    logger.info("Training completed successfully")
    logger.info(f"Liquidity model test loss: {liquidity_loss:.6f}")
    logger.info(f"Supply chain model test loss: {supply_chain_loss:.6f}")
