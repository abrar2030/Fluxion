import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- 1. Model Definition (Kept from original, but with minor cleanup) ---


class LiquidityLSTM(nn.Module):
    """
    LSTM-based model with attention mechanism for predicting synthetic asset liquidity.
    Input features (10): Price, Volume, Volatility, Interest Rate, Time-to-Maturity,
                         Collateral Ratio, Gas Price, Network Congestion, TVL, Pool Size.
    Output (1): Predicted next-period liquidity depth.
    """

    def __init__(self, input_size=10, hidden_size=64, num_layers=3, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        # Bidirectional LSTM output size is 2 * hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * hidden_size, num_heads=num_heads, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, 2 * hidden_size)

        # Attention mechanism: Use the last hidden state as the query for simplicity,
        # or use the full sequence for a more complex attention
        # Here, we use the full sequence for self-attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the output of the last time step after attention
        # attn_output[:, -1] shape: (batch_size, 2 * hidden_size)
        return self.fc(attn_output[:, -1])


# --- 2. Data Simulation and Preprocessing ---


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length):
    """Convert time series data into sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(
            data[i + seq_length, -1]
        )  # Predict the last feature (Liquidity) of the next step
    return np.array(X), np.array(y)


def simulate_data(n_samples=5000, n_features=10, seq_length=20):
    """Simulate time series data for liquidity prediction."""
    logger.info(f"Simulating {n_samples} time steps of data...")

    # Generate correlated random walk data
    np.random.seed(42)
    data = np.zeros((n_samples, n_features))

    # Base price/volume/volatility features
    for i in range(n_features - 1):
        data[:, i] = np.cumsum(np.random.randn(n_samples) * 0.1) + np.random.rand() * 10

    # Last feature is 'Liquidity' (dependent on others)
    # Liquidity = f(Price, Volume, Volatility, TVL) + noise
    data[:, -1] = (
        data[:, 0] * 0.5
        + data[:, 1] * 0.3
        - data[:, 2] * 0.2
        + data[:, 8] * 0.1
        + np.sin(np.arange(n_samples) / 100) * 2
        + np.random.randn(n_samples) * 0.5
    )

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    logger.info(f"Training set size: {len(X_train)} sequences")
    logger.info(f"Testing set size: {len(X_test)} sequences")

    return (
        TimeSeriesDataset(X_train, y_train),
        TimeSeriesDataset(X_test, y_test),
        scaler,
    )


# --- 3. Training Function ---


def train_model(epochs=10, batch_size=64, seq_length=20):
    """
    Full training pipeline for the Liquidity Prediction Model.
    """
    logger.info("Starting Liquidity Prediction Model training...")

    # 1. Data Preparation
    train_dataset, test_dataset, scaler = simulate_data(seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, and Optimizer
    model = LiquidityLSTM(input_size=train_dataset.X.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 4. Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}"
        )

    # 5. Save Model
    model_path = "liquidity_predictor.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model trained and saved to {model_path}")

    # Return the scaler for inverse transformation later (e.g., in a prediction service)
    return model, scaler


if __name__ == "__main__":
    # Example of running the training
    trained_model, data_scaler = train_model(epochs=5)

    # Example of a dummy prediction
    dummy_input = np.random.rand(1, 20, 10)  # 1 sample, 20 sequence length, 10 features
    scaled_dummy_input = data_scaler.transform(dummy_input[0, -1, :].reshape(1, -1))

    # Create a sequence from the scaled data (this is a simplification,
    # in a real scenario you'd use the last 'seq_length' of real-time data)
    dummy_sequence = torch.tensor(dummy_input, dtype=torch.float32)

    trained_model.eval()
    with torch.no_grad():
        prediction = trained_model(dummy_sequence)

    # Inverse transform the prediction (only the last feature, which is liquidity)
    # We need a dummy array to inverse transform the single value
    dummy_inverse = np.zeros((1, 10))
    dummy_inverse[:, -1] = prediction.cpu().numpy().flatten()

    # The scaler was fitted on the full 10 features, so we need to inverse transform all
    # and take the last one.
    original_scale_prediction = data_scaler.inverse_transform(dummy_inverse)[:, -1]

    logger.info(f"Dummy Prediction (Scaled): {prediction.item():.4f}")
    logger.info(
        f"Dummy Prediction (Original Scale - Liquidity): {original_scale_prediction[0]:.2f}"
    )
