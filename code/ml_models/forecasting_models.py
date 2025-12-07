import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _choose_num_heads(embed_dim: int, preferred: int = 8) -> int:
    heads = min(preferred, embed_dim)
    while heads > 1:
        if embed_dim % heads == 0:
            return heads
        heads -= 1
    return 1


class LiquidityLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout),
            bidirectional=self.bidirectional,
        )
        fc_input_size = self.hidden_size * (2 if self.bidirectional else 1)
        num_heads = _choose_num_heads(fc_input_size, preferred=8)
        self.attention = nn.MultiheadAttention(
            embed_dim=fc_input_size, num_heads=num_heads
        )
        self.fc1 = nn.Linear(fc_input_size, max(fc_input_size // 2, 1))
        self.fc2 = nn.Linear(max(fc_input_size // 2, 1), max(fc_input_size // 4, 1))
        self.fc3 = nn.Linear(max(fc_input_size // 4, 1), 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm1 = nn.LayerNorm(self.fc1.out_features)
        self.layer_norm2 = nn.LayerNorm(self.fc2.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(
            lstm_out_permuted, lstm_out_permuted, lstm_out_permuted
        )
        last_output = attn_out[-1]
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
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.conv1 = nn.Conv1d(self.input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout),
        )
        num_heads = _choose_num_heads(self.hidden_size, preferred=4)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=num_heads
        )
        self.fc1 = nn.Linear(self.hidden_size, max(self.hidden_size // 2, 1))
        self.fc2 = nn.Linear(self.fc1.out_features, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_permuted = x.permute(0, 2, 1)
        conv1_out = self.relu(self.conv1(x_permuted))
        conv2_out = self.relu(self.conv2(conv1_out))
        lstm_input = conv2_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(
            lstm_out_permuted, lstm_out_permuted, lstm_out_permuted
        )
        last_output = attn_out[-1]
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)
        return output


class ModelTrainer:
    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-5
    ) -> None:
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ) -> Tuple[List[float], List[float]]:
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError(
                "train_loader and val_loader must contain at least one batch each."
            )

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)
            val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                logger.info("Model saved as best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig("loss_curve.png")
            plt.close()
        except Exception:
            logger.exception("Could not save loss curve figure.")

        return train_losses, val_losses

    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[float, List[float], List[float]]:
        if len(test_loader) == 0:
            raise ValueError("test_loader must contain at least one batch.")

        self.model.eval()
        test_loss = 0.0
        predictions: List[float] = []
        actuals: List[float] = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy().flatten().tolist())
                actuals.extend(targets.cpu().numpy().flatten().tolist())

        test_loss /= max(len(test_loader), 1)
        logger.info(f"Test Loss: {test_loss:.6f}")

        try:
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(actuals)), actuals, label="Actual", alpha=0.7, s=10)
            plt.scatter(
                range(len(predictions)), predictions, label="Predicted", alpha=0.7, s=10
            )
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.title("Predictions vs Actuals")
            plt.legend()
            plt.savefig("predictions_vs_actuals.png")
            plt.close()
        except Exception:
            logger.exception("Could not save predictions vs actuals figure.")

        return test_loss, predictions, actuals


def prepare_data(
    data_path: Optional[str] = None,
    sequence_length: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    try:
        if data_path:
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully from {data_path}")
        else:
            raise FileNotFoundError("No data_path provided, using synthetic data.")
    except Exception as e:
        logger.warning(f"Error loading data or no data_path: {e}")
        logger.info("Creating synthetic data for demonstration")
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        timestamps = pd.date_range(start="2023-01-01", periods=n_samples, freq="H")
        features = np.random.randn(n_samples, n_features)
        target = np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1
        data = np.column_stack([features, target])
        columns = [f"feature_{i}" for i in range(n_features)] + ["target"]
        df = pd.DataFrame(data, index=timestamps, columns=columns)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp"}, inplace=True)

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "target" in df.columns:
        X = df.drop(
            ["target", "timestamp"] if "timestamp" in df.columns else ["target"], axis=1
        ).values
        y = df["target"].values.reshape(-1, 1)
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_sequences: List[np.ndarray] = []
    y_sequences: List[np.ndarray] = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i : i + sequence_length])
        y_sequences.append(y_scaled[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    n_sequences = len(X_sequences)
    if n_sequences == 0:
        raise ValueError(
            "Not enough data to create any sequences. Reduce sequence_length or provide more data."
        )

    train_size = int(n_sequences * train_ratio)
    val_size = int(n_sequences * val_ratio)

    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_val = X_sequences[train_size : train_size + val_size]
    y_val = y_sequences[train_size : train_size + val_size]
    X_test = X_sequences[train_size + val_size :]
    y_test = y_sequences[train_size + val_size :]

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info(
        f"Data prepared: {len(train_loader)} training batches, {len(val_loader)} validation batches, {len(test_loader)} test batches"
    )
    return train_loader, val_loader, test_loader, scaler_X, scaler_y


def train_liquidity_model(
    data_path: Optional[str] = None,
    epochs: int = 100,
    save_path: str = "liquidity_predictor.pt",
) -> Tuple[LiquidityLSTM, float]:
    logger.info("Training liquidity prediction model")
    train_loader, val_loader, test_loader, scaler_X, scaler_y = prepare_data(
        data_path, sequence_length=20, train_ratio=0.7, val_ratio=0.15
    )
    for inputs, _ in train_loader:
        input_size = int(inputs.shape[2])
        break

    model = LiquidityLSTM(input_size=input_size, hidden_size=128, num_layers=4)
    trainer = ModelTrainer(model, learning_rate=0.001, weight_decay=1e-05)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, epochs=epochs, patience=15
    )
    test_loss, predictions, actuals = trainer.evaluate(test_loader)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "input_size": input_size,
            "hidden_size": 128,
            "num_layers": 4,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        save_path,
    )
    logger.info(f"Model saved to {save_path}")
    return model, test_loss


def train_supply_chain_model(
    data_path: Optional[str] = None,
    epochs: int = 100,
    save_path: str = "supply_chain_forecaster.pt",
) -> Tuple[SupplyChainForecaster, float]:
    logger.info("Training supply chain forecasting model")
    train_loader, val_loader, test_loader, scaler_X, scaler_y = prepare_data(
        data_path, sequence_length=30, train_ratio=0.7, val_ratio=0.15
    )
    for inputs, _ in train_loader:
        input_size = int(inputs.shape[2])
        break

    model = SupplyChainForecaster(input_size=input_size, hidden_size=128, num_layers=3)
    trainer = ModelTrainer(model, learning_rate=0.001, weight_decay=1e-05)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, epochs=epochs, patience=15
    )
    test_loss, predictions, actuals = trainer.evaluate(test_loader)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "input_size": input_size,
            "hidden_size": 128,
            "num_layers": 3,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        save_path,
    )
    logger.info(f"Model saved to {save_path}")
    return model, test_loss


if __name__ == "__main__":
    liquidity_model, liquidity_loss = train_liquidity_model(
        data_path="historical_trades.csv", epochs=50, save_path="liquidity_predictor.pt"
    )
    supply_chain_model, supply_chain_loss = train_supply_chain_model(
        data_path="supply_chain_data.csv",
        epochs=50,
        save_path="supply_chain_forecaster.pt",
    )
    logger.info("Training completed successfully")
    logger.info(f"Liquidity model test loss: {liquidity_loss:.6f}")
    logger.info(f"Supply chain model test loss: {supply_chain_loss:.6f}")
