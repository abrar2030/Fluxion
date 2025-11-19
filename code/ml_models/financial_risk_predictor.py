"""
Financial Risk Predictor for Fluxion Platform
Advanced machine learning models for predicting financial risks,
market volatility, and compliance violations in DeFi environments.
"""

import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialRiskLSTM(nn.Module):
    """
    Advanced LSTM model for financial risk prediction
    Predicts multiple risk factors including market risk, credit risk, and operational risk
    """

    def __init__(
        self,
        input_size=20,
        hidden_size=256,
        num_layers=3,
        dropout=0.3,
        num_risk_factors=5,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_risk_factors = num_risk_factors

        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Attention mechanism for focusing on important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, dropout=dropout  # Bidirectional
        )

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Risk factor prediction heads
        self.market_risk_head = nn.Linear(hidden_size // 2, 1)
        self.credit_risk_head = nn.Linear(hidden_size // 2, 1)
        self.liquidity_risk_head = nn.Linear(hidden_size // 2, 1)
        self.operational_risk_head = nn.Linear(hidden_size // 2, 1)
        self.compliance_risk_head = nn.Linear(hidden_size // 2, 1)

        # Overall risk score
        self.risk_aggregator = nn.Sequential(
            nn.Linear(num_risk_factors, num_risk_factors * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_risk_factors * 2, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        x.size(0)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)

        # Apply attention mechanism
        # Reshape for attention: (sequence_length, batch_size, hidden_size * 2)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, attn_weights = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )

        # Use the last time step output
        last_output = attn_out[-1]  # (batch_size, hidden_size * 2)

        # Feature extraction
        features = self.feature_extractor(last_output)

        # Risk factor predictions
        market_risk = torch.sigmoid(self.market_risk_head(features))
        credit_risk = torch.sigmoid(self.credit_risk_head(features))
        liquidity_risk = torch.sigmoid(self.liquidity_risk_head(features))
        operational_risk = torch.sigmoid(self.operational_risk_head(features))
        compliance_risk = torch.sigmoid(self.compliance_risk_head(features))

        # Combine individual risk factors
        risk_factors = torch.cat(
            [
                market_risk,
                credit_risk,
                liquidity_risk,
                operational_risk,
                compliance_risk,
            ],
            dim=1,
        )

        # Overall risk score
        overall_risk = self.risk_aggregator(risk_factors)

        return {
            "overall_risk": overall_risk,
            "market_risk": market_risk,
            "credit_risk": credit_risk,
            "liquidity_risk": liquidity_risk,
            "operational_risk": operational_risk,
            "compliance_risk": compliance_risk,
            "attention_weights": attn_weights,
        }


class AnomalyDetectionModel(nn.Module):
    """
    Autoencoder-based anomaly detection for identifying unusual trading patterns
    and potential fraud or market manipulation
    """

    def __init__(self, input_size=50, encoding_dim=20, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size // 4),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, encoding_dim),
            nn.Tanh(),  # Bounded encoding
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size // 4),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ComplianceViolationDetector(nn.Module):
    """
    Specialized model for detecting potential compliance violations
    including AML, KYC, and regulatory breaches
    """

    def __init__(self, input_size=30, hidden_size=128, num_classes=6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes  # Different types of violations

        # Feature extraction network
        self.feature_network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification heads for different violation types
        self.aml_head = nn.Linear(hidden_size // 2, 1)  # Anti-Money Laundering
        self.kyc_head = nn.Linear(hidden_size // 2, 1)  # Know Your Customer
        self.sanctions_head = nn.Linear(hidden_size // 2, 1)  # Sanctions violations
        self.transaction_limit_head = nn.Linear(
            hidden_size // 2, 1
        )  # Transaction limits
        self.reporting_head = nn.Linear(hidden_size // 2, 1)  # Reporting violations
        self.general_head = nn.Linear(hidden_size // 2, 1)  # General compliance

    def forward(self, x):
        features = self.feature_network(x)

        # Individual violation probabilities
        aml_prob = torch.sigmoid(self.aml_head(features))
        kyc_prob = torch.sigmoid(self.kyc_head(features))
        sanctions_prob = torch.sigmoid(self.sanctions_head(features))
        transaction_limit_prob = torch.sigmoid(self.transaction_limit_head(features))
        reporting_prob = torch.sigmoid(self.reporting_head(features))
        general_prob = torch.sigmoid(self.general_head(features))

        return {
            "aml_violation": aml_prob,
            "kyc_violation": kyc_prob,
            "sanctions_violation": sanctions_prob,
            "transaction_limit_violation": transaction_limit_prob,
            "reporting_violation": reporting_prob,
            "general_violation": general_prob,
        }


class FinancialRiskPredictor:
    """
    Comprehensive financial risk prediction system
    """

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.risk_model = None
        self.anomaly_model = None
        self.compliance_model = None

        # Scalers
        self.risk_scaler = StandardScaler()
        self.anomaly_scaler = RobustScaler()
        self.compliance_scaler = StandardScaler()

        # Isolation Forest for additional anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )

        logger.info(f"FinancialRiskPredictor initialized on device: {self.device}")

    def _default_config(self):
        """Default configuration"""
        return {
            "risk_model": {
                "input_size": 20,
                "hidden_size": 256,
                "num_layers": 3,
                "dropout": 0.3,
                "sequence_length": 30,
            },
            "anomaly_model": {"input_size": 50, "encoding_dim": 20, "dropout": 0.2},
            "compliance_model": {
                "input_size": 30,
                "hidden_size": 128,
                "num_classes": 6,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 100,
                "patience": 15,
                "weight_decay": 1e-5,
            },
        }

    def prepare_risk_data(self, data_path=None, synthetic=True):
        """
        Prepare data for risk prediction training
        """
        if synthetic or data_path is None:
            logger.info("Generating synthetic financial risk data")
            return self._generate_synthetic_risk_data()
        else:
            logger.info(f"Loading risk data from {data_path}")
            return self._load_real_data(data_path)

    def _generate_synthetic_risk_data(self):
        """Generate synthetic financial risk data for training"""
        np.random.seed(42)

        # Generate time series data
        n_samples = 10000
        sequence_length = self.config["risk_model"]["sequence_length"]
        n_features = self.config["risk_model"]["input_size"]

        # Market indicators
        market_volatility = np.random.exponential(0.02, n_samples)
        price_changes = np.random.normal(0, market_volatility)
        volume_changes = np.random.lognormal(0, 0.5, n_samples)

        # Economic indicators
        interest_rates = np.random.normal(0.03, 0.01, n_samples)
        inflation_rates = np.random.normal(0.025, 0.005, n_samples)
        gdp_growth = np.random.normal(0.02, 0.01, n_samples)

        # Technical indicators
        rsi = np.random.uniform(20, 80, n_samples)
        macd = np.random.normal(0, 0.1, n_samples)
        bollinger_position = np.random.uniform(-1, 1, n_samples)

        # Risk factors (targets)
        market_risk = np.clip(
            0.3 * market_volatility
            + 0.2 * np.abs(price_changes)
            + 0.1 * (rsi - 50) / 50
            + np.random.normal(0, 0.05, n_samples),
            0,
            1,
        )

        credit_risk = np.clip(
            0.4 * interest_rates
            + 0.3 * inflation_rates
            + 0.2 * np.maximum(0, -gdp_growth)
            + np.random.normal(0, 0.05, n_samples),
            0,
            1,
        )

        liquidity_risk = np.clip(
            0.5 * (1 / (volume_changes + 1e-6))
            + 0.3 * market_volatility
            + np.random.normal(0, 0.05, n_samples),
            0,
            1,
        )

        operational_risk = np.clip(
            0.2
            + 0.1 * np.sin(np.arange(n_samples) * 2 * np.pi / 252)
            + np.random.normal(0, 0.1, n_samples),
            0,
            1,
        )

        compliance_risk = np.clip(
            0.1
            + 0.05 * np.random.exponential(1, n_samples)
            + np.random.normal(0, 0.05, n_samples),
            0,
            1,
        )

        # Combine features
        features = np.column_stack(
            [
                market_volatility,
                price_changes,
                volume_changes,
                interest_rates,
                inflation_rates,
                gdp_growth,
                rsi,
                macd,
                bollinger_position,
                np.random.normal(
                    0, 1, (n_samples, n_features - 9)
                ),  # Additional features
            ]
        )

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length, n_samples):
            X_sequences.append(features[i - sequence_length : i])
            y_sequences.append(
                [
                    market_risk[i],
                    credit_risk[i],
                    liquidity_risk[i],
                    operational_risk[i],
                    compliance_risk[i],
                ]
            )

        X = np.array(X_sequences)
        y = np.array(y_sequences)

        logger.info(f"Generated {len(X)} sequences with shape {X.shape}")
        return X, y

    def prepare_anomaly_data(self, normal_ratio=0.9):
        """Prepare data for anomaly detection training"""
        np.random.seed(42)

        n_samples = 5000
        n_features = self.config["anomaly_model"]["input_size"]

        # Normal trading patterns
        n_normal = int(n_samples * normal_ratio)
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Anomalous patterns
        n_anomalous = n_samples - n_normal
        anomalous_data = []

        # Type 1: Extreme values
        extreme_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features) * 5,  # Higher variance
            size=n_anomalous // 3,
        )
        anomalous_data.append(extreme_data)

        # Type 2: Unusual correlations
        corr_matrix = np.eye(n_features)
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.9  # Strong correlation
        unusual_corr_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 2, cov=corr_matrix, size=n_anomalous // 3
        )
        anomalous_data.append(unusual_corr_data)

        # Type 3: Shifted distribution
        shifted_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,
            cov=np.eye(n_features) * 0.5,
            size=n_anomalous - 2 * (n_anomalous // 3),
        )
        anomalous_data.append(shifted_data)

        # Combine data
        X_normal = normal_data
        X_anomalous = np.vstack(anomalous_data)

        # Labels (0 = normal, 1 = anomalous)
        y_normal = np.zeros(len(X_normal))
        y_anomalous = np.ones(len(X_anomalous))

        X = np.vstack([X_normal, X_anomalous])
        y = np.hstack([y_normal, y_anomalous])

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        logger.info(
            f"Generated anomaly detection data: {len(X)} samples, "
            f"{np.sum(y)} anomalies ({np.mean(y):.2%})"
        )

        return X, y

    def prepare_compliance_data(self):
        """Prepare data for compliance violation detection"""
        np.random.seed(42)

        n_samples = 8000
        n_features = self.config["compliance_model"]["input_size"]

        # Generate features representing transaction and user characteristics
        transaction_amounts = np.random.lognormal(
            8, 2, n_samples
        )  # Transaction amounts
        transaction_frequency = np.random.poisson(
            5, n_samples
        )  # Daily transaction count
        user_age_days = np.random.exponential(365, n_samples)  # Account age
        kyc_score = np.random.beta(8, 2, n_samples)  # KYC completeness score
        country_risk = np.random.choice(
            [0, 0.2, 0.5, 0.8, 1.0], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03]
        )  # Country risk

        # Additional features
        additional_features = np.random.normal(0, 1, (n_samples, n_features - 5))

        # Combine features
        X = np.column_stack(
            [
                np.log1p(transaction_amounts),  # Log transform
                np.log1p(transaction_frequency),
                np.log1p(user_age_days),
                kyc_score,
                country_risk,
                additional_features,
            ]
        )

        # Generate violation labels based on risk factors
        aml_violations = (
            (transaction_amounts > 50000)
            & (transaction_frequency > 10)
            & (country_risk > 0.5)
        ).astype(float)

        kyc_violations = (kyc_score < 0.3).astype(float)

        sanctions_violations = (
            (country_risk > 0.8) & (transaction_amounts > 10000)
        ).astype(float)

        transaction_limit_violations = (transaction_amounts > 100000).astype(float)

        reporting_violations = (
            (transaction_amounts > 25000)
            & (np.random.random(n_samples) < 0.1)  # 10% chance
        ).astype(float)

        general_violations = (
            (aml_violations | kyc_violations | sanctions_violations)
            & (np.random.random(n_samples) < 0.2)  # 20% chance
        ).astype(float)

        y = np.column_stack(
            [
                aml_violations,
                kyc_violations,
                sanctions_violations,
                transaction_limit_violations,
                reporting_violations,
                general_violations,
            ]
        )

        logger.info(f"Generated compliance data: {len(X)} samples")
        logger.info(
            f"Violation rates: AML: {np.mean(aml_violations):.2%}, "
            f"KYC: {np.mean(kyc_violations):.2%}, "
            f"Sanctions: {np.mean(sanctions_violations):.2%}"
        )

        return X, y

    def train_risk_model(self, X, y, validation_split=0.2):
        """Train the financial risk prediction model"""
        logger.info("Training financial risk prediction model")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )

        # Scale features
        X_train_scaled = self.risk_scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_val_scaled = self.risk_scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.risk_model = FinancialRiskLSTM(**self.config["risk_model"]).to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.risk_model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config["training"]["epochs"]):
            # Training
            self.risk_model.train()
            train_loss = 0.0

            for i in range(
                0, len(X_train_tensor), self.config["training"]["batch_size"]
            ):
                batch_X = X_train_tensor[i : i + self.config["training"]["batch_size"]]
                batch_y = y_train_tensor[i : i + self.config["training"]["batch_size"]]

                optimizer.zero_grad()
                outputs = self.risk_model(batch_X)

                # Calculate loss for each risk factor
                loss = 0
                for j, risk_type in enumerate(
                    [
                        "market_risk",
                        "credit_risk",
                        "liquidity_risk",
                        "operational_risk",
                        "compliance_risk",
                    ]
                ):
                    loss += criterion(outputs[risk_type].squeeze(), batch_y[:, j])

                # Add overall risk loss
                overall_target = torch.mean(batch_y, dim=1)
                loss += criterion(outputs["overall_risk"].squeeze(), overall_target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.risk_model.parameters(), max_norm=1.0
                )
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_train_tensor) // self.config["training"]["batch_size"]
            train_losses.append(train_loss)

            # Validation
            self.risk_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i in range(
                    0, len(X_val_tensor), self.config["training"]["batch_size"]
                ):
                    batch_X = X_val_tensor[
                        i : i + self.config["training"]["batch_size"]
                    ]
                    batch_y = y_val_tensor[
                        i : i + self.config["training"]["batch_size"]
                    ]

                    outputs = self.risk_model(batch_X)

                    loss = 0
                    for j, risk_type in enumerate(
                        [
                            "market_risk",
                            "credit_risk",
                            "liquidity_risk",
                            "operational_risk",
                            "compliance_risk",
                        ]
                    ):
                        loss += criterion(outputs[risk_type].squeeze(), batch_y[:, j])

                    overall_target = torch.mean(batch_y, dim=1)
                    loss += criterion(outputs["overall_risk"].squeeze(), overall_target)

                    val_loss += loss.item()

            val_loss /= len(X_val_tensor) // self.config["training"]["batch_size"]
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']}, "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.risk_model.state_dict(), "best_risk_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config["training"]["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.risk_model.load_state_dict(torch.load("best_risk_model.pt"))

        # Plot training curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Risk Model Training Curves")
        plt.legend()
        plt.savefig("risk_model_training_curves.png")

        logger.info("Risk model training completed")
        return train_losses, val_losses

    def train_anomaly_model(self, X, y, validation_split=0.2):
        """Train the anomaly detection model"""
        logger.info("Training anomaly detection model")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.anomaly_scaler.fit_transform(X_train)
        X_val_scaled = self.anomaly_scaler.transform(X_val)

        # Train isolation forest on normal data only
        normal_data = X_train_scaled[y_train == 0]
        self.isolation_forest.fit(normal_data)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)

        # Initialize autoencoder model
        self.anomaly_model = AnomalyDetectionModel(**self.config["anomaly_model"]).to(
            self.device
        )

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.anomaly_model.parameters(), lr=self.config["training"]["learning_rate"]
        )

        # Training loop (train only on normal data)
        normal_indices = np.where(y_train == 0)[0]
        X_normal_tensor = X_train_tensor[normal_indices]

        best_val_loss = float("inf")
        train_losses = []

        for epoch in range(self.config["training"]["epochs"]):
            self.anomaly_model.train()
            train_loss = 0.0

            for i in range(
                0, len(X_normal_tensor), self.config["training"]["batch_size"]
            ):
                batch_X = X_normal_tensor[i : i + self.config["training"]["batch_size"]]

                optimizer.zero_grad()
                reconstructed, encoded = self.anomaly_model(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_normal_tensor) // self.config["training"]["batch_size"]
            train_losses.append(train_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}")

            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save(self.anomaly_model.state_dict(), "best_anomaly_model.pt")

        # Load best model
        self.anomaly_model.load_state_dict(torch.load("best_anomaly_model.pt"))

        # Evaluate on validation set
        self.anomaly_model.eval()
        with torch.no_grad():
            val_reconstructed, _ = self.anomaly_model(X_val_tensor)
            reconstruction_errors = torch.mean(
                (X_val_tensor - val_reconstructed) ** 2, dim=1
            )
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        # Calculate threshold for anomaly detection
        normal_errors = reconstruction_errors[y_val == 0]
        self.anomaly_threshold = np.percentile(normal_errors, 95)  # 95th percentile

        # Evaluate performance
        (reconstruction_errors > self.anomaly_threshold).astype(int)
        auc_score = roc_auc_score(y_val, reconstruction_errors)

        logger.info(f"Anomaly detection AUC: {auc_score:.4f}")
        logger.info(f"Anomaly threshold: {self.anomaly_threshold:.6f}")

        return train_losses

    def train_compliance_model(self, X, y, validation_split=0.2):
        """Train the compliance violation detection model"""
        logger.info("Training compliance violation detection model")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Scale features
        X_train_scaled = self.compliance_scaler.fit_transform(X_train)
        X_val_scaled = self.compliance_scaler.transform(X_val)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.compliance_model = ComplianceViolationDetector(
            **self.config["compliance_model"]
        ).to(self.device)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.compliance_model.parameters(),
            lr=self.config["training"]["learning_rate"],
        )

        # Training loop
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []

        violation_types = [
            "aml_violation",
            "kyc_violation",
            "sanctions_violation",
            "transaction_limit_violation",
            "reporting_violation",
            "general_violation",
        ]

        for epoch in range(self.config["training"]["epochs"]):
            # Training
            self.compliance_model.train()
            train_loss = 0.0

            for i in range(
                0, len(X_train_tensor), self.config["training"]["batch_size"]
            ):
                batch_X = X_train_tensor[i : i + self.config["training"]["batch_size"]]
                batch_y = y_train_tensor[i : i + self.config["training"]["batch_size"]]

                optimizer.zero_grad()
                outputs = self.compliance_model(batch_X)

                # Calculate loss for each violation type
                loss = 0
                for j, violation_type in enumerate(violation_types):
                    loss += criterion(outputs[violation_type].squeeze(), batch_y[:, j])

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_train_tensor) // self.config["training"]["batch_size"]
            train_losses.append(train_loss)

            # Validation
            self.compliance_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i in range(
                    0, len(X_val_tensor), self.config["training"]["batch_size"]
                ):
                    batch_X = X_val_tensor[
                        i : i + self.config["training"]["batch_size"]
                    ]
                    batch_y = y_val_tensor[
                        i : i + self.config["training"]["batch_size"]
                    ]

                    outputs = self.compliance_model(batch_X)

                    loss = 0
                    for j, violation_type in enumerate(violation_types):
                        loss += criterion(
                            outputs[violation_type].squeeze(), batch_y[:, j]
                        )

                    val_loss += loss.item()

            val_loss /= len(X_val_tensor) // self.config["training"]["batch_size"]
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.compliance_model.state_dict(), "best_compliance_model.pt"
                )

        # Load best model
        self.compliance_model.load_state_dict(torch.load("best_compliance_model.pt"))

        logger.info("Compliance model training completed")
        return train_losses, val_losses

    def predict_risk(self, X):
        """Predict financial risks"""
        if self.risk_model is None:
            raise ValueError("Risk model not trained")

        # Scale input
        X_scaled = self.risk_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(
            X.shape
        )

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Predict
        self.risk_model.eval()
        with torch.no_grad():
            outputs = self.risk_model(X_tensor)

        # Convert to numpy
        predictions = {}
        for key, value in outputs.items():
            if key != "attention_weights":
                predictions[key] = value.cpu().numpy()

        return predictions

    def detect_anomalies(self, X):
        """Detect anomalies in transaction patterns"""
        if self.anomaly_model is None:
            raise ValueError("Anomaly model not trained")

        # Scale input
        X_scaled = self.anomaly_scaler.transform(X)

        # Autoencoder prediction
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.anomaly_model.eval()

        with torch.no_grad():
            reconstructed, encoded = self.anomaly_model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        # Isolation forest prediction
        isolation_predictions = self.isolation_forest.predict(X_scaled)
        isolation_scores = self.isolation_forest.score_samples(X_scaled)

        # Combine predictions
        autoencoder_anomalies = (reconstruction_errors > self.anomaly_threshold).astype(
            int
        )
        isolation_anomalies = (isolation_predictions == -1).astype(int)

        # Final anomaly score (ensemble)
        anomaly_scores = (
            reconstruction_errors / self.anomaly_threshold
            + (1 - (isolation_scores + 1) / 2)
        ) / 2

        return {
            "anomaly_scores": anomaly_scores,
            "autoencoder_anomalies": autoencoder_anomalies,
            "isolation_anomalies": isolation_anomalies,
            "reconstruction_errors": reconstruction_errors,
        }

    def predict_compliance_violations(self, X):
        """Predict compliance violations"""
        if self.compliance_model is None:
            raise ValueError("Compliance model not trained")

        # Scale input
        X_scaled = self.compliance_scaler.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Predict
        self.compliance_model.eval()
        with torch.no_grad():
            outputs = self.compliance_model(X_tensor)

        # Convert to numpy
        predictions = {}
        for key, value in outputs.items():
            predictions[key] = value.cpu().numpy()

        return predictions

    def save_models(self, base_path="financial_risk_models"):
        """Save all trained models and scalers"""
        import os

        os.makedirs(base_path, exist_ok=True)

        # Save models
        if self.risk_model:
            torch.save(self.risk_model.state_dict(), f"{base_path}/risk_model.pt")
        if self.anomaly_model:
            torch.save(self.anomaly_model.state_dict(), f"{base_path}/anomaly_model.pt")
        if self.compliance_model:
            torch.save(
                self.compliance_model.state_dict(), f"{base_path}/compliance_model.pt"
            )

        # Save scalers
        joblib.dump(self.risk_scaler, f"{base_path}/risk_scaler.pkl")
        joblib.dump(self.anomaly_scaler, f"{base_path}/anomaly_scaler.pkl")
        joblib.dump(self.compliance_scaler, f"{base_path}/compliance_scaler.pkl")

        # Save isolation forest
        joblib.dump(self.isolation_forest, f"{base_path}/isolation_forest.pkl")

        # Save config and threshold
        import json

        with open(f"{base_path}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        with open(f"{base_path}/anomaly_threshold.txt", "w") as f:
            f.write(str(self.anomaly_threshold))

        logger.info(f"Models saved to {base_path}")

    def load_models(self, base_path="financial_risk_models"):
        """Load all trained models and scalers"""
        import json
        import os

        # Load config
        with open(f"{base_path}/config.json", "r") as f:
            self.config = json.load(f)

        # Load models
        if os.path.exists(f"{base_path}/risk_model.pt"):
            self.risk_model = FinancialRiskLSTM(**self.config["risk_model"]).to(
                self.device
            )
            self.risk_model.load_state_dict(torch.load(f"{base_path}/risk_model.pt"))

        if os.path.exists(f"{base_path}/anomaly_model.pt"):
            self.anomaly_model = AnomalyDetectionModel(
                **self.config["anomaly_model"]
            ).to(self.device)
            self.anomaly_model.load_state_dict(
                torch.load(f"{base_path}/anomaly_model.pt")
            )

        if os.path.exists(f"{base_path}/compliance_model.pt"):
            self.compliance_model = ComplianceViolationDetector(
                **self.config["compliance_model"]
            ).to(self.device)
            self.compliance_model.load_state_dict(
                torch.load(f"{base_path}/compliance_model.pt")
            )

        # Load scalers
        self.risk_scaler = joblib.load(f"{base_path}/risk_scaler.pkl")
        self.anomaly_scaler = joblib.load(f"{base_path}/anomaly_scaler.pkl")
        self.compliance_scaler = joblib.load(f"{base_path}/compliance_scaler.pkl")

        # Load isolation forest
        self.isolation_forest = joblib.load(f"{base_path}/isolation_forest.pkl")

        # Load threshold
        with open(f"{base_path}/anomaly_threshold.txt", "r") as f:
            self.anomaly_threshold = float(f.read().strip())

        logger.info(f"Models loaded from {base_path}")


def main():
    """Main training and evaluation pipeline"""
    logger.info("Starting Financial Risk Predictor training pipeline")

    # Initialize predictor
    predictor = FinancialRiskPredictor()

    # Prepare and train risk model
    logger.info("Preparing risk prediction data...")
    X_risk, y_risk = predictor.prepare_risk_data(synthetic=True)
    predictor.train_risk_model(X_risk, y_risk)

    # Prepare and train anomaly detection model
    logger.info("Preparing anomaly detection data...")
    X_anomaly, y_anomaly = predictor.prepare_anomaly_data()
    predictor.train_anomaly_model(X_anomaly, y_anomaly)

    # Prepare and train compliance model
    logger.info("Preparing compliance data...")
    X_compliance, y_compliance = predictor.prepare_compliance_data()
    predictor.train_compliance_model(X_compliance, y_compliance)

    # Save all models
    predictor.save_models()

    # Test predictions
    logger.info("Testing predictions...")

    # Test risk prediction
    test_risk_data = X_risk[:5]  # First 5 samples
    risk_predictions = predictor.predict_risk(test_risk_data)
    logger.info(f"Risk predictions shape: {risk_predictions['overall_risk'].shape}")

    # Test anomaly detection
    test_anomaly_data = X_anomaly[:10]  # First 10 samples
    anomaly_predictions = predictor.detect_anomalies(test_anomaly_data)
    logger.info(f"Anomaly scores: {anomaly_predictions['anomaly_scores']}")

    # Test compliance prediction
    test_compliance_data = X_compliance[:5]  # First 5 samples
    compliance_predictions = predictor.predict_compliance_violations(
        test_compliance_data
    )
    logger.info(f"Compliance predictions keys: {list(compliance_predictions.keys())}")

    logger.info("Financial Risk Predictor training completed successfully!")


if __name__ == "__main__":
    main()
