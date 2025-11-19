import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection system for blockchain transactions and supply chain events
    """

    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples="auto",
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fit the anomaly detection models on training data

        Args:
            X (numpy.ndarray): Training data of shape (n_samples, n_features)
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)

        # Fit DBSCAN
        self.dbscan.fit(X_scaled)

        self.is_fitted = True
        logger.info("Anomaly detection models fitted successfully")

    def predict(self, X):
        """
        Predict anomalies in new data

        Args:
            X (numpy.ndarray): Data to predict anomalies for, shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Binary array where -1 indicates anomaly, 1 indicates normal
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Predict with Isolation Forest
        if_predictions = self.isolation_forest.predict(X_scaled)

        # Predict with DBSCAN (cluster labels, -1 for outliers)
        dbscan_labels = self.dbscan.fit_predict(X_scaled)
        dbscan_predictions = np.where(dbscan_labels == -1, -1, 1)

        # Combine predictions (if either model flags as anomaly, consider it anomaly)
        combined_predictions = np.where(
            (if_predictions == -1) | (dbscan_predictions == -1),
            -1,  # Anomaly
            1,  # Normal
        )

        return combined_predictions

    def anomaly_score(self, X):
        """
        Calculate anomaly scores for data points

        Args:
            X (numpy.ndarray): Data to calculate anomaly scores for

        Returns:
            numpy.ndarray: Anomaly scores where higher values indicate more anomalous
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Get decision function from Isolation Forest (negative of anomaly score)
        # Convert to positive anomaly score where higher = more anomalous
        if_scores = -self.isolation_forest.decision_function(X_scaled)

        return if_scores

    def save(self, filepath):
        """
        Save the anomaly detection model

        Args:
            filepath (str): Path to save the model
        """
        import joblib

        joblib.dump(
            {
                "isolation_forest": self.isolation_forest,
                "dbscan": self.dbscan,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "contamination": self.contamination,
                "random_state": self.random_state,
            },
            filepath,
        )
        logger.info(f"Anomaly detection model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved anomaly detection model

        Args:
            filepath (str): Path to the saved model

        Returns:
            AnomalyDetector: Loaded model
        """
        import joblib

        data = joblib.load(filepath)

        detector = cls(
            contamination=data["contamination"], random_state=data["random_state"]
        )
        detector.isolation_forest = data["isolation_forest"]
        detector.dbscan = data["dbscan"]
        detector.scaler = data["scaler"]
        detector.is_fitted = data["is_fitted"]

        logger.info(f"Anomaly detection model loaded from {filepath}")
        return detector


class ModelVersionManager:
    """
    Manages model versioning and A/B testing
    """

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.active_models = {}
        self.model_metrics = {}
        self.ab_test_config = {}

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Load existing models if available
        self._load_model_registry()

    def _load_model_registry(self):
        """Load the model registry from disk"""
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)

                self.models = registry.get("models", {})
                self.active_models = registry.get("active_models", {})
                self.model_metrics = registry.get("model_metrics", {})
                self.ab_test_config = registry.get("ab_test_config", {})

                logger.info(f"Loaded model registry with {len(self.models)} models")
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")

    def _save_model_registry(self):
        """Save the model registry to disk"""
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        try:
            registry = {
                "models": self.models,
                "active_models": self.active_models,
                "model_metrics": self.model_metrics,
                "ab_test_config": self.ab_test_config,
                "last_updated": datetime.now().isoformat(),
            }

            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            logger.info(f"Saved model registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def register_model(self, model_type, model_path, version, metadata=None):
        """
        Register a new model version

        Args:
            model_type (str): Type of model (e.g., 'liquidity', 'supply_chain')
            model_path (str): Path to the saved model file
            version (str): Version identifier
            metadata (dict, optional): Additional metadata about the model

        Returns:
            str: Full model ID
        """
        if model_type not in self.models:
            self.models[model_type] = {}

        model_id = f"{model_type}_v{version}"

        self.models[model_type][version] = {
            "model_id": model_id,
            "model_path": model_path,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Initialize metrics for this model
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {
                "inference_count": 0,
                "avg_inference_time": 0,
                "error_count": 0,
                "performance_metrics": {},
            }

        self._save_model_registry()
        logger.info(f"Registered model {model_id}")

        return model_id

    def activate_model(self, model_type, version):
        """
        Activate a specific model version as the primary model

        Args:
            model_type (str): Type of model
            version (str): Version to activate
        """
        if model_type not in self.models or version not in self.models[model_type]:
            raise ValueError(f"Model {model_type} version {version} not found")

        self.active_models[model_type] = version
        self._save_model_registry()
        logger.info(f"Activated {model_type} version {version}")

    def get_active_model_path(self, model_type):
        """
        Get the path to the currently active model

        Args:
            model_type (str): Type of model

        Returns:
            str: Path to the active model file
        """
        if model_type not in self.active_models:
            raise ValueError(f"No active model for type {model_type}")

        version = self.active_models[model_type]
        return self.models[model_type][version]["model_path"]

    def configure_ab_test(self, model_type, versions, traffic_split):
        """
        Configure A/B testing between model versions

        Args:
            model_type (str): Type of model
            versions (list): List of versions to test
            traffic_split (list): Percentage of traffic for each version (must sum to 100)
        """
        if len(versions) != len(traffic_split):
            raise ValueError(
                "Number of versions must match number of traffic split values"
            )

        if sum(traffic_split) != 100:
            raise ValueError("Traffic split must sum to 100")

        # Verify all versions exist
        for version in versions:
            if model_type not in self.models or version not in self.models[model_type]:
                raise ValueError(f"Model {model_type} version {version} not found")

        self.ab_test_config[model_type] = {
            "versions": versions,
            "traffic_split": traffic_split,
            "started_at": datetime.now().isoformat(),
            "active": True,
        }

        self._save_model_registry()
        logger.info(f"Configured A/B test for {model_type} with versions {versions}")

    def select_model_for_request(self, model_type, request_id=None):
        """
        Select a model version based on A/B testing configuration

        Args:
            model_type (str): Type of model
            request_id (str, optional): Unique request identifier for deterministic selection

        Returns:
            str: Selected model version
        """
        # If no A/B test is active, return the default active model
        if (
            model_type not in self.ab_test_config
            or not self.ab_test_config[model_type]["active"]
        ):
            return self.active_models.get(model_type)

        # Get A/B test configuration
        config = self.ab_test_config[model_type]
        versions = config["versions"]
        traffic_split = config["traffic_split"]

        # If request_id is provided, use it for deterministic selection
        if request_id:
            # Hash the request_id to get a deterministic value
            import hashlib

            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        else:
            # Otherwise use random selection
            hash_value = np.random.randint(0, 100)

        # Select version based on traffic split
        cumulative = 0
        for i, split in enumerate(traffic_split):
            cumulative += split
            if hash_value < cumulative:
                return versions[i]

        # Fallback to the last version
        return versions[-1]

    def record_inference(self, model_id, inference_time, error=False):
        """
        Record inference statistics for a model

        Args:
            model_id (str): Model identifier
            inference_time (float): Time taken for inference in seconds
            error (bool): Whether an error occurred during inference
        """
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {
                "inference_count": 0,
                "avg_inference_time": 0,
                "error_count": 0,
                "performance_metrics": {},
            }

        metrics = self.model_metrics[model_id]

        # Update metrics
        count = metrics["inference_count"]
        avg_time = metrics["avg_inference_time"]

        # Update average inference time
        metrics["inference_count"] += 1
        metrics["avg_inference_time"] = (avg_time * count + inference_time) / (
            count + 1
        )

        if error:
            metrics["error_count"] += 1

        # Save periodically (every 100 inferences)
        if metrics["inference_count"] % 100 == 0:
            self._save_model_registry()

    def update_performance_metrics(self, model_id, metrics):
        """
        Update performance metrics for a model

        Args:
            model_id (str): Model identifier
            metrics (dict): Performance metrics to update
        """
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {
                "inference_count": 0,
                "avg_inference_time": 0,
                "error_count": 0,
                "performance_metrics": {},
            }

        # Update metrics
        self.model_metrics[model_id]["performance_metrics"].update(metrics)
        self._save_model_registry()

    def get_model_metrics(self, model_id=None):
        """
        Get metrics for a specific model or all models

        Args:
            model_id (str, optional): Model identifier

        Returns:
            dict: Model metrics
        """
        if model_id:
            return self.model_metrics.get(model_id, {})
        else:
            return self.model_metrics

    def generate_metrics_report(self, output_path=None):
        """
        Generate a report of model metrics

        Args:
            output_path (str, optional): Path to save the report

        Returns:
            dict: Report data
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "models": self.models,
            "active_models": self.active_models,
            "metrics": self.model_metrics,
            "ab_tests": self.ab_test_config,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            # Generate plots
            self._generate_metrics_plots(os.path.dirname(output_path))

        return report

    def _generate_metrics_plots(self, output_dir):
        """Generate plots for model metrics"""
        # Plot inference times
        plt.figure(figsize=(12, 6))
        model_ids = []
        avg_times = []

        for model_id, metrics in self.model_metrics.items():
            if metrics["inference_count"] > 0:
                model_ids.append(model_id)
                avg_times.append(metrics["avg_inference_time"])

        if model_ids:
            plt.bar(model_ids, avg_times)
            plt.xlabel("Model")
            plt.ylabel("Average Inference Time (s)")
            plt.title("Average Inference Time by Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "inference_times.png"))

        # Plot error rates
        plt.figure(figsize=(12, 6))
        error_rates = []

        for model_id, metrics in self.model_metrics.items():
            if metrics["inference_count"] > 0:
                error_rate = metrics["error_count"] / metrics["inference_count"] * 100
                error_rates.append(error_rate)
            else:
                error_rates.append(0)

        if model_ids:
            plt.bar(model_ids, error_rates)
            plt.xlabel("Model")
            plt.ylabel("Error Rate (%)")
            plt.title("Error Rate by Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "error_rates.png"))


def train_anomaly_detection_model(data_path=None, save_path="anomaly_detector.pkl"):
    """
    Train an anomaly detection model for blockchain transactions

    Args:
        data_path (str, optional): Path to training data
        save_path (str): Path to save the model

    Returns:
        AnomalyDetector: Trained anomaly detection model
    """
    logger.info("Training anomaly detection model")

    # Load or generate synthetic data
    try:
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded anomaly detection training data from {data_path}")
        else:
            # Generate synthetic data for demonstration
            logger.info("Generating synthetic data for anomaly detection training")
            np.random.seed(42)
            n_samples = 1000
            n_features = 10

            # Create normal data
            normal_data = np.random.randn(n_samples, n_features)

            # Create anomalies (5% of data)
            n_anomalies = int(n_samples * 0.05)
            anomalies = np.random.randn(n_anomalies, n_features) * 5 + 5

            # Combine data
            X = np.vstack([normal_data, anomalies])

            # Create labels (1 for normal, -1 for anomaly)
            y = np.ones(n_samples + n_anomalies)
            y[n_samples:] = -1

            # Create DataFrame
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
            df["label"] = y
    except Exception as e:
        logger.error(f"Error loading or generating data: {e}")
        raise

    # Prepare data
    if "label" in df.columns:
        X = df.drop("label", axis=1).values
        y = df["label"].values
    else:
        X = df.values
        y = None

    # Initialize and train anomaly detector
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)

    # Evaluate if labels are available
    if y is not None:
        predictions = detector.predict(X)
        accuracy = np.mean(predictions == y)
        logger.info(f"Anomaly detection accuracy: {accuracy:.4f}")

    # Save model
    detector.save(save_path)
    logger.info(f"Anomaly detection model saved to {save_path}")

    return detector


def setup_model_versioning():
    """
    Set up model versioning and A/B testing framework

    Returns:
        ModelVersionManager: Configured model version manager
    """
    logger.info("Setting up model versioning and A/B testing framework")

    # Initialize model version manager
    manager = ModelVersionManager(model_dir="models")

    # Register initial models if they exist
    if os.path.exists("liquidity_predictor.pt"):
        manager.register_model(
            model_type="liquidity",
            model_path="liquidity_predictor.pt",
            version="1.0",
            metadata={
                "description": "Enhanced liquidity prediction model",
                "architecture": "LSTM with attention",
                "training_date": datetime.now().isoformat(),
            },
        )
        manager.activate_model("liquidity", "1.0")

    if os.path.exists("supply_chain_forecaster.pt"):
        manager.register_model(
            model_type="supply_chain",
            model_path="supply_chain_forecaster.pt",
            version="1.0",
            metadata={
                "description": "Supply chain forecasting model",
                "architecture": "CNN-LSTM hybrid",
                "training_date": datetime.now().isoformat(),
            },
        )
        manager.activate_model("supply_chain", "1.0")

    if os.path.exists("anomaly_detector.pkl"):
        manager.register_model(
            model_type="anomaly",
            model_path="anomaly_detector.pkl",
            version="1.0",
            metadata={
                "description": "Transaction anomaly detection model",
                "architecture": "Isolation Forest + DBSCAN",
                "training_date": datetime.now().isoformat(),
            },
        )
        manager.activate_model("anomaly", "1.0")

    # Generate initial metrics report
    manager.generate_metrics_report("models/initial_metrics_report.json")

    logger.info("Model versioning and A/B testing framework set up successfully")
    return manager


if __name__ == "__main__":
    # Train anomaly detection model
    anomaly_model = train_anomaly_detection_model(
        data_path=None, save_path="anomaly_detector.pkl"  # Use synthetic data
    )

    # Set up model versioning
    version_manager = setup_model_versioning()

    logger.info("ML model enhancements completed successfully")
