from models.TradingModelTrainer import TradingModelTrainer


import numpy as np
import pandas as pd


from typing import Dict, List


class EnsembleModel:
    """
    Ensemble of multiple trading models
    """

    def __init__(self, model_types: List[str] = ["lstm", "cnn_lstm", "transformer"]):
        self.models = {}
        self.model_types = model_types
        self.weights = None

    def train(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Train all models in the ensemble
        """
        results = {}

        for model_type in self.model_types:
            print(f"Training {model_type} model...")
            trainer = TradingModelTrainer(model_type=model_type)
            result = trainer.train(df, **kwargs)

            self.models[model_type] = trainer
            results[model_type] = result

        # Calculate ensemble weights based on validation accuracy
        val_accuracies = [results[model_type]['final_metrics']['accuracy']
                         for model_type in self.model_types]

        # Softmax weights
        exp_accuracies = np.exp(np.array(val_accuracies))
        self.weights = exp_accuracies / np.sum(exp_accuracies)

        return results

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        """
        predictions = {}

        for model_type, trainer in self.models.items():
            pred = trainer.predict(df)
            if len(pred) > 0:
                predictions[model_type] = pred

        if not predictions:
            return np.array([])

        # Weighted voting
        min_length = min(len(pred) for pred in predictions.values())
        ensemble_pred = np.zeros(min_length)

        for i, model_type in enumerate(self.model_types):
            if model_type in predictions:
                ensemble_pred += self.weights[i] * predictions[model_type][:min_length]

        return np.round(ensemble_pred).astype(int)