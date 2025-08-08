from models.LSTM import LSTM
from models.Transformer import Transformer
from models.CNN_LSTM import CNN_LSTM
from models.GRU import GRU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from typing import Dict, Tuple, List


class TradingModelTrainer:
    """
    Trainer class for trading models
    """

    def __init__(self, model_type: str = "lstm", sequence_length: int = 60):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model: nn.Module = None
        self.scaler: StandardScaler = None
        self.feature_columns: List[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history: Dict = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training
        """
        # Remove any columns with all NaN values
        df = df.dropna(axis=1, how='all')

        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target-related columns if they exist
        exclude_columns = {'Symbol', 'Close_Future', 'Target', 'Signal'}
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

        # Get the feature DataFrame
        feature_df = df[numeric_columns].copy()
        
        # Handle infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        
        # For each column, handle NaN values
        for col in feature_df.columns:
            if feature_df[col].isna().all():
                # If entire column is NaN, drop it
                feature_df = feature_df.drop(columns=[col])
            else:
                # Forward fill first, then backward fill, then fill remaining with median
                feature_df[col] = feature_df[col].ffill().bfill()
                if feature_df[col].isna().any():
                    median_val = feature_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    feature_df[col] = feature_df[col].fillna(median_val)

        # Store feature columns
        self.feature_columns = feature_df.columns.tolist()

        return feature_df

    def create_targets(self, df: pd.DataFrame, prediction_horizon: int = 5,
                      threshold: float = 0.001) -> pd.Series:
        """
        Create binary trading targets (0: Down, 1: Up)
        """
        # Calculate returns x (prediction_horizon) days into the future 
        # (% return = price in x days / price today - 1)
        future_returns = df['Close'].shift(-prediction_horizon) / df['Close'] - 1

        # Create binary targets: 1 if price goes up, 0 if price goes down
        # Use a small threshold to avoid noise
        targets = np.where(future_returns > threshold, 1, 0)  # 1: Up, 0: Down

        up_targets = np.sum(targets)
        num_targets = len(targets)
        print(f"Target distribution: Up={up_targets}, Down={num_targets - up_targets}")
        print(f"Target ratio: {(up_targets / num_targets):.3f} (Up ratio)")
        
        return pd.Series(targets, index=df.index)

    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        To see how this works, see: https://claude.ai/share/892517ac-889e-4ace-8bdf-d6ec87e6c216
        """
        n_samples = len(features) - self.sequence_length
        
        if n_samples <= 0:
            return np.array([]), np.array([])
        
        # Create indices for all sequences at once
        indices = np.arange(n_samples)[:, None] + np.arange(self.sequence_length)
        
        X = features[indices]
        y = targets[self.sequence_length:]
        
        return X, y

    def prepare_data(self, df: pd.DataFrame, prediction_horizon: int = 5,
                    threshold: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training
        """
        print(f"Initial data shape: {df.shape}")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        print(f"Features shape after preparation: {feature_df.shape}")

        # Create targets
        targets = self.create_targets(df, prediction_horizon, threshold)
        print(f"Targets shape: {targets.shape}")

        # Align features and targets by index
        common_index = feature_df.index.intersection(targets.index)
        feature_df = feature_df.loc[common_index]
        targets = targets.loc[common_index]

        # Remove rows with NaN targets (ideally we would not have any)
        valid_indices = ~targets.isna()
        feature_df = feature_df[valid_indices]
        targets = targets[valid_indices]
        
        print(f"Valid data shape after removing NaN targets: {feature_df.shape}")

        if len(feature_df) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length} due to sequence length {self.sequence_length}, got {len(feature_df)}")

        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(feature_df)
        
        # Check for NaN values after scaling
        if np.isnan(features_scaled).any():
            print("Warning: NaN values found after scaling. Replacing with zeros.")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        X, y = self.create_sequences(features_scaled, targets.values)
        
        print(f"Final sequence shapes - X: {X.shape}, y: {y.shape}")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        return X_tensor, y_tensor

    def initialize_model(self, input_size: int):
        """
        Initialize the model based on type
        """
        if self.model_type == "lstm":
            self.model = LSTM(input_size=input_size)
        elif self.model_type == "cnn_lstm":
            self.model = CNN_LSTM(input_size=input_size, sequence_length=self.sequence_length)
        elif self.model_type == "transformer":
            self.model = Transformer(input_size=input_size)
        elif self.model_type == "gru":
            self.model = GRU(input_size=input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.to(self.device)

    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, validation_split: float = 0.2, 
              progress_callback=None) -> Dict:
        """
        Train the model
        """
        # Prepare data
        X, y = self.prepare_data(df)

        # Check for class imbalance
        unique_classes, class_counts = torch.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes.cpu().numpy(), class_counts.cpu().numpy()))}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=0, stratify=y.cpu()
        )

        # Initialize model
        self.initialize_model(X.shape[2])

        # Initialize optimizer and loss function with class weights for imbalanced data
        if len(class_counts) == 2:
            # For binary classification, balance the classes better
            total_samples = torch.sum(class_counts)
            class_weights = total_samples / (2.0 * class_counts)
            class_weights = class_weights / torch.sum(class_weights) * 2.0  # Normalize
        else:
            class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
        
        class_weights = class_weights.to(self.device)
        print(f"Using class weights: {class_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)

        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            num_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Check for NaN values in outputs
                if torch.isnan(outputs).any():
                    print(f"NaN detected in model outputs at epoch {epoch}, batch {i//batch_size}")
                    continue
                
                loss = criterion(outputs, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {i//batch_size}")
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            if num_batches == 0:
                print(f"No valid batches in epoch {epoch}, stopping training")
                break

            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            val_batches = 0

            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]

                    outputs = self.model(batch_X)
                    
                    # Skip if NaN outputs
                    if torch.isnan(outputs).any():
                        continue
                        
                    loss = criterion(outputs, batch_y)
                    
                    # Skip if NaN loss
                    if torch.isnan(loss):
                        continue
                        
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    val_batches += 1

            if val_batches == 0:
                print(f"No valid validation batches in epoch {epoch}")
                continue

            # Calculate metrics
            avg_train_loss = train_loss / num_batches
            avg_val_loss = val_loss / val_batches
            val_accuracy = correct / total if total > 0 else 0

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Call progress callback if provided
            if progress_callback:
                progress_percentage = min(95, 30 + (epoch / epochs) * 65)  # 30% to 95% during training
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'progress': progress_percentage
                })

            # Print progress
            if epoch % 5 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
                
                # Debug: Check if model outputs are changing
                # if epoch == 0 or epoch % 10 == 0:
                #     with torch.no_grad():
                #         sample_outputs = self.model(X_val[:5])
                #         sample_probs = torch.softmax(sample_outputs, dim=1)
                #         print(f'Sample predictions probabilities: {sample_probs.cpu().numpy()}')
        
        print(f"Final train loss: {train_losses[-1]:.4f}, Final val loss: {val_losses[-1]:.4f}, Final val accuracy: {val_accuracies[-1]:.4f}")

        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

        # Final evaluation
        final_metrics = self.evaluate(X_val, y_val, batch_size)

        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length
        }

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> Dict:
        """
        Evaluate the model
        """
        self.model.eval()
        predictions = np.array([], dtype=int)
        true_labels = np.array([], dtype=int)

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)

                predictions = np.append(predictions, predicted.cpu().numpy())
                true_labels = np.append(true_labels, batch_y.cpu().numpy())

        # Calculate metrics
        accuracy = float(accuracy_score(true_labels, predictions))
        
        # Calculate macro averages for better class representation
        precision_macro = float(precision_score(true_labels, predictions, average='macro', zero_division=0))
        recall_macro = float(recall_score(true_labels, predictions, average='macro', zero_division=0))
        f1_macro = float(f1_score(true_labels, predictions, average='macro', zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist(),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")

        # Always prepare features the same way as during training
        print("Preparing features for prediction data...")
        feature_df = self.prepare_features(df)

        # Handle missing columns by adding them with default values
        for col in self.feature_columns:
            if col not in feature_df.columns:
                print(f"Adding missing feature '{col}' with default value 0.0")
                feature_df[col] = 0.0

        # Remove any extra columns that weren't in training
        feature_df = feature_df[self.feature_columns]
        
        # Handle any remaining NaN values
        feature_df = feature_df.ffill().bfill().fillna(0)

        # Scale features
        features_scaled = self.scaler.transform(feature_df)

        # Create sequences
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])

        if len(X) == 0:
            print(f"Warning: Not enough data for prediction. Need at least {self.sequence_length} samples, got {len(features_scaled)}")
            return np.array([])

        X = np.array(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Debug: print some probabilities to see what's happening
            # if len(probabilities) > 0:
            #     print(f"Sample probabilities: {probabilities[:5]}")
            #     print(f"Mean probabilities: {torch.mean(probabilities, dim=0)}")
            
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy()

    def save_model(self, filepath: str):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'training_history': self.training_history
        }, filepath)

    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model_type = checkpoint['model_type']
        self.sequence_length = checkpoint['sequence_length']
        self.feature_columns = checkpoint['feature_columns']
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint.get('training_history', [])

        # Initialize and load model
        if len(self.feature_columns) > 0:
            self.initialize_model(len(self.feature_columns))
            self.model.load_state_dict(checkpoint['model_state_dict'])