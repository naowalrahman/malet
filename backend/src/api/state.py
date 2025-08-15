import os
import sys
import logging
import pickle
from datetime import datetime
from typing import Dict, List, TypedDict

import pandas as pd
from dotenv import load_dotenv
from google import genai


# Ensure project src is importable for sibling modules like data_fetcher, indicators, etc.
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_fetcher import DataFetcher  # noqa: E402
from indicators import TechnicalIndicators  # noqa: E402
from models.TradingModelTrainer import TradingModelTrainer  # noqa: E402


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Typed structures for trained models stored in `models`
class TrainingHistory(TypedDict):
    train_losses: List[float]
    val_losses: List[float]
    val_accuracies: List[float]


class FinalMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions: List[int]
    true_labels: List[int]


class TrainingResult(TypedDict):
    training_history: TrainingHistory
    final_metrics: FinalMetrics
    model_type: str
    sequence_length: int


class TrainingParams(TypedDict):
    sequence_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    prediction_horizon: int
    threshold: float
    start_date: str
    end_date: str


class TrainedModelDetails(TypedDict):
    model_name: str
    trainer: TradingModelTrainer
    symbol: str
    model_type: str
    training_result: TrainingResult
    created_at: str
    training_params: TrainingParams

# Initialize Google GenAI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "no key")
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# Global state shared across routers
training_jobs: Dict[str, Dict] = {}
models: Dict[str, TrainedModelDetails] = {}
data_cache: Dict[str, pd.DataFrame] = {}
market_analysis_cache: Dict[str, Dict] = {}


# Initialize components
data_fetcher = DataFetcher()
tech_indicators = TechnicalIndicators()


# Load saved models
SAVED_MODELS_DIR = os.getenv("SAVED_MODELS_DIR")
if SAVED_MODELS_DIR:
    for file in os.listdir(SAVED_MODELS_DIR):
        with open(f"{SAVED_MODELS_DIR}/{file}", "rb") as f:
            job_id = file.split(".")[0]
            model = pickle.load(f)
            models[job_id] = model


