from pydantic import BaseModel
from typing import List


class StockRequest(BaseModel):
    symbol: str
    period: str = "1mo"


class HistoricalDataRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str


class TrainingRequest(BaseModel):
    symbol: str
    model_type: str = "lstm"
    sequence_length: int = 60
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    prediction_horizon: int = 5
    threshold: float = 0.02
    start_date: str
    end_date: str


class BacktestRequest(BaseModel):
    symbol: str
    model_ids: List[str]
    initial_capital: float = 10000
    start_date: str
    end_date: str


class PredictionRequest(BaseModel):
    symbol: str
    model_id: str
    date: str


