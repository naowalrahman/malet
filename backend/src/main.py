from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import uuid
from datetime import datetime, timedelta
import logging
import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_fetcher import get_popular_symbols, validate_symbol, DataFetcher
from indicators import TechnicalIndicators
from models.TradingModelTrainer import TradingModelTrainer
from backtesting_engine import BacktestEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Platform API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_jobs = {}
models = {}
model_names = {} # Only purpose is to pass to backtest engine
data_cache = {}
market_analysis_cache = {}

# Pydantic models for API
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
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestRequest(BaseModel):
    symbol: str
    model_ids: List[str]
    initial_capital: float = 10000
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PredictionRequest(BaseModel):
    symbol: str
    model_id: str

# Initialize components
data_fetcher = DataFetcher()
tech_indicators = TechnicalIndicators()

@app.get("/")
async def root():
    return {"message": "AI Trading Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Data endpoints
@app.get("/popular-symbols")
async def get_popular_symbols_endpoint():
    """Get list of popular stock symbols"""
    return {"symbols": get_popular_symbols()}

@app.post("/validate-symbol")
async def validate_symbol_endpoint(request: StockRequest):
    """Validate if a stock symbol exists"""
    is_valid = validate_symbol(request.symbol)
    return {"symbol": request.symbol, "valid": is_valid}

@app.post("/stock-data")
async def get_stock_data(request: StockRequest):
    """Fetch daily stock data"""
    try:
        data = data_fetcher.fetch_daily_data(request.symbol, request.period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Add technical indicators
        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        
        # Cache the data
        cache_key = f"{request.symbol}_{request.period}"
        data_cache[cache_key] = data_with_indicators
        
        # Convert to JSON-serializable format
        result = {
            "symbol": request.symbol,
            "period": request.period,
            "data_points": len(data_with_indicators),
            "start_date": data_with_indicators.index[0].isoformat(),
            "end_date": data_with_indicators.index[-1].isoformat(),
            "data": data_with_indicators.fillna(0).to_dict('records')[:1000]  # Limit for API response
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/historical-data")
async def get_historical_data(request: HistoricalDataRequest):
    """Fetch historical stock data for custom date range"""
    try:
        data = data_fetcher.fetch_historical_data(
            request.symbol, request.start_date, request.end_date
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Add technical indicators
        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        
        result = {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "data_points": len(data_with_indicators),
            "data": data_with_indicators.fillna(0).to_dict('records')[:1000]
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get basic information about a stock"""
    try:
        info = data_fetcher.get_stock_info(symbol)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Machine Learning endpoints

@app.get("/available-models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            {"model_id": "lstm", "model_type": "LSTM"}, 
            {"model_id": "cnn_lstm", "model_type": "CNN-LSTM"}, 
            {"model_id": "transformer", "model_type": "Transformer"},
            # {"model_id": "ensemble", "model_type": "Ensemble"}
        ]
    }

async def train_model_background(job_id: str, request: TrainingRequest):
    """Background task for model training"""
    try:
        training_jobs[job_id]["status"] = "fetching_data"
        training_jobs[job_id]["progress"] = 10
        
        # Fetch data
        if request.start_date and request.end_date:
            data = data_fetcher.fetch_historical_data(request.symbol, request.start_date, request.end_date)
        else:
            data = data_fetcher.fetch_daily_data(request.symbol, "6mo")
        
        if data.empty:
            training_jobs[job_id]["status"] = "error"
            training_jobs[job_id]["error"] = f"No data found for symbol {request.symbol}"
            return
        
        training_jobs[job_id]["status"] = "calculating_indicators"
        training_jobs[job_id]["progress"] = 20
        
        # Add technical indicators
        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        
        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["progress"] = 30
        
        # Initialize trainer
        trainer = TradingModelTrainer(
            model_type=request.model_type,
            sequence_length=request.sequence_length
        )
        
        # Train model
        training_result = trainer.train(
            data_with_indicators,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = training_result
        
        # Store the trained model
        models[job_id] = {
            "trainer": trainer,
            "symbol": request.symbol,
            "model_type": request.model_type,
            "training_result": training_result,
            "created_at": datetime.now().isoformat(),
            "training_params": {
                "sequence_length": request.sequence_length,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "prediction_horizon": request.prediction_horizon,
                "threshold": request.threshold,
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }
        
        model_names[job_id] = f"{request.symbol} - {request.model_type} ({(100 * training_result['final_metrics']['accuracy']):.1f}%)"
        
    except Exception as e:
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = str(e)

@app.post("/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    job_id = str(uuid.uuid4())
    
    training_jobs[job_id] = {
        "id": job_id,
        "status": "started",
        "progress": 0,
        "symbol": request.symbol,
        "model_type": request.model_type,
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(train_model_background, job_id, request)
    
    return {"job_id": job_id, "status": "started"}

@app.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@app.get("/trained-models")
async def list_trained_models():
    """List all trained models with detailed information"""
    model_list = []
    for model_id, model_info in models.items():
        training_result = model_info["training_result"]
        model_list.append({
            "model_id": model_id,
            "symbol": model_info["symbol"],
            "model_type": model_info["model_type"],
            "created_at": model_info["created_at"],
            "accuracy": training_result["final_metrics"].get("accuracy", 0),
            "training_params": model_info.get("training_params", {}),
            "training_metrics": training_result["final_metrics"],
            "training_history": training_result.get("training_history", {})
        })
    
    return {"models": model_list}


@app.delete("/trained-models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del models[model_id]
    if model_id in training_jobs:
        del training_jobs[model_id]
    
    return {"message": "Model deleted successfully"}

# Prediction endpoints
@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = models[request.model_id]
        trainer = model_info["trainer"]
        
        # Fetch recent data
        data = data_fetcher.fetch_daily_data(request.symbol, "1mo")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Add technical indicators
        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        
        # Make predictions
        predictions = trainer.predict(data_with_indicators)
        
        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Unable to make predictions with current data")
        
        # Get the latest prediction
        latest_prediction = int(predictions[-1])
        confidence = 0.8  # Placeholder - you could calculate actual confidence
        
        signal_map = {0: "DOWN", 1: "UP"}
        
        return {
            "symbol": request.symbol,
            "model_id": request.model_id,
            "prediction": latest_prediction,
            "signal": signal_map[latest_prediction],
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(data["Close"].iloc[-1])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting endpoints
@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run backtesting comparison with multiple models"""
    try:
        # Validate all models exist
        for model_id in request.model_ids:
            if model_id not in models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Fetch data for backtesting
        if request.start_date and request.end_date:
            data = data_fetcher.fetch_historical_data(
                request.symbol, request.start_date, request.end_date
            )
        else:
            data = data_fetcher.fetch_daily_data(request.symbol, "3mo")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Add technical indicators
        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        
        # Run backtesting
        backtest_engine = BacktestEngine(model_names)
        results = backtest_engine.run_comparison(
            data_with_indicators, 
            [models[model_id]["trainer"] for model_id in request.model_ids],
            request.model_ids,
            request.initial_capital
        )
        
        # Generate plots
        plots = backtest_engine.generate_plots()
        
        return {
            "symbol": request.symbol,
            "model_ids": request.model_ids,
            "initial_capital": request.initial_capital,
            "results": results,
            "plots": plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analysis endpoints
@app.get("/market-analysis")
async def get_market_analysis(symbols: str):
    """
    Get comprehensive market analysis for a list of symbols
    Expects a comma-separated list of symbols
    """
    try:
        # Parse comma-separated symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        analyses = []
        for symbol in symbol_list:
            if symbol in market_analysis_cache and datetime.now() - market_analysis_cache[symbol]["date"] < timedelta(days=1):
                print("Using cached market analysis")
                analyses.append(market_analysis_cache[symbol])
                continue
            
            print("Fetching new market analysis")

            # Fetch data
            data = data_fetcher.fetch_daily_data(symbol, "2mo")
            
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
            
            # Add technical indicators
            data_with_indicators = tech_indicators.calculate_all_indicators(data)
            
            # Calculate current signals
            signals = tech_indicators.get_signal_strength(data_with_indicators)
            
            latest_data = signals.iloc[-1]
            
            # Market analysis
            analysis = {
                "date": datetime.now(),
                "symbol": symbol,
                "current_price": float(latest_data["Close"]),
                "price_change": float(latest_data["Close"] - data["Close"].iloc[-2]),
                "price_change_pct": float((latest_data["Close"] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100),
                "volume": int(latest_data["Volume"]),
                "rsi": float(latest_data.get("RSI", 0)),
                "macd": float(latest_data.get("MACD", 0)),
                "bollinger_position": float(latest_data.get("BB_Position", 0)),
                "combined_signal": int(latest_data.get("Combined_Signal", 0)),
                "volatility": float(latest_data.get("Volatility", 0)),
                "support_levels": [
                    float(latest_data.get("S1", 0)),
                    float(latest_data.get("S2", 0))
                ],
                "resistance_levels": [
                    float(latest_data.get("R1", 0)),
                    float(latest_data.get("R2", 0))
                ]
            }
            
            market_analysis_cache[symbol] = analysis
            analyses.append(analysis)
            
        return analyses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance-metrics/{model_id}")
async def get_model_performance(model_id: str):
    """Get detailed model performance metrics"""
    try:
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = models[model_id]
        training_result = model_info["training_result"]
        
        return {
            "model_id": model_id,
            "symbol": model_info["symbol"],
            "model_type": model_info["model_type"],
            "training_metrics": training_result["final_metrics"],
            "training_history": training_result["training_history"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
