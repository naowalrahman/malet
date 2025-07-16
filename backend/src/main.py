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
from google import genai
from dotenv import load_dotenv

# For consistency:
# model_id = the auto-generated job id for a particular model training run
# model_type = the code (lstm, transformer, cnn_lstm, wavenet) used to identify the model architecture in the backend
# model_name = the name (LSTM, Transformer, CNN-LSTM, WaveNet) used to identify the model architecture in the UI

load_dotenv()

sys.path.append(os.path.dirname(__file__))

from data_fetcher import get_popular_symbols, validate_symbol, DataFetcher
from indicators import TechnicalIndicators
from models.TradingModelTrainer import TradingModelTrainer
from backtesting_engine import BacktestEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google GenAI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "no key")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="MALET API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_ANALYSIS_PROMPT_FORMAT = """
Analyze the current market conditions for {asset_name} based on the following technical indicators:

**Current Market Data:**
- Price: ${current_price:.2f} ({price_change_pct:+.2f}% today)
- RSI: {rsi:.1f} ({rsi_interpretation})
- MACD: {macd:.4f}
- Bollinger Band Position: {bollinger_position:.1f}%
- Average True Range: {average_true_range:.2f}%
- Combined Signal: {signal_interpretation}
- Support Levels: ${support_level_0:.2f}, ${support_level_1:.2f}
- Resistance Levels: ${resistance_level_0:.2f}, ${resistance_level_1:.2f}

**Instructions:**
Provide a concise market analysis in 2-3 paragraphs that focuses on:
1. Current market sentiment and momentum based on the technical indicators
2. Key price levels to watch (support/resistance) and potential trading opportunities
3. Risk assessment and what traders should be cautious about

**Format Requirements:**
- Write in professional, accessible language suitable for traders
- Use markdown formatting for emphasis. Specifically, use *italic* for key terms and **bold** for important levels
- Do not include headers or titles
- Keep the analysis practical and actionable
- Focus on insights that go beyond just restating the numbers
"""

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
            {"model_type": "lstm", "model_name": "LSTM"}, 
            {"model_type": "cnn_lstm", "model_name": "CNN-LSTM"}, 
            {"model_type": "transformer", "model_name": "Transformer"},
            {"model_type": "gru", "model_name": "GRU"},
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
            "model_id": model_id, # really the job id, but model_id is clearer on the frontend
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

# Analysis endpoints and functions

def get_technical_analysis(symbol: str):
    """
    Fetch and compute technical analysis for a given symbol.
    Uses cache if available and fresh, otherwise fetches new data.
    Returns the analysis dict.
    """
    cache_key = f"{symbol}_technical"
    # Check cache for technical analysis only
    if cache_key in market_analysis_cache and datetime.now() - market_analysis_cache[cache_key]["date"] < timedelta(days=1):
        print("Using cached technical analysis")
        return market_analysis_cache[cache_key]

    print("Fetching new technical analysis")
    # Fetch data
    data = data_fetcher.fetch_daily_data(symbol, "60d")
    if data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

    # Add technical indicators
    data_with_indicators = tech_indicators.calculate_all_indicators(data)

    # Calculate current signals
    signals = tech_indicators.get_signal_strength(data_with_indicators)
    latest_data = signals.iloc[-1]

    # Technical analysis (without AI)
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
        "average_true_range": float(latest_data.get("ATR", 0)),
        "support_levels": [
            float(latest_data.get("S1", 0)),
            float(latest_data.get("S2", 0))
        ],
        "resistance_levels": [
            float(latest_data.get("R1", 0)),
            float(latest_data.get("R2", 0))
        ]
    }

    market_analysis_cache[cache_key] = analysis
    return analysis

@app.get("/market-analysis")
async def get_market_analysis(symbols: str):
    """
    Get technical market analysis for a list of symbols (without AI analysis)
    Expects a comma-separated list of symbols
    """
    try:
        # Parse comma-separated symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        analyses = []
        for symbol in symbol_list:
            analysis = get_technical_analysis(symbol)
            analyses.append(analysis)
        return analyses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-analysis/{symbol}/ai")
async def get_market_ai_analysis(symbol: str):
    """
    Get AI-powered market analysis for a specific symbol
    """
    try:
        symbol = symbol.upper()
        
        # Check cache for AI analysis
        cache_key = f"{symbol}_ai"
        if cache_key in market_analysis_cache and datetime.now() - market_analysis_cache[cache_key]["date"] < timedelta(days=1):
            print("Using cached AI analysis")
            return market_analysis_cache[cache_key]
        
        print("Generating new AI analysis")
        
        technical_analysis = get_technical_analysis(symbol)
        
        asset_name = {
            "SPY": "S&P 500 ETF (SPY)",
            "DIA": "Dow Jones Industrial Average ETF (DIA)", 
            "QQQ": "NASDAQ-100 ETF (QQQ)"
        }.get(symbol, f"{symbol} stock")
        
        signal_interpretation = {
            1: "bullish",
            0: "neutral", 
            -1: "bearish"
        }.get(technical_analysis["combined_signal"], "neutral")
        
        # Get recent price trend
        price_trend = "rising" if technical_analysis["price_change"] > 0 else "declining"
        
        # RSI interpretation
        rsi_interpretation = ""
        if technical_analysis["rsi"] > 70:
            rsi_interpretation = "overbought territory"
        elif technical_analysis["rsi"] < 30:
            rsi_interpretation = "oversold territory"
        else:
            rsi_interpretation = "neutral range"
        
        prompt = AI_ANALYSIS_PROMPT_FORMAT.format(
            asset_name=asset_name,
            current_price=technical_analysis["current_price"],
            price_change_pct=technical_analysis["price_change_pct"],
            rsi=technical_analysis["rsi"],
            rsi_interpretation=rsi_interpretation,
            macd=technical_analysis["macd"],
            bollinger_position=technical_analysis["bollinger_position"] * 100,
            average_true_range=technical_analysis["average_true_range"] * 100,
            signal_interpretation=signal_interpretation,
            support_level_0=technical_analysis["support_levels"][0],
            support_level_1=technical_analysis["support_levels"][1],
            resistance_level_0=technical_analysis["resistance_levels"][0],
            resistance_level_1=technical_analysis["resistance_levels"][1]
        )

        ai_analysis = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        ai_result = {
            "symbol": symbol,
            "ai_analysis": ai_analysis.text,
            "date": datetime.now(),
            "generated_at": datetime.now().isoformat()
        }
        
        market_analysis_cache[cache_key] = ai_result
        
        return ai_result
    except Exception as e:
        if e.code == 400:
            raise HTTPException(status_code=400, detail=e.message)
        else:
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
