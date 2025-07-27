import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import uuid
from datetime import datetime, timedelta
import logging
import sys
import os
from google import genai
from dotenv import load_dotenv
import pickle
from fastapi.concurrency import run_in_threadpool
import tempfile

# For consistency:
# model_id = the auto-generated job id for a particular model training run
# model_type = the code (lstm, transformer, cnn_lstm, gru) used to identify the model architecture in the backend
# model_name = the name (LSTM, Transformer, CNN-LSTM, GRU) used to identify the model architecture in the UI

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
- Average True Range: ${average_true_range:.2f}
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
training_jobs: Dict[str, Dict] = {}
models: Dict[str, Dict] = {}
data_cache: Dict[str, pd.DataFrame] = {}
market_analysis_cache: Dict[str, Dict] = {}

# Load saved models
SAVED_MODELS_DIR = os.getenv("SAVED_MODELS_DIR")
for file in os.listdir(SAVED_MODELS_DIR):
    with open(f"{SAVED_MODELS_DIR}/{file}", "rb") as f:
        job_id = file.split(".")[0]
        model = pickle.load(f)
        models[job_id] = model

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
        raise HTTPException(status_code=500, detail=traceback.format_exc())

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
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get basic information about a stock"""
    try:
        info = data_fetcher.get_stock_info(symbol)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

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
        def fetch_data():
            training_jobs[job_id]["status"] = "fetching_data"
            training_jobs[job_id]["progress"] = 10
            
            # Fetch data
            data = data_fetcher.fetch_historical_data(request.symbol, request.start_date, request.end_date)
            
            if data.empty:
                training_jobs[job_id]["status"] = "error"
                training_jobs[job_id]["error"] = f"No data found for symbol {request.symbol}"
                return None
            
            return data
        
        data = await run_in_threadpool(fetch_data)
        
        if data is None:
            return
        
        def do_training():
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
            
            # Progress callback function
            def progress_callback(progress_info):
                training_jobs[job_id].update({
                    "status": "training",
                    "progress": progress_info["progress"],
                    "current_epoch": progress_info["epoch"],
                    "total_epochs": progress_info["total_epochs"],
                    "train_loss": progress_info["train_loss"],
                    "val_loss": progress_info["val_loss"],
                    "val_accuracy": progress_info["val_accuracy"]
                })
            
            # Train model with progress callback
            training_result = trainer.train(
                data_with_indicators,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                progress_callback=progress_callback
            )

            return training_result, trainer

        training_result, trainer = await run_in_threadpool(do_training)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = training_result
        
        # Store the trained model
        models[job_id] = {
            "model_name": f"{request.symbol} - {request.model_type} ({(100 * training_result['final_metrics']['accuracy']):.1f}%)",
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

        # serialize and save models[job_id]
        with open(f"{SAVED_MODELS_DIR}/{job_id}.pkl", "wb") as file:
            pickle.dump(models[job_id], file)
        
        
    except Exception as e:
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = traceback.format_exc()

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
    os.remove(f"{SAVED_MODELS_DIR}/{model_id}.pkl")
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
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    
async def get_backtest_data(request: BacktestRequest):
    try:
        max_sequence_length = 0
        for model_id in request.model_ids:
            if model_id not in models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            max_sequence_length = max(max_sequence_length, models[model_id]["training_params"]["sequence_length"])
        
        # Fetch data for backtesting
        test_start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        # subtracting 52 weeks from the test start date ensures there is enough padding at the beginning
        # to account for the maximum permissible sequence length (240 trading days)
        fetch_start_date = test_start_date - timedelta(weeks=52)
        padding_data = data_fetcher.fetch_historical_data(request.symbol, fetch_start_date.strftime("%Y-%m-%d"), request.start_date)
        test_data = data_fetcher.fetch_historical_data(request.symbol, request.start_date, request.end_date)
        
        if padding_data.empty or test_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Add technical indicators
        padding_data_with_indicators = tech_indicators.calculate_all_indicators(padding_data)
        test_data_with_indicators = tech_indicators.calculate_all_indicators(test_data)

        return padding_data_with_indicators, test_data_with_indicators, max_sequence_length
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

# Backtesting endpoints
@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run backtesting comparison with multiple models"""
    try:
        padding_data_with_indicators, test_data_with_indicators, max_sequence_length = await get_backtest_data(request)
        
        # Run backtesting
        backtest_engine = BacktestEngine(models)
        results = backtest_engine.run_comparison(
            padding_data_with_indicators, 
            test_data_with_indicators,
            request.model_ids,
            request.initial_capital,
            max_sequence_length
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
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.post("/backtest/export")
async def export_backtest_results(request: BacktestRequest):
    """Export backtest results as Excel file with detailed trade information"""
    
    def convert_to_timezone_naive(date_obj):
        """Convert any date object to timezone-naive datetime for Excel compatibility"""
        if date_obj is None:
            return None
        
        if isinstance(date_obj, str):
            dt = pd.to_datetime(date_obj)
        else:
            dt = date_obj
        
        # If it's a pandas Timestamp with timezone
        if hasattr(dt, 'tz') and dt.tz is not None:
            return dt.tz_localize(None)
        # If it's a regular datetime with timezone
        elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        # If it's already timezone-naive
        else:
            return dt
    
    try:
        padding_data_with_indicators, test_data_with_indicators, max_sequence_length = await get_backtest_data(request)
        
        # Run backtesting
        backtest_engine = BacktestEngine(models)
        results = backtest_engine.run_comparison(
            padding_data_with_indicators, 
            test_data_with_indicators,
            request.model_ids,
            request.initial_capital,
            max_sequence_length
        )
        
        # Generate Excel file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        excel_path = temp_file.name
        temp_file.close()
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet with comparison metrics
            summary_data = []
            
            # Add Buy & Hold data
            bh_data = results['buy_and_hold']
            summary_data.append({
                'Strategy': 'Buy & Hold',
                'Model ID': 'N/A',
                'Final Value': bh_data['final_value'],
                'Total Return': bh_data['total_return'],
                'Total Trades': bh_data['total_trades'],
                'Sharpe Ratio': bh_data.get('sharpe_ratio', 0),
                'Max Drawdown': bh_data.get('max_drawdown', 0),
                'Win Rate': bh_data.get('win_rate', 0),
                'Volatility': bh_data.get('volatility', 0),
                'Calmar Ratio': bh_data.get('calmar_ratio', 0)
            })
            
            # Add ML strategies data
            for model_id in request.model_ids:
                ml_data = results['ml_strategies'][model_id]
                model_info = models[model_id]
                
                summary_data.append({
                    'Strategy': model_info['model_name'],
                    'Model ID': model_id,
                    'Final Value': ml_data['final_value'],
                    'Total Return': ml_data['total_return'],
                    'Total Trades': ml_data['total_trades'],
                    'Sharpe Ratio': ml_data.get('sharpe_ratio', 0),
                    'Max Drawdown': ml_data.get('max_drawdown', 0),
                    'Win Rate': ml_data.get('win_rate', 0),
                    'Volatility': ml_data.get('volatility', 0),
                    'Calmar Ratio': ml_data.get('calmar_ratio', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create detailed sheets for each ML model
            for model_id in request.model_ids:
                ml_data = results['ml_strategies'][model_id]
                model_info = models[model_id]
                
                trades_data = []
                portfolio_values = ml_data.get('portfolio_values', [])
                dates = ml_data.get('dates', [])
                predictions = ml_data.get('predictions', [])
                
                trades_by_date = {}
                for trade in ml_data.get('trades', []):
                    trade_date = convert_to_timezone_naive(trade['date'])
                    # Convert to date for comparison
                    if hasattr(trade_date, 'date'):
                        trade_date = trade_date.date()
                    trades_by_date[trade_date] = trade
                
                # Track buy prices for gain/loss calculation
                buy_price_stack = []  
                
                # Create comprehensive daily log 
                for i, date in enumerate(dates):
                    portfolio_value = portfolio_values[i]
                    prediction = predictions[i]
                    
                    current_date = convert_to_timezone_naive(date)
                    if hasattr(current_date, 'date'):
                        current_date = current_date.date()
                    
                    trade_on_date = trades_by_date.get(current_date)
                    
                    daily_return = 0
                    if i > 0 and portfolio_values[i-1] > 0:
                        daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
                    
                    gain_loss_amount = 0
                    gain_loss_pct = 0
                    
                    if trade_on_date:
                        if trade_on_date['action'] in ['BUY', 'INITIAL_BUY']:
                            # Record buy price for future gain/loss calculation
                            buy_price_stack.append({
                                'price': trade_on_date['price'],
                                'shares': trade_on_date['shares'],
                                'date': date
                            })
                        elif trade_on_date['action'] == 'SELL' and buy_price_stack:
                            # Calculate gain/loss using most recent buy
                            last_buy = buy_price_stack.pop()
                            buy_price = last_buy['price']
                            sell_price = trade_on_date['price']
                            shares = trade_on_date['shares']
                            
                            gain_loss_amount = (sell_price - buy_price) * shares
                            gain_loss_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    row = {
                        'Date': convert_to_timezone_naive(date),
                        'Portfolio_Value': portfolio_value,
                        'Daily_Return_Pct': daily_return * 100,
                        'Prediction': 'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'N/A',
                        'Trade_Action': trade_on_date['action'] if trade_on_date else 'HOLD',
                        'Trade_Shares': trade_on_date['shares'] if trade_on_date else 0,
                        'Trade_Price': trade_on_date['price'] if trade_on_date else 0,
                        'Trade_Value': trade_on_date['value'] if trade_on_date else 0,
                        'Trade_Fee': trade_on_date['fee'] if trade_on_date else 0,
                        'Gain_Loss_Amount': gain_loss_amount,
                        'Gain_Loss_Pct': gain_loss_pct
                    }
                    
                    trades_data.append(row)
                
                # Create DataFrame and save to sheet
                sheet_name = model_info['model_name'][:31]  # Excel sheet name limit
                
                trades_df = pd.DataFrame(trades_data)
                
                # Ensure Date column is timezone-naive
                if 'Date' in trades_df.columns:
                    trades_df['Date'] = trades_df['Date'].apply(convert_to_timezone_naive)
                
                trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format the sheet
                worksheet = writer.sheets[sheet_name]
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 20)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Return the file
        filename = f"backtest_results_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return FileResponse(
            path=excel_path,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())

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

    # Calculate combined signal strength
    data_with_indicators = tech_indicators.get_signal_strength(data_with_indicators)
    latest_data = data_with_indicators.iloc[-1]

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
        raise HTTPException(status_code=500, detail=traceback.format_exc())

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
            average_true_range=technical_analysis["average_true_range"],
            signal_interpretation=signal_interpretation,
            support_level_0=technical_analysis["support_levels"][0],
            support_level_1=technical_analysis["support_levels"][1],
            resistance_level_0=technical_analysis["resistance_levels"][0],
            resistance_level_1=technical_analysis["resistance_levels"][1]
        )

        ai_analysis = await genai_client.aio.models.generate_content(
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
            raise HTTPException(status_code=500, detail=traceback.format_exc())

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
        raise HTTPException(status_code=500, detail=traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
