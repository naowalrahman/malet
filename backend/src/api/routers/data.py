import traceback
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import StockRequest, HistoricalDataRequest
from api.state import data_fetcher, tech_indicators, data_cache
from data_fetcher import get_popular_symbols, validate_symbol


router = APIRouter()


@router.get("/popular-symbols")
async def get_popular_symbols_endpoint():
    return {"symbols": get_popular_symbols()}


@router.post("/validate-symbol")
async def validate_symbol_endpoint(request: StockRequest):
    is_valid = validate_symbol(request.symbol)
    return {"symbol": request.symbol, "valid": is_valid}


@router.post("/stock-data")
async def get_stock_data(request: StockRequest):
    try:
        data = data_fetcher.fetch_daily_data(request.symbol, request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        data_with_indicators = tech_indicators.calculate_all_indicators(data)

        cache_key = f"{request.symbol}_{request.period}"
        data_cache[cache_key] = data_with_indicators

        result = {
            "symbol": request.symbol,
            "period": request.period,
            "data_points": len(data_with_indicators),
            "start_date": data_with_indicators.index[0].isoformat(),
            "end_date": data_with_indicators.index[-1].isoformat(),
            "data": data_with_indicators.fillna(0).to_dict('records')[:1000],
        }
        return result
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post("/historical-data")
async def get_historical_data(request: HistoricalDataRequest):
    try:
        data = data_fetcher.fetch_historical_data(
            request.symbol, request.start_date, request.end_date
        )
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        data_with_indicators = tech_indicators.calculate_all_indicators(data)

        result = {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "data_points": len(data_with_indicators),
            "data": data_with_indicators.fillna(0).to_dict('records')[:1000],
        }
        return result
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    try:
        info = data_fetcher.get_stock_info(symbol)
        return info
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


