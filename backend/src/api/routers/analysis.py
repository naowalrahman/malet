import traceback
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.constants import AI_ANALYSIS_PROMPT_FORMAT
from api.state import market_analysis_cache, data_fetcher, tech_indicators, genai_client


router = APIRouter()


def get_technical_analysis(symbol: str):
    cache_key = f"{symbol}_technical"
    if cache_key in market_analysis_cache and datetime.now() - market_analysis_cache[cache_key]["date"] < timedelta(days=1):
        return market_analysis_cache[cache_key]

    data = data_fetcher.fetch_daily_data(symbol, "60d")
    if data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

    data_with_indicators = tech_indicators.calculate_all_indicators(data)
    data_with_indicators = tech_indicators.get_signal_strength(data_with_indicators)
    latest_data = data_with_indicators.iloc[-1]

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
        "support_levels": [float(latest_data.get("S1", 0)), float(latest_data.get("S2", 0))],
        "resistance_levels": [float(latest_data.get("R1", 0)), float(latest_data.get("R2", 0))],
    }

    market_analysis_cache[cache_key] = analysis
    return analysis


@router.get("/market-analysis")
async def get_market_analysis(symbols: str):
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        analyses = [get_technical_analysis(symbol) for symbol in symbol_list]
        return analyses
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.get("/market-analysis/{symbol}/ai")
async def get_market_ai_analysis(symbol: str):
    try:
        symbol = symbol.upper()
        cache_key = f"{symbol}_ai"
        if cache_key in market_analysis_cache and datetime.now() - market_analysis_cache[cache_key]["date"] < timedelta(days=1):
            return market_analysis_cache[cache_key]

        technical_analysis = get_technical_analysis(symbol)
        asset_name = {
            "SPY": "S&P 500 ETF (SPY)",
            "DIA": "Dow Jones Industrial Average ETF (DIA)",
            "QQQ": "NASDAQ-100 ETF (QQQ)",
        }.get(symbol, f"{symbol} stock")

        signal_interpretation = {1: "bullish", 0: "neutral", -1: "bearish"}.get(technical_analysis["combined_signal"], "neutral")

        price_trend = "rising" if technical_analysis["price_change"] > 0 else "declining"
        rsi_interpretation = (
            "overbought territory" if technical_analysis["rsi"] > 70 else
            "oversold territory" if technical_analysis["rsi"] < 30 else
            "neutral range"
        )

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
            resistance_level_1=technical_analysis["resistance_levels"][1],
        )

        ai_analysis = await genai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        ai_result = {
            "symbol": symbol,
            "ai_analysis": ai_analysis.text,
            "date": datetime.now(),
            "generated_at": datetime.now().isoformat(),
        }

        market_analysis_cache[cache_key] = ai_result
        return ai_result
    except Exception as e:
        try:
            if e.code == 400:
                raise HTTPException(status_code=400, detail=e.message)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=traceback.format_exc())


