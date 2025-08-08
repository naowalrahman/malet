import tempfile
import traceback
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas import BacktestRequest
from api.state import data_fetcher, tech_indicators, models
from backtesting_engine import BacktestEngine


router = APIRouter()


async def get_backtest_data(request: BacktestRequest):
    try:
        max_sequence_length = 0
        for model_id in request.model_ids:
            if model_id not in models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            max_sequence_length = max(max_sequence_length, models[model_id]["training_params"]["sequence_length"])

        test_start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        fetch_start_date = test_start_date - timedelta(weeks=52)
        padding_data = data_fetcher.fetch_historical_data(
            request.symbol, fetch_start_date.strftime("%Y-%m-%d"), request.start_date
        )
        test_data = data_fetcher.fetch_historical_data(request.symbol, request.start_date, request.end_date)

        if padding_data.empty or test_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        data_with_indicators = tech_indicators.calculate_all_indicators(pd.concat([padding_data, test_data]))
        padding_data_with_indicators = data_with_indicators.iloc[: len(padding_data)]
        test_data_with_indicators = data_with_indicators.iloc[len(padding_data) :]

        return padding_data_with_indicators, test_data_with_indicators, max_sequence_length
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    try:
        padding_data_with_indicators, test_data_with_indicators, max_sequence_length = await get_backtest_data(request)

        backtest_engine = BacktestEngine(models)
        results = backtest_engine.run_comparison(
            padding_data_with_indicators,
            test_data_with_indicators,
            request.model_ids,
            request.initial_capital,
            max_sequence_length,
        )

        plots = backtest_engine.generate_plots()

        return {
            "symbol": request.symbol,
            "model_ids": request.model_ids,
            "initial_capital": request.initial_capital,
            "results": results,
            "plots": plots,
        }
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post("/backtest/export")
async def export_backtest_results(request: BacktestRequest):
    def convert_to_timezone_naive(date_obj):
        if date_obj is None:
            return None
        dt = pd.to_datetime(date_obj) if isinstance(date_obj, str) else date_obj
        if hasattr(dt, "tz") and dt.tz is not None:
            return dt.tz_localize(None)
        elif hasattr(dt, "tzinfo") and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        else:
            return dt

    try:
        padding_data_with_indicators, test_data_with_indicators, max_sequence_length = await get_backtest_data(request)

        backtest_engine = BacktestEngine(models)
        results = backtest_engine.run_comparison(
            padding_data_with_indicators,
            test_data_with_indicators,
            request.model_ids,
            request.initial_capital,
            max_sequence_length,
        )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        excel_path = temp_file.name
        temp_file.close()

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_data = []
            bh_data = results["buy_and_hold"]
            summary_data.append({
                "Strategy": "Buy & Hold",
                "Model ID": "N/A",
                "Final Value": bh_data["final_value"],
                "Total Return": bh_data["total_return"],
                "Total Trades": bh_data["total_trades"],
                "Sharpe Ratio": bh_data.get("sharpe_ratio", 0),
                "Max Drawdown": bh_data.get("max_drawdown", 0),
                "Win Rate": bh_data.get("win_rate", 0),
                "Volatility": bh_data.get("volatility", 0),
                "Calmar Ratio": bh_data.get("calmar_ratio", 0),
            })

            for model_id in request.model_ids:
                ml_data = results["ml_strategies"][model_id]
                model_info = models[model_id]
                summary_data.append({
                    "Strategy": model_info["model_name"],
                    "Model ID": model_id,
                    "Final Value": ml_data["final_value"],
                    "Total Return": ml_data["total_return"],
                    "Total Trades": ml_data["total_trades"],
                    "Sharpe Ratio": ml_data.get("sharpe_ratio", 0),
                    "Max Drawdown": ml_data.get("max_drawdown", 0),
                    "Win Rate": ml_data.get("win_rate", 0),
                    "Volatility": ml_data.get("volatility", 0),
                    "Calmar Ratio": ml_data.get("calmar_ratio", 0),
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            for model_id in request.model_ids:
                ml_data = results["ml_strategies"][model_id]
                model_info = models[model_id]
                trades_data = []
                portfolio_values = ml_data.get("portfolio_values", [])
                dates = ml_data.get("dates", [])
                predictions = ml_data.get("predictions", [])

                trades_by_date = {}
                for trade in ml_data.get("trades", []):
                    trade_date = convert_to_timezone_naive(trade["date"])
                    if hasattr(trade_date, "date"):
                        trade_date = trade_date.date()
                    trades_by_date[trade_date] = trade

                buy_price_stack = []
                for i, date in enumerate(dates):
                    portfolio_value = portfolio_values[i]
                    prediction = predictions[i]

                    current_date = convert_to_timezone_naive(date)
                    if hasattr(current_date, "date"):
                        current_date = current_date.date()

                    trade_on_date = trades_by_date.get(current_date)

                    daily_return = 0
                    if i > 0 and portfolio_values[i - 1] > 0:
                        daily_return = (portfolio_value - portfolio_values[i - 1]) / portfolio_values[i - 1]

                    gain_loss_amount = 0
                    gain_loss_pct = 0

                    if trade_on_date:
                        if trade_on_date["action"] in ["BUY", "INITIAL_BUY"]:
                            buy_price_stack.append({
                                "price": trade_on_date["price"],
                                "shares": trade_on_date["shares"],
                                "date": date,
                            })
                        elif trade_on_date["action"] == "SELL" and buy_price_stack:
                            last_buy = buy_price_stack.pop()
                            buy_price = last_buy["price"]
                            sell_price = trade_on_date["price"]
                            shares = trade_on_date["shares"]
                            gain_loss_amount = (sell_price - buy_price) * shares
                            gain_loss_pct = ((sell_price - buy_price) / buy_price) * 100

                    row = {
                        "Date": convert_to_timezone_naive(date),
                        "Ticker Open Price": test_data_with_indicators.loc[date, "Open"],
                        "Ticker Close Price": test_data_with_indicators.loc[date, "Close"],
                        "Portfolio Value": portfolio_value,
                        "Daily Return Pct": daily_return * 100,
                        "Prediction": "UP" if prediction == 1 else "DOWN" if prediction == 0 else "N/A",
                        "Trade Action": trade_on_date["action"] if trade_on_date else "HOLD",
                        "Trade Shares": trade_on_date["shares"] if trade_on_date else 0,
                        "Trade Value": trade_on_date["value"] if trade_on_date else 0,
                        "Trade Fee": trade_on_date["fee"] if trade_on_date else 0,
                        "Gain/Loss Amount": gain_loss_amount,
                        "Gain/Loss Pct": gain_loss_pct,
                    }

                    trades_data.append(row)

                sheet_name = model_info["model_name"][:31]
                trades_df = pd.DataFrame(trades_data)
                if "Date" in trades_df.columns:
                    trades_df["Date"] = trades_df["Date"].apply(convert_to_timezone_naive)
                trades_df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 1))
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except Exception:
                            pass
                    adjusted_width = min(max_length + 2, 20)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        filename = f"backtest_results_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return FileResponse(
            path=excel_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())


