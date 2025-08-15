import traceback
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest
from api.state import data_fetcher, tech_indicators, models


router = APIRouter()


@router.post("/predict")
async def make_prediction(request: PredictionRequest):
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")

        model_info = models[request.model_id]
        trainer = model_info["trainer"]

        data = data_fetcher.fetch_historical_data(
            request.symbol,
            (datetime.strptime(request.date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"),
            request.date,
        )

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        data_with_indicators = tech_indicators.calculate_all_indicators(data)
        predictions, confidences = trainer.predict(data_with_indicators)

        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Unable to make predictions with current data")

        return {
            "symbol": request.symbol,
            "model_id": request.model_id,
            "prediction": "UP" if predictions[-1] == 1 else "DOWN",
            "confidence": float(confidences[-1]),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())


