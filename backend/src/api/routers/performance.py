import traceback
from fastapi import APIRouter, HTTPException

from api.state import models


router = APIRouter()


@router.get("/performance-metrics/{model_id}")
async def get_model_performance(model_id: str):
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
            "training_history": training_result["training_history"],
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


