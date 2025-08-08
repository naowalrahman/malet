import os
import uuid
import pickle
import traceback
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import TrainingRequest
from api.state import (
    training_jobs,
    models,
    data_fetcher,
    tech_indicators,
    SAVED_MODELS_DIR,
)
from models.TradingModelTrainer import TradingModelTrainer


router = APIRouter()


@router.get("/available-models")
async def get_available_models():
    return {
        "models": [
            {"model_type": "lstm", "model_name": "LSTM"},
            {"model_type": "cnn_lstm", "model_name": "CNN-LSTM"},
            {"model_type": "transformer", "model_name": "Transformer"},
            {"model_type": "gru", "model_name": "GRU"},
        ]
    }


async def train_model_background(job_id: str, request: TrainingRequest):
    try:
        def fetch_data():
            training_jobs[job_id]["status"] = "fetching_data"
            training_jobs[job_id]["progress"] = 10
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
            data_with_indicators = tech_indicators.calculate_all_indicators(data)

            training_jobs[job_id]["status"] = "training"
            training_jobs[job_id]["progress"] = 30

            trainer = TradingModelTrainer(
                model_type=request.model_type,
                sequence_length=request.sequence_length,
            )

            def progress_callback(progress_info):
                training_jobs[job_id].update({
                    "status": "training",
                    "progress": progress_info["progress"],
                    "current_epoch": progress_info["epoch"],
                    "total_epochs": progress_info["total_epochs"],
                    "train_loss": progress_info["train_loss"],
                    "val_loss": progress_info["val_loss"],
                    "val_accuracy": progress_info["val_accuracy"],
                })

            training_result = trainer.train(
                data_with_indicators,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                progress_callback=progress_callback,
            )
            return training_result, trainer

        training_result, trainer = await run_in_threadpool(do_training)

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = training_result

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
                "end_date": request.end_date,
            },
        }

        if SAVED_MODELS_DIR:
            with open(f"{SAVED_MODELS_DIR}/{job_id}.pkl", "wb") as file:
                pickle.dump(models[job_id], file)
    except Exception:
        print(traceback.format_exc())
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = traceback.format_exc()


@router.post("/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {
        "id": job_id,
        "status": "started",
        "progress": 0,
        "symbol": request.symbol,
        "model_type": request.model_type,
        "created_at": datetime.now().isoformat(),
    }
    background_tasks.add_task(train_model_background, job_id, request)
    return {"job_id": job_id, "status": "started"}


@router.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_jobs[job_id]


@router.get("/trained-models")
async def list_trained_models():
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
            "training_history": training_result.get("training_history", {}),
        })
    return {"models": model_list}


@router.delete("/trained-models/{model_id}")
async def delete_model(model_id: str):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    del models[model_id]
    if SAVED_MODELS_DIR:
        try:
            os.remove(f"{SAVED_MODELS_DIR}/{model_id}.pkl")
        except FileNotFoundError:
            pass
    if model_id in training_jobs:
        del training_jobs[model_id]
    return {"message": "Model deleted successfully"}


