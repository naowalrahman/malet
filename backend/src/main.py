import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure local src directory is importable when running as a script
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from api.routers import (
    core_router,
    data_router,
    ml_router,
    prediction_router,
    backtesting_router,
    analysis_router,
    performance_router,
)


load_dotenv()

app = FastAPI(title="MALET API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (no prefixes to preserve existing paths)
app.include_router(core_router)
app.include_router(data_router)
app.include_router(ml_router)
app.include_router(prediction_router)
app.include_router(backtesting_router)
app.include_router(analysis_router)
app.include_router(performance_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
