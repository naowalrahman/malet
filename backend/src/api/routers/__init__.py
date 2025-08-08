from .core import router as core_router
from .data import router as data_router
from .ml import router as ml_router
from .prediction import router as prediction_router
from .backtesting import router as backtesting_router
from .analysis import router as analysis_router
from .performance import router as performance_router

__all__ = [
    "core_router",
    "data_router",
    "ml_router",
    "prediction_router",
    "backtesting_router",
    "analysis_router",
    "performance_router",
]


