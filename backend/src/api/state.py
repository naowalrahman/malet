import os
import sys
import logging
import pickle
from datetime import datetime
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from google import genai


# Ensure project src is importable for sibling modules like data_fetcher, indicators, etc.
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_fetcher import DataFetcher  # noqa: E402
from indicators import TechnicalIndicators  # noqa: E402


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google GenAI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "no key")
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# Global state shared across routers
training_jobs: Dict[str, Dict] = {}
models: Dict[str, Dict] = {}
data_cache: Dict[str, pd.DataFrame] = {}
market_analysis_cache: Dict[str, Dict] = {}


# Initialize components
data_fetcher = DataFetcher()
tech_indicators = TechnicalIndicators()


# Load saved models
SAVED_MODELS_DIR = os.getenv("SAVED_MODELS_DIR")
if SAVED_MODELS_DIR:
    for file in os.listdir(SAVED_MODELS_DIR):
        with open(f"{SAVED_MODELS_DIR}/{file}", "rb") as f:
            job_id = file.split(".")[0]
            model = pickle.load(f)
            models[job_id] = model


