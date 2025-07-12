<div align="center">
 
```
 __       __   ______   __        ________  ________ 
/  \     /  | /      \ /  |      /        |/        |
$$  \   /$$ |/$$$$$$  |$$ |      $$$$$$$$/ $$$$$$$$/ 
$$$  \ /$$$ |$$ |__$$ |$$ |      $$ |__       $$ |   
$$$$  /$$$$ |$$    $$ |$$ |      $$    |      $$ |   
$$ $$ $$/$$ |$$$$$$$$ |$$ |      $$$$$/       $$ |   
$$ |$$$/ $$ |$$ |  $$ |$$ |_____ $$ |_____    $$ |   
$$ | $/  $$ |$$ |  $$ |$$       |$$       |   $$ |   
$$/      $$/ $$/   $$/ $$$$$$$$/ $$$$$$$$/    $$/    
```
</div>

---                                                     

MALET (**MA**chine **LE**arning **T**rader) is a platform to fetch financial data, train various deep learning models for trading, and backtest performance against historical market data.

![MALET dashboard](img/dashboard.svg)

## ✨ Key Features

* **Interactive Dashboard:** Get a high-level overview of market conditions, model performance, and quick actions.
* **Dynamic Data Fetching:** Pulls historical stock data from yfinance for any given ticker and date range.
* **Automated Feature Engineering:** Automatically calculates over 50 technical indicators (e.g., RSI, MACD, Bollinger Bands, Ichimoku Cloud) to enrich the dataset.
* **Versatile Model Training:** Train multiple types of neural networks to predict market movements:
    * LSTM (Long Short-Term Memory)
    * CNN-LSTM (Convolutional + LSTM)
    * Transformer
* **Comprehensive Backtesting Engine:**
    * Compare your trained model's performance against a baseline "Buy and Hold" strategy.
    * Analyze key metrics like Total Return, Sharpe Ratio, Max Drawdown, and Win Rate.
    * Visualize results with interactive Plotly charts for portfolio growth, return distribution, and drawdown analysis.
* **RESTful API:** A robust FastAPI backend serves all data and machine learning functionalities.

## 🛠️ Tech Stack

This project is a monorepo composed of the frontend application and a backend service.

### Frontend

* **Framework:** React with Vite
* **Language:** TypeScript
* **UI Components:** Material UI (MUI) for a modern, responsive design.
* **Charting:** Plotly.js and react-plotly.js for interactive data visualizations.
* **API Communication:** Axios

### Backend

* **Framework:** FastAPI
* **Language:** Python
* **Machine Learning:** PyTorch
* **Data Manipulation:** Pandas & NumPy
* **Financial Data:** yfinance
* **Technical Analysis:** ta
* **API Server:** Uvicorn

## 🧠 Machine Learning Models

MALET utilizes deep learning for time-series forecasting. The goal is to predict whether the price of a stock will move up or down over a defined future period based (binary classification).

1.  **LSTM (`LSTMTradingModel.py`):** A standard Long Short-Term Memory network, which is well-suited for learning from sequential data like stock prices.
2.  **CNN-LSTM (`CNNLSTMModel.py`):** A hybrid model that uses 1D Convolutional Neural Networks to extract spatial features from the input sequences before feeding them into an LSTM layer to capture temporal dependencies.
3.  **Transformer (`TransformerTradingModel.py`):** An attention-based model, inspired by its success in natural language processing, adapted here to identify complex patterns and relationships in financial time-series data.
4.  **Ensemble Model (`EnsembleModel.py`):** This model combines the predictions from the LSTM, CNN-LSTM, and Transformer models, using a weighted average based on their validation accuracy to make a more robust final prediction.

## 🚀 Getting Started

To get the application running locally, follow these steps.

### Prerequisites

* Python 3.8+ and `pip`
* Node.js and `npm` (or `yarn`)

### 1. Backend Setup

Create a `.env` file in the backend directory and add your Gemini API key.

```bash
GEMINI_API_KEY=your_gemini_api_key
```

Navigate to the backend directory and set up a virtual environment.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required Python packages
pip install -r requirements.txt

# Run the backend server
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

The backend API will now be running on `http://localhost:8001`.

### 2. Frontend Setup

In a separate terminal, navigate to the frontend directory.

```bash
cd frontend

# Install the required Node.js packages
npm install

# Run the development server
npm run dev
```

The frontend application will now be running on `http://localhost:5173` (or another port if 5173 is in use).

## 🗺️ Application Pages

* **/dashboard**: The main landing page showing a summary of market data, trained models, and quick navigation links.
* **/training**: Configure, train, and manage your machine learning models. Set hyperparameters and monitor training progress in real-time.
* **/backtesting**: Select a trained model and a date range to run a historical simulation. Compare the model's performance against a baseline and view detailed results and charts.
* **/data (Future)**: A planned section for in-depth data exploration and visualization.
* **/live (Future)**: A planned section for simulated or real live trading.

## 🔗 API Endpoints

The FastAPI backend provides several endpoints to support the application. Here are some of the key ones:

| Method | Path                               | Description                                     |
| :----- | :--------------------------------- | :---------------------------------------------- |
| `GET`  | `/popular-symbols`                 | Get a list of popular stock symbols.            |
| `POST` | `/stock-data`                      | Fetch historical stock data for a given period. |
| `POST` | `/train-model`                     | Start a new model training job.                 |
| `GET`  | `/training-status/{job_id}`        | Check the status and progress of a training job.|
| `GET`  | `/trained-models`                  | List all successfully trained models.           |
| `DELETE`| `/trained-models/{model_id}`      | Delete a specific trained model.                |
| `POST` | `/backtest`                        | Run a backtest on a trained model.              |
| `GET`  | `/market-analysis/{symbol}`        | Get a comprehensive technical analysis summary. |
