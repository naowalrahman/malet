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

> [!NOTE]
> I'm current working on blog post explaining the project in detail, as well is an analysis of the models and their performance. Check it out on my [website](https://naowalrahman.rocks/#/blog), where I'll be posting it soon.

MALET (**MA**chine **LE**arning **T**rader) is a platform to fetch financial data, train various deep learning models for trading, and backtest performance against historical market data. It performs very well when trained over 5+ years of daily close data for diverse ETFs like SPY and QQQ, beating buy and hold strategies by a wide margin.

![MALET dashboard](img/dashboard.svg)

## Todos

- [ ] List buy and sell accuracy separately in addition to the total accuracy
- [ ] Add ideal buy and sell signals to the backtesting results
- [ ] Add option to specify separate tickers for buy and sell

## ✨ Key Features

* **Interactive Dashboard:** Get a high-level overview of market conditions, model performance, and quick actions. Utilizes Gemini API to create a comprehensive AI-powered analysis of market data for SPY, QQQ, and DJIA.
* **Dynamic Data Fetching:** Pulls historical stock data from yfinance for any given ticker and date range.
* **Automated Feature Engineering:** Automatically calculates over 50 technical indicators (e.g., RSI, MACD, Bollinger Bands, Ichimoku Cloud, etc.) to enrich the dataset.
* **Versatile Model Training:** Train multiple types of neural networks to predict market movements:
    * LSTM (Long Short-Term Memory)
    * CNN-LSTM (Convolutional + LSTM)
    * Transformer
    * GRU (Gated Recurrent Unit)
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
* **UI Components:** Material UI (MUI)
* **Charting:** Plotly.js and react-plotly.js
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

MALET utilizes deep learning for time-series forecasting. The goal is to predict whether the price of a stock will move up or down over a defined future period based on a set of technical indicators and previous close prices spanning a defined time period (sequence length). As such, all the models involve some sort of deep learning architecture followed by a classification layer to predict the binary outcome.

1.  **LSTM (`LSTM.py`):** A standard Long Short-Term Memory network, which is well-suited for learning from sequential data like stock prices.
2.  **CNN-LSTM (`CNN_LSTM.py`):** A hybrid model that uses 1D Convolutional Neural Networks to extract spatial features from the input sequences before feeding them into an LSTM layer to capture temporal dependencies.
3.  **Transformer (`Transformer.py`):** An attention-based model, inspired by its success in natural language processing, adapted here to identify complex patterns and relationships in financial time-series data.
4.  **GRU (`GRU.py`):** A variant of the LSTM model that uses a Gated Recurrent Unit architecture, which is generally more efficient, performant, and better at dealing with vanishing gradients.

## 🚀 Getting Started

To get the application running locally, follow these steps.

### Prerequisites

* Python 3.8+ and `pip`
* Node.js and `npm` (or `yarn`)

### 1. Backend Setup

Create a `.env` file in the backend directory and add your Gemini API key and the directory to save the trained models.

```bash
GEMINI_API_KEY=your_gemini_api_key
SAVED_MODELS_DIR=/your/path/to/save/models
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

* **/dashboard**: The main landing page showing a summary of market data/analysis, trained models, and quick navigation links.
* **/training**: Configure, train, and manage your machine learning models. Set hyperparameters and monitor training progress.
* **/backtesting**: Select multiple trained models and a date range to run a historical simulation. Compare the model's performance against a baseline buy and hold strategy and view detailed results and charts.
* **/data (Future)**: A planned section for in-depth data exploration and visualization.
* **/live (Future)**: A planned section for simulated or real live trading.

## 🔗 API Endpoints

The FastAPI backend provides several endpoints to support the application. Here are some of the key ones:

| Method   | Path                               | Description                                             |
| :------  | :--------------------------------- | :------------------------------------------------------ |
| `GET`    | `/popular-symbols`                 | Get a list of popular stock symbols.                    |
| `POST`   | `/stock-data`                      | Fetch historical stock data for a given period.         |
| `POST`   | `/train-model`                     | Start a new model training job.                         |
| `GET`    | `/training-status/{job_id}`        | Check the status and progress of a training job.        |
| `GET`    | `/trained-models`                  | List all successfully trained models.                   |
| `DELETE` | `/trained-models/{model_id}`       | Delete a specific trained model.                        |
| `POST`   | `/backtest`                        | Run a backtest on a trained model.                      |
| `GET`    | `/market-analysis/{symbol}`        | Get a comprehensive technical analysis summary.         |
| `GET`    | `/market-analysis/{symbol}/ai`     | Get a comprehensive AI-powered analysis of market data. |
