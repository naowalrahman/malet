import axios from "axios";

const API_BASE_URL = "http://localhost:8001";

export interface StockData {
  symbol: string;
  period: string;
  data_points: number;
  start_date: string;
  end_date: string;
  data: any[];
}

export interface StockInfo {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  beta: number;
  pe_ratio: number;
  dividend_yield: number;
  fifty_two_week_high: number;
  fifty_two_week_low: number;
}

export interface TrainingRequest {
  symbol: string;
  model_type: string;
  sequence_length: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  prediction_horizon: number;
  threshold: number;
  start_date?: string;
  end_date?: string;
}

export interface TrainingJob {
  id: string;
  status: string;
  progress: number;
  symbol: string;
  model_type: string;
  created_at: string;
  results?: any;
  result?: any;
  error?: string;
}

export interface ModelDetails {
  model_id: string;
  symbol: string;
  model_type: string;
  created_at: string;
  accuracy: number;
}

export interface TrainedModelDetails {
  model_id: string;
  symbol: string;
  model_type: string;
  created_at: string;
  accuracy: number;
  training_params: {
    sequence_length: number;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    prediction_horizon: number;
    threshold: number;
    start_date?: string;
    end_date?: string;
  };
  training_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  training_history: {
    train_losses?: number[];
    val_losses?: number[];
  };
}

export interface Prediction {
  symbol: string;
  model_id: string;
  prediction: number;
  signal: string;
  confidence: number;
  timestamp: string;
  current_price: number;
}

export interface BacktestRequest {
  symbol: string;
  model_ids: string[];
  initial_capital: number;
  start_date?: string;
  end_date?: string;
}

export interface BacktestResults {
  symbol: string;
  model_ids: string[];
  initial_capital: number;
  results: {
    buy_and_hold: any;
    ml_strategies: { [key: string]: any };
    comparison_metrics: { [key: string]: any };
    model_ids: string[];
  };
  plots: any;
}

export interface MarketAnalysis {
  date: Date;
  symbol: string;
  current_price: number;
  price_change: number;
  price_change_pct: number;
  volume: number;
  rsi: number;
  macd: number;
  bollinger_position: number;
  combined_signal: number;
  volatility: number;
  support_levels: number[];
  resistance_levels: number[];
}

class ApiService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    axios.defaults.timeout = 30000; // 30 seconds timeout
  }

  // Data endpoints
  async getPopularSymbols(): Promise<{ symbols: string[] }> {
    const response = await axios.get(`${this.baseURL}/popular-symbols`);
    return response.data;
  }

  async validateSymbol(symbol: string): Promise<{ symbol: string; valid: boolean }> {
    const response = await axios.post(`${this.baseURL}/validate-symbol`, { symbol });
    return response.data;
  }

  async getStockData(symbol: string, period: string = "1mo"): Promise<StockData> {
    const response = await axios.post(`${this.baseURL}/stock-data`, { symbol, period });
    return response.data;
  }

  async getHistoricalData(symbol: string, startDate: string, endDate: string): Promise<StockData> {
    const response = await axios.post(`${this.baseURL}/historical-data`, {
      symbol,
      start_date: startDate,
      end_date: endDate,
    });
    return response.data;
  }

  async getStockInfo(symbol: string): Promise<StockInfo> {
    const response = await axios.get(`${this.baseURL}/stock-info/${symbol}`);
    return response.data;
  }

  // Machine Learning endpoints
  async trainModel(request: TrainingRequest): Promise<{ job_id: string; status: string }> {
    const response = await axios.post(`${this.baseURL}/train-model`, request);
    return response.data;
  }

  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    const response = await axios.get(`${this.baseURL}/training-status/${jobId}`);
    return response.data;
  }

  async getTrainedModels(): Promise<{ models: TrainedModelDetails[] }> {
    const response = await axios.get(`${this.baseURL}/trained-models`);
    return response.data;
  }

  async getAvailableModels(): Promise<{ models: ModelDetails[] }> {
    const response = await axios.get(`${this.baseURL}/available-models`);
    return response.data;
  }

  async deleteModel(modelId: string): Promise<{ message: string }> {
    const response = await axios.delete(`${this.baseURL}/trained-models/${modelId}`);
    return response.data;
  }

  async downloadModel(modelId: string): Promise<Blob> {
    const response = await axios.get(`${this.baseURL}/trained-models/${modelId}/download`, {
      responseType: "blob",
    });
    return response.data;
  }

  async getModelPerformance(modelId: string): Promise<any> {
    const response = await axios.get(`${this.baseURL}/performance-metrics/${modelId}`);
    return response.data;
  }

  // Prediction endpoints
  async makePrediction(symbol: string, modelId: string): Promise<Prediction> {
    const response = await axios.post(`${this.baseURL}/predict`, {
      symbol,
      model_id: modelId,
    });
    return response.data;
  }

  // Backtesting endpoints
  async runBacktest(request: BacktestRequest): Promise<BacktestResults> {
    const response = await axios.post(`${this.baseURL}/backtest`, request);
    return response.data;
  }

  // Analysis endpoints
  async getMarketAnalysis(symbols: string[]): Promise<MarketAnalysis[]> {
    const response = await axios.get(`${this.baseURL}/market-analysis`, {
      params: { symbols: symbols.join(",") },
    });
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await axios.get(`${this.baseURL}/health`);
    return response.data;
  }
}

export const apiService = new ApiService();
export default ApiService;
