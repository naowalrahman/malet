from datetime import datetime, timedelta
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from models.TradingModelTrainer import TradingModelTrainer

class TradingStrategy:
    """
    Base class for trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
class BuyAndHoldStrategy(TradingStrategy):
    """
    Simple buy and hold strategy for comparison
    """
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest buy and hold strategy
        """
        if data.empty:
            return self._empty_results()
        
        # Buy at first price, hold until end
        first_price = data['Close'].iloc[0]
        last_price = data['Close'].iloc[-1]
        
        shares = self.initial_capital / first_price
        transaction_cost = shares * first_price * self.transaction_cost
        
        final_value = shares * last_price
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Create portfolio value series
        portfolio_values = shares * data['Close']
        
        return {
            'strategy': 'Buy and Hold',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': 1,
            'shares': shares,
            'buy_price': first_price,
            'sell_price': last_price,
            'transaction_costs': transaction_cost,
            'portfolio_values': portfolio_values.tolist(),
            'dates': data.index.tolist()
        }
    
    def _empty_results(self):
        return {
            'strategy': 'Buy and Hold',
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0,
            'total_trades': 0,
            'shares': 0,
            'buy_price': 0,
            'sell_price': 0,
            'transaction_costs': 0,
            'portfolio_values': [],
            'dates': []
        }

class MLTradingStrategy(TradingStrategy):
    """
    Machine Learning based trading strategy
    """
    
    def __init__(self, model: TradingModelTrainer, initial_capital: float = 10000, 
                 transaction_cost: float = 0):
        super().__init__(initial_capital, transaction_cost)
        self.model = model
        
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest ML strategy
        """
        if data.empty:
            return self._empty_results()
        
        try:
            # Get predictions from the model
            predictions = self.model.predict(data)
            
            if len(predictions) == 0:
                return self._empty_results()
            
            # Initialize tracking variables
            current_capital = self.initial_capital
            current_shares = 0
            portfolio_values = []
            trades = []
            transaction_costs = 0
            
            # Align predictions with data (predictions start from sequence_length)
            start_idx = len(data) - len(predictions)
            aligned_data = data.iloc[start_idx:].copy()
            original_dates = aligned_data.index.tolist()
            aligned_data = aligned_data.reset_index(drop=True)

            # Start by buying shares with initial capital (aggressive strategy)
            if len(predictions) > 0 and len(aligned_data) > 0:
                first_price = aligned_data['Close'].iloc[0]
                initial_shares = current_capital / first_price
                if initial_shares > 0:
                    cost = initial_shares * first_price
                    transaction_fee = cost * self.transaction_cost
                    
                    current_shares = initial_shares
                    current_capital -= (cost + transaction_fee)
                    transaction_costs += transaction_fee
                    
                    trades.append({
                        'date': original_dates[0],
                        'action': 'INITIAL_BUY',
                        'shares': initial_shares,
                        'price': first_price,
                        'value': cost,
                        'fee': transaction_fee
                    })
            
            for i, (idx, row) in enumerate(aligned_data.iterrows()):
                if i >= len(predictions):
                    break
                    
                current_price = row['Close']
                prediction = predictions[i]
                
                # Trading logic based on binary predictions
                # 0: Down (Sell), 1: Up (Buy)
                if prediction == 1 and current_shares == 0:  # Buy signal (price going up)
                    shares_to_buy = current_capital / current_price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        transaction_fee = cost * self.transaction_cost
                        
                        if current_capital >= cost + transaction_fee:
                            current_shares = shares_to_buy
                            current_capital -= cost + transaction_fee
                            transaction_costs += transaction_fee
                            
                            trades.append({
                                'date': original_dates[i],
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'value': cost,
                                'fee': transaction_fee
                            })
                
                elif prediction == 0 and current_shares > 0:  # Sell signal (price going down)
                    revenue = current_shares * current_price
                    transaction_fee = revenue * self.transaction_cost
                    
                    current_capital += revenue - transaction_fee
                    transaction_costs += transaction_fee
                    
                    trades.append({
                        'date': original_dates[i],
                        'action': 'SELL',
                        'shares': current_shares,
                        'price': current_price,
                        'value': revenue,
                        'fee': transaction_fee
                    })
                    
                    current_shares = 0
                
                # Calculate current portfolio value
                portfolio_value = current_capital + (current_shares * current_price)
                portfolio_values.append(portfolio_value)
            
            # Final liquidation if holding shares
            if current_shares > 0:
                final_price = aligned_data['Close'].iloc[-1]
                revenue = current_shares * final_price
                transaction_fee = revenue * self.transaction_cost
                current_capital += revenue - transaction_fee
                transaction_costs += transaction_fee
                current_shares = 0
            
            final_value = current_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital

            return {
                'strategy': 'ML Trading',
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': len(trades),
                'transaction_costs': transaction_costs,
                'trades': trades,
                'portfolio_values': portfolio_values,
                'dates': original_dates,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            }
            
        except Exception as e:
            print(f"Error in ML backtesting: {traceback.format_exc()}")
            return self._empty_results()
    
    def _empty_results(self):
        return {
            'strategy': 'ML Trading',
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0,
            'total_trades': 0,
            'transaction_costs': 0,
            'trades': [],
            'portfolio_values': [],
            'dates': [],
            'predictions': []
        }

class BacktestEngine:
    """
    Comprehensive backtesting engine
    """
    
    def __init__(self, models: Dict[str, Dict]):
        self.results = {}
        self.models = models
        
    def run_comparison(self, padding_data: pd.DataFrame, test_data: pd.DataFrame, model_ids: List[str], initial_capital: float = 10000, max_sequence_length: int = 0) -> Dict:
        """
        Run comparison between Buy & Hold and multiple ML strategies
        """
        ml_results = {}

        # run backtest for each model
        for model_id in model_ids:
            ml_strategy = MLTradingStrategy(self.models[model_id]["trainer"], initial_capital)
            sequence_length = self.models[model_id]["training_params"]["sequence_length"]
            # get the last sequence_length days of the padding data and combine with the test data
            ml_result = ml_strategy.backtest(pd.concat([padding_data.iloc[-sequence_length:], test_data]))
            ml_metrics = self.calculate_metrics(ml_result)

            ml_results[model_id] = {**ml_result, **ml_metrics}

        # Run Buy and Hold strategy once on aligned data
        bh_strategy = BuyAndHoldStrategy(initial_capital)
        bh_results = bh_strategy.backtest(test_data)
        bh_metrics = self.calculate_metrics(bh_results)
        
        # Create comparison metrics for each model vs buy and hold
        comparison_metrics = {}
        for model_id in model_ids:
            comparison_metrics[model_id] = self.compare_strategies(bh_results, ml_results[model_id])
        
        multi_comparison = {
            'buy_and_hold': {**bh_results, **bh_metrics},
            'ml_strategies': ml_results,
            'comparison_metrics': comparison_metrics,
            'model_ids': model_ids
        }
        
        self.results = multi_comparison
        return multi_comparison
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not results.get('portfolio_values') or len(results['portfolio_values']) == 0:
            return self._empty_metrics()
        
        portfolio_values = np.array(results['portfolio_values'])
        
        # Returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized for daily data
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualized for daily data
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Win rate (for ML strategy)
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        
        if results.get('trades'):
            profitable_trades = 0
            total_profit = 0
            total_loss = 0
            profit_count = 0
            loss_count = 0
            
            for i in range(1, len(results['trades'])):
                if results['trades'][i]['action'] == 'SELL' and i > 0:
                    buy_price = results['trades'][i-1]['price']
                    sell_price = results['trades'][i]['price']
                    profit = (sell_price - buy_price) / buy_price
                    
                    if profit > 0:
                        profitable_trades += 1
                        total_profit += profit
                        profit_count += 1
                    else:
                        total_loss += abs(profit)
                        loss_count += 1
            
            total_trades = len([t for t in results['trades'] if t['action'] == 'SELL'])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            avg_win = total_profit / profit_count if profit_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (results['total_return'] / max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        sortino_ratio *= np.sqrt(252 * 24 * 60)  # Annualized
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': (avg_win / avg_loss) if avg_loss > 0 else 0
        }
    
    def _empty_metrics(self):
        return {
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }
    
    def compare_strategies(self, bh_results: Dict, ml_results: Dict) -> Dict:
        """
        Compare performance between strategies
        """
        return {
            'return_difference': ml_results['total_return'] - bh_results['total_return'],
            'value_difference': ml_results['final_value'] - bh_results['final_value'],
            'outperformed': bool(ml_results['total_return'] > bh_results['total_return']),
            'ml_advantage': (ml_results['total_return'] - bh_results['total_return']) / abs(bh_results['total_return']) if bh_results['total_return'] != 0 else 0
        }
    
    def generate_plots(self) -> Dict:
        """
        Generate plots for multiple model comparison
        """
        if not self.results:
            raise ValueError("No backtesting results found; run_comparison must be called first")

        plots = {}
        
        # Portfolio value comparison (all models together)
        plots['portfolio_comparison'] = self.plot_portfolio_comparison()
        
        # Individual model plots
        model_ids = self.results['model_ids']
        
        # Returns distribution for each model
        plots['returns_distribution'] = {}
        for model_id in model_ids:
            plots['returns_distribution'][model_id] = self.plot_returns_distribution(model_id)
        
        # Drawdown analysis for each model
        plots['drawdown_analysis'] = {}
        for model_id in model_ids:
            plots['drawdown_analysis'][model_id] = self.plot_drawdown_analysis(model_id)
        
        # Trade analysis for each model
        plots['trade_analysis'] = {}
        for model_id in model_ids:
            if self.results['ml_strategies'][model_id].get('trades'):
                plots['trade_analysis'][model_id] = self.plot_trade_analysis(model_id)
        
        return plots
    
    def plot_portfolio_comparison(self):
        """
        Create portfolio value comparison plot for multiple models
        """
        try:
            bh_values = self.results['buy_and_hold']['portfolio_values']
            bh_dates = self.results['buy_and_hold']['dates']
            
            if len(bh_values) == 0:
                return None
            
            fig = go.Figure()
            
            # Add Buy & Hold trace
            fig.add_trace(go.Scatter(
                x=bh_dates,
                y=bh_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ))
            
            # Add traces for each ML model
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, model_id in enumerate(self.results['model_ids']):
                strategy = self.results['ml_strategies'][model_id]
                ml_values = strategy['portfolio_values']
                ml_dates = strategy['dates']
                model_name = self.models[model_id]['model_name']

                if len(ml_values) > 0:
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=ml_dates,
                        y=ml_values,
                        mode='lines',
                        name=model_name,
                        line=dict(color=color, width=2)
                    ))
            
            fig.update_layout(
                title='Portfolio Value Comparison',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(tickformat='%b %d, %Y')
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating multi-model portfolio comparison plot: {traceback.format_exc()}")
            return None
    
    def plot_returns_distribution(self, model_id: str):
        """
        Create returns distribution plot for a specific model
        """
        try:
            bh_values = np.array(self.results['buy_and_hold']['portfolio_values'])
            ml_values = np.array(self.results['ml_strategies'][model_id]['portfolio_values'])
            model_name = self.models[model_id]['model_name']

            if len(bh_values) <= 1 or len(ml_values) <= 1:
                return None
            
            bh_returns = np.diff(bh_values) / bh_values[:-1]
            ml_returns = np.diff(ml_values) / ml_values[:-1]
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=bh_returns,
                name='Buy & Hold',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=ml_returns,
                name=model_name,
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.update_layout(
                title=f'Returns Distribution - {model_name}',
                xaxis_title='Returns',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating returns distribution plot for model {model_id}: {traceback.format_exc()}")
            return None
    
    def plot_drawdown_analysis(self, model_id: str):
        """
        Create drawdown analysis plot for a specific model
        """
        try:
            bh_values = np.array(self.results['buy_and_hold']['portfolio_values'])
            strategy = self.results['ml_strategies'][model_id]
            ml_values = np.array(strategy['portfolio_values'])
            bh_dates = self.results['buy_and_hold']['dates']
            ml_dates = strategy['dates']
            model_name = self.models[model_id]['model_name']
            
            if len(bh_values) == 0 or len(ml_values) == 0:
                return None
            
            # Calculate drawdowns
            bh_peak = np.maximum.accumulate(bh_values)
            bh_drawdown = (bh_values - bh_peak) / bh_peak
            
            ml_peak = np.maximum.accumulate(ml_values)
            ml_drawdown = (ml_values - ml_peak) / ml_peak
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=bh_dates,
                y=bh_drawdown,
                mode='lines',
                name='Buy & Hold',
                fill='tonexty',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=ml_dates,
                y=ml_drawdown,
                mode='lines',
                name=f'{model_name}',
                fill='tonexty',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'Drawdown Analysis - {model_name}',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                hovermode='x unified',
                xaxis=dict(tickformat='%b %d, %Y')
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating drawdown analysis plot for model {model_id}: {traceback.format_exc()}")
            return None
    
    def plot_trade_analysis(self, model_id: str):
        """
        Create trade analysis plot for a specific model
        """
        try:
            trades = self.results['ml_strategies'][model_id]['trades']
            model_name = self.models[model_id]['model_name']
            
            if not trades:
                return None
            
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            fig = go.Figure()
            
            if buy_trades:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in buy_trades],
                    y=[t['price'] for t in buy_trades],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            if sell_trades:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in sell_trades],
                    y=[t['price'] for t in sell_trades],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title=f'Trading Signals - {model_name}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='closest',
                xaxis=dict(tickformat='%b %d, %Y')
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating trade analysis plot for model {model_id}: {traceback.format_exc()}")
            return None
