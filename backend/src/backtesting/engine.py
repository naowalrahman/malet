import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TradingStrategy:
    """
    Base class for trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.positions = []
        self.trades = []
        self.portfolio_value = []
        
    def calculate_position_size(self, current_capital: float, price: float, 
                              risk_per_trade: float = 0.02) -> int:
        """
        Calculate position size based on risk management
        """
        max_risk_amount = current_capital * risk_per_trade
        position_value = current_capital * 0.95  # Use 95% of capital
        shares = int(position_value / price)
        return max(1, shares)

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
        
        shares = int(self.initial_capital / first_price)
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
    
    def __init__(self, model, initial_capital: float = 10000, 
                 transaction_cost: float = 0.001, confidence_threshold: float = 0.6):
        super().__init__(initial_capital, transaction_cost)
        self.model = model
        self.confidence_threshold = confidence_threshold
        
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest ML strategy
        """
        if data.empty:
            print("problem")
            return self._empty_results()
        
        try:
            # Get predictions from the model
            predictions = self.model.predict(data)
            print(predictions)
            
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
            aligned_data = aligned_data.reset_index(drop=True)
            
            for i, (idx, row) in enumerate(aligned_data.iterrows()):
                if i >= len(predictions):
                    break
                    
                current_price = row['Close']
                prediction = predictions[i]
                
                # Trading logic based on predictions
                # 0: Hold, 1: Buy, 2: Sell
                if prediction == 1 and current_shares == 0:  # Buy signal
                    shares_to_buy = self.calculate_position_size(current_capital, current_price)
                    cost = shares_to_buy * current_price
                    transaction_fee = cost * self.transaction_cost
                    
                    if current_capital >= cost + transaction_fee:
                        current_shares = shares_to_buy
                        current_capital -= (cost + transaction_fee)
                        transaction_costs += transaction_fee
                        
                        trades.append({
                            'date': row.name if hasattr(row, 'name') else i,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'value': cost,
                            'fee': transaction_fee
                        })
                
                elif prediction == 2 and current_shares > 0:  # Sell signal
                    revenue = current_shares * current_price
                    transaction_fee = revenue * self.transaction_cost
                    
                    current_capital += (revenue - transaction_fee)
                    transaction_costs += transaction_fee
                    
                    trades.append({
                        'date': row.name if hasattr(row, 'name') else i,
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
                current_capital += (revenue - transaction_fee)
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
                'dates': aligned_data.index.tolist() if hasattr(aligned_data.index, 'tolist') else list(range(len(portfolio_values))),
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            }
            
        except Exception as e:
            print(f"Error in ML backtesting: {str(e)}")
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
    
    def __init__(self):
        self.results = {}
        
    def run_comparison(self, data: pd.DataFrame, ml_model, 
                      initial_capital: float = 10000) -> Dict:
        """
        Run comparison between Buy & Hold and ML strategy
        """
        # Buy and Hold strategy
        bh_strategy = BuyAndHoldStrategy(initial_capital)
        bh_results = bh_strategy.backtest(data)
        
        # ML strategy
        ml_strategy = MLTradingStrategy(ml_model, initial_capital)
        ml_results = ml_strategy.backtest(data)
        
        # Calculate additional metrics
        bh_metrics = self.calculate_metrics(bh_results, data)
        ml_metrics = self.calculate_metrics(ml_results, data)
        
        comparison = {
            'buy_and_hold': {**bh_results, **bh_metrics},
            'ml_strategy': {**ml_results, **ml_metrics},
            'comparison_metrics': self.compare_strategies(bh_results, ml_results)
        }
        
        # print(comparison)
        
        self.results = comparison
        return comparison
    
    def calculate_metrics(self, results: Dict, data: pd.DataFrame) -> Dict:
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
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        sharpe_ratio *= np.sqrt(252 * 24 * 60)  # Annualized
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
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
        calmar_ratio = (results['total_return'] / abs(max_drawdown)) if max_drawdown != 0 else 0
        
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
        Generate visualization plots
        """
        if not self.results:
            return {}
        
        plots = {}
        
        # Portfolio value comparison
        plots['portfolio_comparison'] = self.plot_portfolio_comparison()
        
        # Returns distribution
        plots['returns_distribution'] = self.plot_returns_distribution()
        
        # Drawdown analysis
        plots['drawdown_analysis'] = self.plot_drawdown_analysis()
        
        # Trade analysis (for ML strategy)
        if self.results['ml_strategy'].get('trades'):
            plots['trade_analysis'] = self.plot_trade_analysis()
        
        return plots
    
    def plot_portfolio_comparison(self):
        """
        Create portfolio value comparison plot
        """
        try:
            bh_values = self.results['buy_and_hold']['portfolio_values']
            ml_values = self.results['ml_strategy']['portfolio_values']
            
            # Handle different lengths
            min_len = min(len(bh_values), len(ml_values))
            if min_len == 0:
                return None
            
            bh_values = bh_values[:min_len]
            ml_values = ml_values[:min_len]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=bh_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                y=ml_values,
                mode='lines',
                name='ML Strategy',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Portfolio Value Comparison',
                xaxis_title='Time',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified'
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating portfolio comparison plot: {str(e)}")
            return None
    
    def plot_returns_distribution(self):
        """
        Create returns distribution plot
        """
        try:
            bh_values = np.array(self.results['buy_and_hold']['portfolio_values'])
            ml_values = np.array(self.results['ml_strategy']['portfolio_values'])
            
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
                name='ML Strategy',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.update_layout(
                title='Returns Distribution',
                xaxis_title='Returns',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating returns distribution plot: {str(e)}")
            return None
    
    def plot_drawdown_analysis(self):
        """
        Create drawdown analysis plot
        """
        try:
            bh_values = np.array(self.results['buy_and_hold']['portfolio_values'])
            ml_values = np.array(self.results['ml_strategy']['portfolio_values'])
            
            if len(bh_values) == 0 or len(ml_values) == 0:
                return None
            
            # Calculate drawdowns
            bh_peak = np.maximum.accumulate(bh_values)
            bh_drawdown = (bh_values - bh_peak) / bh_peak
            
            ml_peak = np.maximum.accumulate(ml_values)
            ml_drawdown = (ml_values - ml_peak) / ml_peak
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=bh_drawdown,
                mode='lines',
                name='Buy & Hold Drawdown',
                fill='tonexty',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                y=ml_drawdown,
                mode='lines',
                name='ML Strategy Drawdown',
                fill='tonexty',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Time',
                yaxis_title='Drawdown (%)',
                hovermode='x unified'
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating drawdown analysis plot: {str(e)}")
            return None
    
    def plot_trade_analysis(self):
        """
        Create trade analysis plot
        """
        try:
            trades = self.results['ml_strategy']['trades']
            
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
                title='Trading Signals',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                hovermode='closest'
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"Error creating trade analysis plot: {str(e)}")
            return None
