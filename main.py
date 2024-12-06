# main.py
import pandas as pd
from data_fetcher import (
    fetch_sp500_constituents,
    fetch_equity_data_for_symbols,
    fetch_options_data
)

from backtesting_module import    BacktestEngine
from data_utils import  validate_date_ranges
from visualization import plot_portfolio_vs_market, save_results
from portfolio_state import PortfolioState

from data_preparation import  ETFDataPipeline, fetch_treasury_rates

from models_utils import prepare_training_data
from graph_layers import GraphConvLayer

from model_builder import (
    
    CombinedModel,
    create_lstm_sequences,
    train_option_pricing_model,
    
    
    build_graph_data,
    train_combined_model
    
)
from enhanced_models import EnhancedCombinedModel, train_enhanced_model
from data_utils import fetch_and_validate_data, validate_input_data
from options_model import OptionsPricingMLP
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
#from spektral.layers import GCNConv
from sklearn.metrics import  mean_absolute_error, r2_score, root_mean_squared_error
from config import (
    INITIAL_CAPITAL, REBALANCE_FREQUENCY,
    TRANSACTION_COST, ALLOCATION_RATIOS, EQUITY_START_DATE, EQUITY_END_DATE,
     SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, OPTION_FEATURES, OPTION_TARGET, FEATURES, TARGET, TREASURY_FALLBACK_RATES, RISK_FREE_RATE,
     BACKTEST_START_DATE, MINIMUM_HISTORY_YEARS
)


# Utilities
from models_utils import (
    create_datasets,
    build_graph_data,
    create_lstm_sequences
)
from visualization import (
    plot_portfolio_performance,
    plot_transaction_analysis
)
from yfinance_cache import yf_cache
import os
import sys
import pickle
import yfinance as yf
from datetime import datetime
import traceback
import logging
from collections import defaultdict
import io
from typing import List, Dict, Optional, Tuple, Union





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging to both file and console"""
    # Create formatters and handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler
    file_name = f'etf_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return file_name

def analyze_training_history(history):
    """
    Analyze and plot model training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for phase_history in history:
        if isinstance(phase_history, tf.keras.callbacks.History):
            # Direct History object
            plt.plot(phase_history.history['loss'], 
                    label=f'Train Loss')
            if 'val_loss' in phase_history.history:
                plt.plot(phase_history.history['val_loss'], '--', 
                        label=f'Val Loss')
        elif isinstance(phase_history, dict):
            # Dictionary containing history
            for metric in ['loss', 'val_loss']:
                if metric in phase_history:
                    plt.plot(phase_history[metric], 
                            label=f'{metric}')
    
    plt.title('Options Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    metrics = calculate_training_metrics(history)
    plt.bar(range(len(metrics)), list(metrics.values()), align='center')
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
    plt.title('Training Metrics')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('options_model_training.png')
    plt.close()
    
    # Log metrics
    logger.info("\nTraining Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

def calculate_training_metrics(history):
    """
    Calculate training metrics from history
    """
    metrics = {}
    
    try:
        # Find best loss
        min_loss = float('inf')
        min_val_loss = float('inf')
        
        for h in history:
            if isinstance(h, tf.keras.callbacks.History):
                # Direct History object
                min_loss = min(min_loss, min(h.history['loss']))
                if 'val_loss' in h.history:
                    min_val_loss = min(min_val_loss, min(h.history['val_loss']))
            elif isinstance(h, dict):
                # Dictionary containing history
                if 'loss' in h:
                    min_loss = min(min_loss, min(h['loss']))
                if 'val_loss' in h:
                    min_val_loss = min(min_val_loss, min(h['val_loss']))
        
        metrics['Best Train Loss'] = min_loss
        if min_val_loss < float('inf'):
            metrics['Best Val Loss'] = min_val_loss
        
        # Calculate improvements
        for i, h in enumerate(history):
            if isinstance(h, tf.keras.callbacks.History):
                losses = h.history['loss']
            elif isinstance(h, dict):
                losses = h.get('loss', [])
            
            if losses:
                start_loss = losses[0]
                end_loss = losses[-1]
                improvement = (start_loss - end_loss) / start_loss * 100
                metrics[f'Improvement Phase {i+1}'] = improvement
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        metrics['Error'] = -1
        
    return metrics

def save_training_results(history, model):
    """
    Save training results and model info
    """
    try:
        # Save history
        with open('options_model_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        # Save model summary
        with open('options_model_summary.txt', 'w') as f:
            # Header
            f.write("Options Model Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Training info
            f.write("Training History:\n")
            for i, h in enumerate(history):
                f.write(f"\nPhase {i+1}:\n")
                if isinstance(h, tf.keras.callbacks.History):
                    f.write(f"Starting loss: {h.history['loss'][0]:.4f}\n")
                    f.write(f"Final loss: {h.history['loss'][-1]:.4f}\n")
                    if 'val_loss' in h.history:
                        f.write(f"Final val loss: {h.history['val_loss'][-1]:.4f}\n")
                elif isinstance(h, dict):
                    if 'loss' in h:
                        f.write(f"Starting loss: {h['loss'][0]:.4f}\n")
                        f.write(f"Final loss: {h['loss'][-1]:.4f}\n")
            
            # Model info
            f.write("\nModel Configuration:\n")
            f.write(f"Feature columns: {model.get_feature_columns()}\n")
            f.write(f"Required features: {model.get_required_features()}\n")
        
        logger.info("Training results saved to:")
        logger.info("- options_model_history.pkl")
        logger.info("- options_model_summary.txt")
        logger.info("- options_model_training.png")
        
    except Exception as e:
        logger.error(f"Error saving training results: {str(e)}")




def main():
    """Enhanced main execution for ETF-focused strategy"""
    try:
        logger.info("Starting ETF strategy backtesting")
        
        # Initialize components
        data_pipeline = ETFDataPipeline()
        backtest_engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            data_pipeline=data_pipeline,
            strategy_type='etf'
        )
        
        # 1. Fetch and validate S&P 500 constituents
        symbols, sector_map = fetch_sp500_constituents()
        if not symbols:
            logger.error("Failed to fetch S&P 500 constituents")
            return None
        logger.info(f"Fetched {len(symbols)} symbols")
        
        # 2. Fetch equity data
        logger.info("\nFetching equity data...")
        equity_data = fetch_equity_data_for_symbols(
            symbols=symbols,
            start_date=EQUITY_START_DATE,
            end_date=EQUITY_END_DATE
        )
        if not equity_data:
            logger.error("Failed to fetch equity data")
            return None
            
        # 3. Prepare and validate data based on strategy
        features, sector_constraints, dates = prepare_data(
            data_pipeline=data_pipeline,
            equity_data=equity_data,
            strategy_type='etf'
        )
        
        if not features:
            logger.error("Failed to prepare features")
            return None
            
        # 4. Validate all input data
        valid_data = fetch_and_validate_data(dates, data_pipeline)
        if not valid_data:
            logger.error("Data validation failed")
            return None
            
        # 5. Train prediction models
        logger.info("\nTraining models...")
        combined_model = train_combined_model(
            df=pd.concat([pd.DataFrame(v) for v in features.values()]), 
            features=FEATURES,
            target=TARGET,
            sequence_length=SEQUENCE_LENGTH,
            sector_map=sector_map
        )
        
        if not combined_model:
            logger.error("Failed to train combined model")
            return None
            
        # 6. Train options model if needed
        options_model = None
        if valid_data.get('options_data'):
            logger.info("Training options model...")
            options_model = OptionsPricingMLP()
            X, y = data_pipeline.get_training_data(valid_data['options_data'])
            if X is not None and y is not None:
                options_model.train(X, y)
                logger.info("Options model training completed")
        
        # 7. Run backtest
        results = backtest_engine.run_backtest(
            equity_data=valid_data['equity_data'],
            options_data=valid_data['options_data'],
            model=combined_model,
            features=FEATURES,
            rebalance_frequency=REBALANCE_FREQUENCY,
            transaction_cost=TRANSACTION_COST,
            allocation_ratios=ALLOCATION_RATIOS,
            start_date=EQUITY_START_DATE,
            end_date=EQUITY_END_DATE,
            sequence_length=SEQUENCE_LENGTH,
            option_pricing_model=options_model,
            treasury_data=valid_data['treasury_data'],
            sector_map=sector_map,
            sector_constraints=sector_constraints
        )
        
        # 8. Process and save results
        if results:
            save_and_plot_results(
                results=results,
                dates={'start': EQUITY_START_DATE, 'end': EQUITY_END_DATE},
                sector_map=sector_map
            )
            return results
        
        logger.error("Backtest failed to produce results")
        return None
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def _prepare_equity_data(equity_data: Dict[str, pd.DataFrame], 
                        data_pipeline: ETFDataPipeline) -> Tuple[Dict, Optional[Dict]]:
    """Process equity-specific data"""
    if not equity_data:
        logger.error("No equity data provided")
        return {}, None
        
    sample_symbol = list(equity_data.keys())[0]
    logger.info(f"Processing equity data with {len(equity_data)} symbols")
    logger.info(f"Sample data structure for {sample_symbol}:")
    logger.info(f"Columns: {equity_data[sample_symbol].columns.tolist()}")
    
    # Now using passed data_pipeline
    features = data_pipeline.prepare_features(equity_data)
    if not features:
        logger.error("Failed to calculate equity features")
        return {}, None
        
    return features, None

def _prepare_etf_data(equity_data: Dict[str, pd.DataFrame], 
                      data_pipeline: ETFDataPipeline) -> Tuple[Dict, Dict]:
    """Process ETF-specific data including SPY benchmark"""
    features = {}
    
    # Get SPY holdings and sector weights
    spy_holdings = data_pipeline.fetch_etf_holdings_alphavantage('SPY')
    if not spy_holdings:
        logger.warning("Failed to fetch SPY holdings")
        return {}, {}
        
    # Calculate sector weights
    spy_sector_weights = defaultdict(float)
    
    # Process sectors DataFrame
    sectors_df = spy_holdings['sectors']
    for _, row in sectors_df.iterrows():
        sector = row['sector']
        weight = float(row['weight'])  # Convert string weight to float
        spy_sector_weights[sector] = weight
    
    # Calculate ETF features
    try:
        features = data_pipeline.prepare_etf_features(equity_data)
        if not features:
            logger.error("Failed to prepare ETF features")
            return {}, dict(spy_sector_weights)
    except Exception as e:
        logger.error(f"ETF feature calculation failed: {str(e)}")
        return {}, dict(spy_sector_weights)
        
    return features, dict(spy_sector_weights)

def prepare_data(equity_data: Dict[str, pd.DataFrame], 
                data_pipeline: ETFDataPipeline,
                strategy_type: str = 'etf') -> Tuple[Dict, Dict, Dict]:
    """Main data preparation function with strategy routing"""
    dates = {
        'equity_start': pd.to_datetime(EQUITY_START_DATE),
        'equity_end': pd.to_datetime(EQUITY_END_DATE),
        'backtest_start': pd.to_datetime(BACKTEST_START_DATE)
    }
    
    if strategy_type == 'etf':
        features, sector_constraints = _prepare_etf_data(equity_data, data_pipeline)
    else:
        features, sector_constraints = _prepare_equity_data(equity_data, data_pipeline)
        
    return features, sector_constraints, dates


def save_and_plot_results(results, dates, sector_map=None, etf_holdings=None):
    """Enhanced results processing with ETF analysis"""
    try:
        logger.info("\nProcessing backtest results...")
        
        # Basic performance metrics
        portfolio_value = results['portfolio_value']
        initial_value = portfolio_value.iloc[0]
        final_value = portfolio_value.iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        logger.info("\nOverall Performance:")
        logger.info(f"Total Return: {total_return:.2f}%")
        
        # ETF-specific analysis
        if etf_holdings:
            logger.info("\nETF Analysis:")
            for symbol, holdings in etf_holdings.items():
                if symbol in results['transactions']:
                    trades = results['transactions'][results['transactions']['symbol'] == symbol]
                    logger.info(f"\n{symbol}:")
                    logger.info(f"Number of trades: {len(trades)}")
                    logger.info(f"Holdings overlap: {len(set(holdings['holdings']['symbol']) & set(trades['symbol']))}")
        
        # Sector performance
        if sector_map:
            sector_returns = defaultdict(list)
            for symbol in results['transactions']['symbol'].unique():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    trades = results['transactions'][results['transactions']['symbol'] == symbol]
                    returns = trades['value'].sum() / trades['cost'].sum() - 1
                    sector_returns[sector].append(returns)
            
            logger.info("\nSector Performance:")
            for sector, returns in sector_returns.items():
                avg_return = np.mean(returns) if returns else 0
                logger.info(f"{sector}: {avg_return:.2%}")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'portfolio_value': results['portfolio_value'],
            'benchmark': results.get('spy_benchmark')
        })
        results_df.to_csv('backtest_results.csv')
        
        # Save transaction history
        pd.DataFrame(results['transactions']).to_csv('transaction_history.csv')
        
        logger.info("\nResults saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")



def log_results_summary(results):
    """Log summary of backtest results"""
    try:
        logger.info("\n=== Backtest Results Summary ===")
        logger.info(f"Total Trading Days: {len(results['portfolio_value'])}")
        logger.info(f"Total Transactions: {len(results['transactions'])}")
        
        initial_value = results['portfolio_value'].iloc[0]
        final_value = results['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        logger.info(f"Initial Portfolio Value: ${initial_value:,.2f}")
        logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        
        for metric_name, value in results['metrics'].items():
            logger.info(f"{metric_name}: {value:.2f}")
            
    except Exception as e:
        logger.error(f"Error logging results summary: {str(e)}")

if __name__ == '__main__':
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting backtest execution. Logging to: {log_file}")
    
    # Run backtest
    results = main()
    
    # Final validation
    if results:
        logger.info("\nBacktest completed successfully")
    else:
        logger.error("Backtest failed")

