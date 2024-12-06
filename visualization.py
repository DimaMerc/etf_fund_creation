
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import RISK_FREE_RATE
import traceback
import logging

logger = logging.getLogger(__name__)




def plot_portfolio_vs_market(portfolio_value, spy_benchmark=None, start_date=None, end_date=None, save_path='portfolio_vs_market.png'):
    """Plot portfolio performance against SPY benchmark with debug logging"""
    try:
        logger.info(f"\nGenerating performance plot:")
        logger.info(f"Start date: {start_date}")
        logger.info(f"End date: {end_date}")
        logger.info(f"Portfolio value points: {len(portfolio_value)}")
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Plot portfolio data
        portfolio_normalized = portfolio_value / portfolio_value.iloc[0] * 100
        plt.plot(portfolio_normalized.index, portfolio_normalized.values, 'b-', 
                label='Portfolio', linewidth=2)
        
        # Plot benchmark if available
        if spy_benchmark is not None:
            spy_normalized = spy_benchmark / spy_benchmark.iloc[0] * 100
            plt.plot(spy_normalized.index, spy_normalized.values, 'r--', 
                    label='S&P 500', linewidth=2)
            
            # Calculate benchmark stats
            spy_return = (spy_benchmark.iloc[-1] / spy_benchmark.iloc[0] - 1) * 100
            logger.info(f"Benchmark return: {spy_return:.2f}%")
        
        # Add labels and title
        plt.title('Portfolio Performance vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Value (Normalized to 100)')
        plt.grid(True)
        plt.legend()
        
        # Add performance metrics
        portfolio_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        
        # Enhanced performance stats
        performance_text = f'Portfolio Return: {portfolio_return:.1f}%\n'
        if spy_benchmark is not None:
            performance_text += f'S&P 500 Return: {spy_return:.1f}%\n'
            alpha = portfolio_return - spy_return
            performance_text += f'Alpha: {alpha:.1f}%'
            
        plt.text(0.02, 0.98, performance_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved successfully to {save_path}")
        
    except Exception as e:
        logger.error(f"Error plotting portfolio vs market: {str(e)}")
        logger.error(traceback.format_exc())



def save_results(results, filename='backtest_results'):
    """Save backtest results with enhanced error handling"""
    try:
        if not results:
            logger.warning("No results to save")
            return
            
        excel_file = f"{filename}.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            # Save portfolio value
            if isinstance(results.get('portfolio_value'), pd.Series):
                portfolio_df = pd.DataFrame(results['portfolio_value'])
                portfolio_df.columns = ['Value']
                portfolio_df.to_excel(writer, sheet_name='Portfolio Value')
            
            # Handle transactions
            if 'transactions' in results:
                trans_data = results['transactions']
                trans_df = None
                
                # Convert list to DataFrame if necessary
                if isinstance(trans_data, list):
                    if trans_data:  # Only if list is not empty
                        trans_df = pd.DataFrame(trans_data)
                elif isinstance(trans_data, pd.DataFrame):
                    trans_df = trans_data
                
                # Process transactions DataFrame if it exists
                if trans_df is not None and not trans_df.empty:
                    # Convert any missing columns to string
                    for col in trans_df.columns:
                        if trans_df[col].dtype == 'object':
                            trans_df[col] = trans_df[col].astype(str)
                            
                    trans_df.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Save metrics if they exist
            if results.get('metrics'):
                metrics_df = pd.DataFrame([results['metrics']])
                metrics_df.to_excel(writer, sheet_name='Metrics')
            
            # Save additional analysis if available
            if results.get('spy_benchmark') is not None:
                benchmark_df = pd.DataFrame(results['spy_benchmark'])
                benchmark_df.columns = ['Value']
                benchmark_df.to_excel(writer, sheet_name='Benchmark')
        
        logger.info(f"Results saved to {excel_file}")
        
        # Save portfolio value to CSV
        if isinstance(results.get('portfolio_value'), pd.Series):
            csv_file = f"{filename}_portfolio_value.csv"
            results['portfolio_value'].to_csv(csv_file)
            logger.info(f"Portfolio values saved to {csv_file}")
            
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.error(traceback.format_exc())

def plot_portfolio_performance(portfolio_value: pd.Series, transactions: pd.DataFrame):
    """Plot and log portfolio performance metrics"""
    try:
        # Calculate metrics
        returns = portfolio_value.pct_change()
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252))
        
        # Log metrics
        logger.info("\nPortfolio Performance Metrics:")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Volatility: {volatility:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Number of Trades: {len(transactions)}")
        
    except Exception as e:
        logger.error(f"Error plotting portfolio performance: {str(e)}")



def _plot_transaction_counts(transactions: pd.DataFrame, ax):
    """Plot transaction counts by type"""
    try:
        action_counts = transactions['action'].value_counts()
        action_counts.plot(kind='bar', ax=ax)
        ax.set_title('Transaction Counts by Action Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(action_counts):
            ax.text(i, v, str(v), ha='center', va='bottom')
            
    except Exception as e:
        logger.error(f"Error plotting transaction counts: {str(e)}")
        ax.text(0.5, 0.5, 'Error plotting data', ha='center', va='center')

def _plot_transaction_values(transactions: pd.DataFrame, ax):
    """Plot transaction values over time"""
    try:
        daily_values = transactions.groupby('date')['value'].sum()
        daily_values.plot(ax=ax)
        ax.set_title('Daily Transaction Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value ($)')
        ax.grid(True)
        
        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis dates for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    except Exception as e:
        logger.error(f"Error plotting transaction values: {str(e)}")
        ax.text(0.5, 0.5, 'Error plotting data', ha='center', va='center')

def _plot_cumulative_value(transactions: pd.DataFrame, ax):
    """Plot cumulative transaction value"""
    try:
        daily_values = transactions.groupby('date')['value'].sum()
        cumulative_value = daily_values.cumsum()
        cumulative_value.plot(ax=ax)
        ax.set_title('Cumulative Transaction Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Value ($)')
        ax.grid(True)
        
        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis dates for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add final value annotation
        if len(cumulative_value) > 0:
            final_value = cumulative_value.iloc[-1]
            ax.text(0.98, 0.95, f'Final Value: ${final_value:,.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.8))
            
    except Exception as e:
        logger.error(f"Error plotting cumulative value: {str(e)}")
        ax.text(0.5, 0.5, 'Error plotting data', ha='center', va='center')

def _plot_size_distribution(transactions: pd.DataFrame, ax):
    """Plot transaction size distribution"""
    try:
        transactions['value'].hist(bins=50, ax=ax)
        ax.set_title('Transaction Size Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Transaction Value ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        # Format x-axis with dollar signs
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add summary statistics
        mean_val = transactions['value'].mean()
        median_val = transactions['value'].median()
        ax.text(0.98, 0.95, 
               f'Mean: ${mean_val:,.2f}\nMedian: ${median_val:,.2f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(facecolor='white', alpha=0.8))
               
    except Exception as e:
        logger.error(f"Error plotting size distribution: {str(e)}")
        ax.text(0.5, 0.5, 'Error plotting data', ha='center', va='center')

def plot_transaction_analysis(transactions: pd.DataFrame):
    """Plot detailed transaction analysis"""
    try:
        if transactions.empty:
            logger.warning("No transactions to analyze")
            return
            
        # First check which value column is available
        value_column = None
        if 'value' in transactions.columns:
            value_column = 'value'
        elif 'total_cost' in transactions.columns:
            value_column = 'total_cost'
        
        if value_column is None:
            logger.warning("No value column found in transactions")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Convert date column to datetime if needed
        if 'date' in transactions.columns:
            transactions['date'] = pd.to_datetime(transactions['date'])
        
        # Plot 1: Transaction counts by type
        if 'action' in transactions.columns:
            action_counts = transactions['action'].value_counts()
            action_counts.plot(kind='bar', ax=ax1)
            ax1.set_title('Transaction Counts by Action Type', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i, v in enumerate(action_counts):
                ax1.text(i, v, str(v), ha='center', va='bottom')
        
        # Plot 2: Transaction values over time
        if value_column and 'date' in transactions.columns:
            daily_values = transactions.groupby('date')[value_column].sum()
            daily_values.plot(ax=ax2)
            ax2.set_title('Daily Transaction Values', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Value ($)')
            ax2.grid(True)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Cumulative transaction value
        if value_column and 'date' in transactions.columns:
            daily_values = transactions.groupby('date')[value_column].sum()
            cumulative_value = daily_values.cumsum()
            cumulative_value.plot(ax=ax3)
            ax3.set_title('Cumulative Transaction Value', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Cumulative Value ($)')
            ax3.grid(True)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            if len(cumulative_value) > 0:
                final_value = cumulative_value.iloc[-1]
                ax3.text(0.98, 0.95, f'Final Value: ${final_value:,.2f}', 
                       transform=ax3.transAxes, ha='right', va='top',
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 4: Transaction size distribution
        if value_column:
            transactions[value_column].hist(bins=50, ax=ax4)
            ax4.set_title('Transaction Size Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Transaction Value ($)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            mean_val = transactions[value_column].mean()
            median_val = transactions[value_column].median()
            ax4.text(0.98, 0.95, 
                   f'Mean: ${mean_val:,.2f}\nMedian: ${median_val:,.2f}',
                   transform=ax4.transAxes, ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('transaction_analysis.png')
        plt.close()
        
        # Log analysis summary
        logger.info("\nTransaction Analysis Summary:")
        logger.info(f"Total transactions: {len(transactions)}")
        
        if 'date' in transactions.columns:
            logger.info(f"Date range: {transactions['date'].min()} to {transactions['date'].max()}")
            
        if value_column:
            logger.info(f"Total value: ${transactions[value_column].sum():,.2f}")
        
        # Get action breakdown
        if 'action' in transactions.columns:
            action_counts = transactions['action'].value_counts()
            logger.info("\nTransaction breakdown:")
            for action, count in action_counts.items():
                logger.info(f"{action}: {count}")
        
        logger.info("Transaction analysis plots saved")
        
    except Exception as e:
        logger.error(f"Error plotting transaction analysis: {str(e)}")
        logger.error(traceback.format_exc())

def save_and_plot_results(results, dates, sector_map=None):
    """Save and plot comprehensive results"""
    try:
        # Plot portfolio performance
        plot_portfolio_vs_market(
            portfolio_value=results['portfolio_value'],
            spy_benchmark=results.get('spy_benchmark'),
            start_date=dates['start'],
            end_date=dates['end']
        )
        
        # Plot transaction analysis if transactions exist
        if 'transactions' in results and not results['transactions'].empty:
            plot_transaction_analysis(results['transactions'])
            
        # Save results to files
        save_results(results)
        
        logger.info("Results saved and plotted successfully")
        
    except Exception as e:
        logger.error(f"Error saving and plotting results: {str(e)}")