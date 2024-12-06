#portfolio_state.py


from portfolio_base import PortfolioState
from options_trading import EnhancedOptionsStrategy

import pandas as pd
import numpy as np
import yfinance as yf
from config import TRANSACTION_COST, MIN_RECOVERY, MAX_RECOVERY_DAYS, MAX_VOLATILITY, MAX_DRAWDOWN
import traceback
from datetime import datetime, date
from typing import List, Dict, Optional
import logging
from collections import defaultdict
from column_utils import standardize_dataframe_columns, get_column_name


logger = logging.getLogger(__name__)

class PortfolioState:
    """Class to manage portfolio state with proper validation and error handling"""
    
    def __init__(self, initial_capital, combined_prices, strategy_type='etf'):
        self.strategy_type = strategy_type
        self._state = {
            'cash': initial_capital,
            'initial_capital': initial_capital,  
            'holdings': {
                'equities': {},
                'options': {}
            },
            'portfolio_value': pd.Series(initial_capital, index=[combined_prices.index[0]]),
            'equity_value': 0.0,
            'options_value': 0.0,
            'total_value': initial_capital,
            'previous_value': initial_capital,
            'combined_prices': combined_prices,
            'start_date': combined_prices.index[0],  
            'end_date': combined_prices.index[-1],
            'options_enabled': False   
        }

        self.peak_value = initial_capital 
        self.options_strategy = EnhancedOptionsStrategy(min_mispricing=0.02)  # Initialize options strategy
        logger.info(f"Initialized portfolio with ${initial_capital:,.2f}")

    
        
    

    @property
    def holdings(self):
        """Get holdings"""
        return self._state['holdings']

    @property
    def total_value(self):
        """Get total portfolio value"""
        return self._state['total_value']

    @property
    def cash(self):
        """Get cash"""
        return self._state['cash']
    
    @total_value.setter
    def total_value(self, value: float):
        """Set total portfolio value"""
        self._state['total_value'] = value

    # This helps avoid accidentally creating new attributes
    def __getattr__(self, name):
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def initialize_data(self, price_data, options_data, treasury_data, option_pricing_model):
        """Initialize additional data required for trading"""
        try:
            self._state.update({
                'price_data': price_data,
                'options_data': options_data,
                'treasury_data': treasury_data,
                'options_enabled': option_pricing_model is not None and getattr(option_pricing_model, 'is_trained', False),
                'option_pricing_model': option_pricing_model
            })

             #  ETF-specific state if needed
            if self.strategy_type == 'etf':
                self._state.update({
                    'sector_weights': defaultdict(float),
                    'tracking_error': [],
                    'sector_map': {}
                })
            
            logger.info("\nPortfolio State Initialized:")
            
            logger.info(f"Options Enabled: {self._state['options_enabled']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _initialize_portfolio_state(self, combined_prices, price_data, options_data, 
                                  treasury_data, start_date, end_date):
        """Initialize the complete portfolio state"""
        try:
            logger.info("\nInitializing portfolio state...")
            initial_capital = self._state['initial_capital']
            
            # Initialize basic state
            self._state.update({
                'cash': initial_capital,
                'holdings': {'equities': {}, 'options': {}},
                'portfolio_value': pd.Series(initial_capital, index=[start_date]),
                'equity_value': 0.0,
                'options_value': 0.0,
                'total_value': initial_capital,
                'previous_value': initial_capital,
                'combined_prices': combined_prices,
                'price_data': price_data,
                'options_data': options_data,
                'treasury_data': treasury_data,
                'start_date': start_date,
                'end_date': end_date,
                'last_rebalance': start_date - pd.Timedelta(days=1),
                'active_positions': {},  # Track active positions
                'trades_history': []     # Track all trades
            })
            
            # Log initialization
            logger.info(f"Initial Capital: ${initial_capital:,.2f}")
            logger.info(f"Start Date: {start_date}")
            logger.info(f"End Date: {end_date}")
            logger.info(f"Trading Days: {len(combined_prices)}")
            logger.info(f"Options Data Available: {not options_data.empty if isinstance(options_data, pd.DataFrame) else len(options_data) > 0}")

            if not hasattr(self, 'options_strategy'):
                self.options_strategy = EnhancedOptionsStrategy(min_mispricing=0.02) 
            
            # Set initial portfolio metrics
            self._update_portfolio_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio state: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        try:
            # Calculate equity value
            equity_value = sum(
                shares * self._state['combined_prices'].loc[
                    self._state['combined_prices'].index[-1], symbol
                ]
                for symbol, shares in self._state['holdings']['equities'].items()
            )
            
            # Calculate options value
            options_value = sum(
                position['market_value']
                for position in self._state['holdings']['options'].values()
            )
            
            # Update state
            self._state['equity_value'] = equity_value
            self._state['options_value'] = options_value
            self._state['total_value'] = self._state['cash'] + equity_value + options_value
            
            # Add to portfolio value series
            current_date = self._state['combined_prices'].index[-1]
            self._state['portfolio_value'].loc[current_date] = self._state['total_value']
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {str(e)}")

    def initialize_data(self, price_data, options_data, treasury_data, option_pricing_model):
        """Initialize additional data required for trading"""
        try:
            success = self._initialize_portfolio_state(
                combined_prices=self._state['combined_prices'],
                price_data=price_data,
                options_data=options_data,
                treasury_data=treasury_data,
                start_date=self._state['start_date'],
                end_date=self._state['end_date']
            )
            
            if not success:
                return False
                
            # Initialize options if model is available
            self._state['options_enabled'] = (
                option_pricing_model is not None and 
                hasattr(option_pricing_model, 'is_trained') and 
                option_pricing_model.is_trained
            )
            
            self._state['option_pricing_model'] = option_pricing_model
            
            logger.info(f"Data initialization completed")
            logger.info(f"Options trading enabled: {self._state['options_enabled']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def __getitem__(self, key):
        return self._state[key]
        
    def __setitem__(self, key, value):
        self._state[key] = value
        
    
    def update(self, date, combined_prices):
        """Update portfolio state with current values"""
        try:
            # Calculate equity value
            equity_value = sum(
                shares * float(combined_prices.loc[date, symbol])
                for symbol, shares in self._state['holdings']['equities'].items()
            )
            
            # Calculate options value using the strategy's method
            options_value = 0
            for option in self._state['holdings'].get('options', {}).values():
                current_price = combined_prices.loc[date, option['symbol']]
                options_value += self.options_strategy._calculate_option_value(
                    option, 
                    current_price
                )
            
            # Update state values
            self._state['equity_value'] = equity_value
            self._state['options_value'] = options_value
            self._state['total_value'] = self._state['cash'] + equity_value + options_value
            
            # Update portfolio value series
            if not isinstance(self._state['portfolio_value'], pd.Series):
                self._state['portfolio_value'] = pd.Series(dtype=float)
                
            self._state['portfolio_value'].loc[date] = self._state['total_value']
            
            # Calculate daily stats if we have enough history
            if len(self._state['portfolio_value']) > 1:
                daily_return = (self._state['total_value'] - self._state['previous_value']) / self._state['previous_value']
                logger.debug(f"Daily return: {daily_return:.2%}")
                
            self._state['previous_value'] = self._state['total_value']
            
        except Exception as e:
            logger.error(f"Error updating state: {str(e)}")
    
    
   
    
    def execute_trades(self, date, selected_symbols, weights, allocation_ratios, 
                      transaction_cost, combined_prices, trackers):
        """Execute trades with proper allocation ratios and position sizing"""
        try:
            logger.info("\nExecuting trades...")
            logger.info(f"Selected symbols: {len(selected_symbols)}")
            logger.info(f"Current cash: ${self._state['cash']:,.2f}")
            logger.info(f"Portfolio value: ${self._state['total_value']:,.2f}")
            
            # Calculate allocations
            total_value = self._state['total_value']
            options_allocation = allocation_ratios.get('options', 0.10)
            equities_allocation = allocation_ratios.get('equities', 0.90)
            
            # Reserve cash for options
            options_cash_reserve = total_value * options_allocation
            available_for_equities = self._state['cash'] - options_cash_reserve
            
            # Calculate current values and exposures
            current_options_value = sum(
                position['market_value'] 
                for position in self._state['holdings'].get('options', {}).values()
            )
            current_equities_value = sum(
                shares * combined_prices.loc[date, symbol]
                for symbol, shares in self._state['holdings'].get('equities', {}).items()
            )
            
            # Log allocation analysis
            logger.info("\nAllocation Analysis:")
            logger.info(f"Target Equities Allocation: {equities_allocation:.1%}")
            logger.info(f"Target Options Allocation: {options_allocation:.1%}")
            logger.info(f"Current Equities Value: ${current_equities_value:,.2f}")
            logger.info(f"Current Options Value: ${current_options_value:,.2f}")
            logger.info(f"Available for Equities: ${available_for_equities:,.2f}")
            
            if available_for_equities <= 0:
                logger.warning("Insufficient funds for equity trades after options reserve")
                return False
            
            # Sell positions not in new selection
            for symbol in list(self._state['holdings']['equities'].keys()):
                if symbol not in selected_symbols:
                    df, col_map = standardize_dataframe_columns(combined_prices)
                    if df is None:
                        continue
                    shares = self._state['holdings']['equities'][symbol]
                    price = combined_prices.loc[date, symbol]
                    sale_value = shares * price * (1 - transaction_cost)
                    trade_cost = shares * price * transaction_cost
                    
                    self._state['cash'] += sale_value
                    del self._state['holdings']['equities'][symbol]
                    
                    trackers['trade'].record_trade(
                        date=date,
                        action='sell',
                        symbol=symbol,
                        shares=-shares,
                        price=price,
                        value=shares * price,
                        cost=trade_cost,
                        trade_type='equity'
                    )
                    
                    logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f}")
            
            # Calculate target positions with allocation constraint
            available_cash = min(self._state['cash'] - options_cash_reserve, 
                            total_value * equities_allocation - current_equities_value)
            
            if available_cash <= 0:
                logger.warning("No cash available for new positions")
                return False
                
            target_positions = {}
            total_target_value = 0
            
            for symbol in selected_symbols:
                df, col_map = standardize_dataframe_columns(combined_prices)
                if df is None:
                    continue
                    
                price = combined_prices.loc[date, symbol]
                target_value = total_value * weights[symbol] * equities_allocation
                target_shares = int(target_value / price)
                
                # Minimum position size check
                min_position = max(2000, total_value * 0.02)
                if target_shares * price >= min_position:
                    target_positions[symbol] = {
                        'shares': target_shares,
                        'price': price,
                        'value': target_shares * price
                    }
                    total_target_value += target_shares * price
            
            # Scale positions if needed
            if total_target_value > available_cash:
                scale_factor = available_cash / total_target_value
                for target in target_positions.values():
                    target['shares'] = int(target['shares'] * scale_factor)
                    target['value'] = target['shares'] * target['price']
            
            # Execute trades
            executed_trades = []
            for symbol, target in target_positions.items():
                try:
                    current_shares = self._state['holdings']['equities'].get(symbol, 0)
                    shares_diff = target['shares'] - current_shares
                    
                    if shares_diff == 0:
                        continue
                        
                    trade_value = abs(shares_diff * target['price'])
                    trade_cost = trade_value * transaction_cost
                    
                    if shares_diff > 0:  # Buy
                        if trade_value + trade_cost <= self._state['cash'] - options_cash_reserve:
                            self._state['cash'] -= (trade_value + trade_cost)
                            
                            if symbol not in self._state['holdings']['equities']:
                                self._state['holdings']['equities'][symbol] = shares_diff
                            else:
                                self._state['holdings']['equities'][symbol] += shares_diff
                                
                            trackers['trade'].record_trade(
                                date=date,
                                action='buy',
                                symbol=symbol,
                                shares=shares_diff,
                                price=target['price'],
                                value=trade_value,
                                cost=trade_cost,
                                trade_type='equity'
                            )
                            executed_trades.append('buy')
                            
                    elif shares_diff < 0:  # Sell
                        self._state['holdings']['equities'][symbol] += shares_diff
                        if self._state['holdings']['equities'][symbol] == 0:
                            del self._state['holdings']['equities'][symbol]
                        self._state['cash'] += trade_value - trade_cost
                        
                        trackers['trade'].record_trade(
                            date=date,
                            action='sell',
                            symbol=symbol,
                            shares=shares_diff,
                            price=target['price'],
                            value=trade_value,
                            cost=trade_cost,
                            trade_type='equity'
                        )
                        executed_trades.append('sell')
                        
                except Exception as e:
                    logger.error(f"Error trading {symbol}: {str(e)}")
                    continue
            
            if self.strategy_type == 'etf' and symbol in self._state.get('sector_map', {}):
                sector = self._state['sector_map'][symbol]
                # Update sector weights after trade
                self._update_sector_weights(date, combined_prices)

            # Log execution summary
            logger.info("\nTrade Execution Summary:")
            logger.info(f"Trades executed: {len(executed_trades)}")
            logger.info(f"Final cash: ${self._state['cash']:,.2f}")
            logger.info(f"Options cash reserve: ${options_cash_reserve:,.2f}")
            logger.info(f"Final equity value: ${sum(shares * combined_prices.loc[date, sym] for sym, shares in self._state['holdings']['equities'].items()):,.2f}")
            
            return len(executed_trades) > 0
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            return False
        
    def _update_sector_weights(self, date, combined_prices):
        """Helper method to update sector weights"""
        try:
            total_value = self._state['total_value']
            sector_weights = defaultdict(float)
            
            for symbol, shares in self._state['holdings']['equities'].items():
                if symbol in self._state.get('sector_map', {}):
                    sector = self._state['sector_map'][symbol]
                    price = combined_prices.loc[date, symbol]
                    position_value = shares * price
                    sector_weights[sector] += position_value / total_value
                    
            self._state['sector_weights'] = sector_weights
            
        except Exception as e:
            logger.error(f"Error updating sector weights: {str(e)}")
        
    def _sell_existing_positions(self, date, new_symbols, transaction_cost, 
                               combined_prices, trackers):
        """Sell positions that are not in new selection"""
        try:
            for symbol in list(self._state['holdings']['equities'].keys()):
                if symbol not in new_symbols:
                    shares = self._state['holdings']['equities'][symbol]
                    price = combined_prices.loc[date, symbol]
                    
                    sale_value = shares * price
                    trade_cost = sale_value * transaction_cost
                    net_value = sale_value - trade_cost
                    
                    # Update cash and holdings
                    self._state['cash'] += net_value
                    del self._state['holdings']['equities'][symbol]
                    
                    # Record trade
                    trackers['trade'].record_trade(
                        date=date,
                        action='sell',
                        symbol=symbol,
                        shares=-shares,
                        price=price,
                        value=sale_value,
                        cost=trade_cost
                    )
                    
                    # Add to transactions
                    trackers['transactions'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': -shares,
                        'price': price,
                        'value': sale_value,
                        'cost': trade_cost
                    })
                    
                    logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f}")
                    logger.info(f"Sale value: ${sale_value:,.2f}")
                    
        except Exception as e:
            logger.error(f"Error selling positions: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _sell_all_positions(self, date, transaction_cost, combined_prices, trackers):
        """Sell all positions with proper cash management"""
        try:
            for symbol, shares in list(self._state['holdings']['equities'].items()):
                price = combined_prices.loc[date, symbol]
                sale_value = shares * price
                trade_cost = sale_value * transaction_cost
                net_value = sale_value - trade_cost
                
                # Update cash and holdings
                self._state['cash'] += net_value
                del self._state['holdings']['equities'][symbol]
                
                # Record trade
                trackers['trade'].record_trade(
                    date=date,
                    action='sell',
                    symbol=symbol,
                    shares=-shares,
                    price=price,
                    value=sale_value,
                    cost=trade_cost
                )
                
                logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f}")
                
        except Exception as e:
            logger.error(f"Error selling positions: {str(e)}")
    
    def handle_options_trading(self, date, options_data, price_data, treasury_data,
                              option_pricing_model, allocation_ratios, transaction_cost, trackers):
        """Delegate options trading to EnhancedOptionsStrategy while maintaining portfolio state"""
        try:
            # Run options trading through strategy
            new_holdings, new_cash, transactions = self.options_strategy.trade_options(
                holdings=self._state['holdings'],
                cash=self._state['cash'],
                options_data=options_data,
                price_data=price_data,
                date=date,
                allocation_ratios=allocation_ratios,
                transaction_cost=transaction_cost,
                trackers=trackers, 
                treasury_data=treasury_data,
                option_model=option_pricing_model
            )
            
            # Update portfolio state with results
            self._state['holdings'] = new_holdings
            self._state['cash'] = new_cash
            
            # Record transactions
            for transaction in transactions:
                trackers['transactions'].append(transaction)
                    
        except Exception as e:
            logger.error(f"Error in options trading: {str(e)}")
            logger.error(traceback.format_exc())


        
    def update_options_positions(self, date, options_data, price_data, treasury_data):
        """Update options positions and handle expirations"""
        try:
            # Track value before update
            previous_options_value = self._state['options_value']
            
            # Update each option position
            for option_key in list(self._state['holdings']['options'].keys()):
                position = self._state['holdings']['options'][option_key]
                
                # Check expiration
                if pd.to_datetime(position['expiration']) <= date:
                    # Calculate expiration value
                    intrinsic_value = self._calculate_option_value(
                        position, date, self._state['combined_prices']
                    )
                    
                    # Add value to cash
                    self._state['cash'] += intrinsic_value
                    
                    # Remove position
                    del self._state['holdings']['options'][option_key]
                    
                    logger.info(f"Option expired: {option_key}")
                    logger.info(f"Expiration value: ${intrinsic_value:,.2f}")
                    
                else:
                    # Update market value using available options data
                    symbol_options = options_data.get(position['symbol'])
                    if symbol_options is not None:
                        matching_options = symbol_options[
                            (symbol_options['strike'] == position['strike']) &
                            (symbol_options['optionType'] == position['type']) &
                            (symbol_options['expirationDate'] == position['expiration'])
                        ]
                        
                        if not matching_options.empty:
                            mid_price = (matching_options['bid'].iloc[0] + 
                                       matching_options['ask'].iloc[0]) / 2
                            market_value = mid_price * position['contracts'] * 100
                        else:
                            # Use theoretical value if market data not available
                            market_value = self._calculate_option_value(
                                position, date, self._state['combined_prices']
                            )
                            
                        position['market_value'] = market_value
                    
            # Calculate total options value
            self._state['options_value'] = sum(
                pos['market_value'] 
                for pos in self._state['holdings']['options'].values()
            )
            
            # Log options update
            logger.info("\nOptions Update:")
            logger.info(f"Options positions: {len(self._state['holdings']['options'])}")
            logger.info(f"Total options value: ${self._state['options_value']:,.2f}")
            logger.info(f"Value change: ${self._state['options_value'] - previous_options_value:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating options positions: {str(e)}")
            logger.error(traceback.format_exc())
            return False

                   
    def _calculate_option_payoff(self, option_type, underlying_price, strike, quantity):
        """Calculate option payoff at expiration"""
        if option_type == 'call':
            payoff = max(0, underlying_price - strike) * quantity * 100
        else:
            payoff = max(0, strike - underlying_price) * quantity * 100
        return payoff

    def _get_option_market_value(self, option, current_options_data):
        """Helper method to get current market value of an option"""
        try:
            matching_options = current_options_data[
                (current_options_data['strike'] == option['contract']['strike']) &
                (current_options_data['optionType'] == option['contract']['optionType']) &
                (current_options_data['expirationDate'] == option['contract']['expirationDate'])
            ]
            
            if not matching_options.empty:
                mid_price = (matching_options['bid'].iloc[0] + 
                            matching_options['ask'].iloc[0]) / 2
                return mid_price * option['quantity'] * 100
            else:
                logger.warning(f"No market data found for option: {option['contract']}")
                return option.get('market_value', 0)  # Return last known value
                
        except Exception as e:
            logger.error(f"Error getting option market value: {str(e)}")
            return 0
        
    
        
    def handle_risk_breach(self, date, trackers):
        """Handle risk limit breaches with recovery logic"""
        try:
            # Make sure date is a pd.Timestamp
            date = pd.to_datetime(date)
            
            logger.info(f"\nRISK BREACH DETECTED on {date}")
            logger.info(f"Current portfolio value: ${self._state['total_value']:,.2f}")
            logger.info(f"Initial capital: ${self._state['initial_capital']:,.2f}")
            
            # Enter recovery mode with proper date handling
            self._state.update({
                'in_recovery': True,
                'recovery_start_value': float(self._state['total_value']),
                'recovery_start_date': date,
                'lowest_value_since_exit': float(self._state['total_value']),
                'last_recovery_check': date
            })
            
            # Verify recovery state was set correctly
            logger.info("\nVerifying recovery state:")
            logger.info(f"In recovery: {self._state['in_recovery']}")
            logger.info(f"Recovery start date: {self._state['recovery_start_date']}")
            logger.info(f"Recovery start value: ${self._state['recovery_start_value']:,.2f}")
            
            # Record the risk event
            trackers['trade'].record_risk_event(
                date=date,
                event_type='risk_limit_breach',
                details={
                    'portfolio_value': self._state['total_value'],
                    'initial_capital': self._state['initial_capital'],
                    'recovery_start_date': self._state['recovery_start_date']
                }
            )
            
            # Liquidate all positions
            if self._state['holdings']['equities'] or self._state['holdings']['options']:
                logger.info("\nLiquidating all positions...")
                
                # Liquidate equity positions
                self._liquidate_all_positions(date, trackers)
                
                # Liquidate options positions
                self._liquidate_options_positions(date, trackers)  # Remove state parameter
                
                logger.info("Liquidation complete")
                
                # Update portfolio state
                self.update(date, self._state['combined_prices'])
                
        except Exception as e:
            logger.error(f"Error handling risk breach: {str(e)}")
            logger.error(traceback.format_exc())

    def _liquidate_all_positions(self, date, trackers):
        """Liquidate all positions and update state"""
        try:
            # Liquidate equities
            for symbol, shares in list(self._state['holdings']['equities'].items()):
                try:
                    price = self._state['combined_prices'].loc[date, symbol]
                    sale_value = shares * price * (1 - TRANSACTION_COST)
                    trade_cost = shares * price * TRANSACTION_COST
                    
                    self._state['cash'] += sale_value

                    trackers['transactions'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': -shares,
                        'price': price,
                        'value': sale_value,
                        'cost': trade_cost,
                        'reason': 'risk_liquidation'
                    })
                    
                    logger.info(f"Liquidated {shares} shares of {symbol} at ${price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error liquidating {symbol}: {str(e)}")
                    continue
            
            # Clear equity holdings
            self._state['holdings']['equities'] = {}
            
            # Liquidate options
            for symbol in list(self._state['holdings']['options'].keys()):
                try:
                    for option in self._state['holdings']['options'][symbol]:
                        trackers['transactions'].append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'close_option',
                            'quantity': -option['quantity'],
                            'price': option.get('entry_price', 0),
                            'value': 0,
                            'cost': 0,
                            'reason': 'risk_liquidation'
                        })
                except Exception as e:
                    logger.error(f"Error liquidating options for {symbol}: {str(e)}")
                    continue
            
            # Clear options holdings
            self._state['holdings']['options'] = {}
            
            logger.info(f"All positions liquidated. Cash: ${self._state['cash']:,.2f}")
            
        except Exception as e:
            logger.error(f"Error in liquidation: {str(e)}")
            logger.error(traceback.format_exc())

    def _liquidate_options_positions(self, date, trackers):
        """Liquidate all options positions"""
        try:
            if not self._state['holdings']['options']:
                return
                
            for option_id in list(self._state['holdings']['options'].keys()):
                position = self._state['holdings']['options'][option_id]
                current_price = self._get_option_price(position, date)
                
                if current_price is None:
                    continue
                    
                # Calculate trade details
                trade_value = position['contracts'] * current_price * 100
                trade_cost = abs(trade_value) * TRANSACTION_COST
                net_value = trade_value - trade_cost
                
                # Update cash
                self._state['cash'] += net_value
                
                # Record trade
                trackers['trade'].record_trade(
                    date=date,
                    action='close_option',
                    symbol=position['symbol'],
                    quantity=position['contracts'],
                    price=current_price,
                    value=trade_value,
                    cost=trade_cost
                )
                
                # Remove position
                del self._state['holdings']['options'][option_id]
                
        except Exception as e:
            logger.error(f"Error liquidating options positions: {str(e)}")

    def check_recovery_status(self, date):
        """Check if portfolio has recovered enough to resume trading"""
        try:
            if not self._state.get('in_recovery', False):
                return True
                
            current_value = self._state['portfolio_value'].iloc[-1]
            recovery_value = self._state['recovery_start_value']
            days_in_recovery = (date - self._state['recovery_start_date']).days
            recovery_pct = (current_value - recovery_value) / recovery_value
            
            # Exit recovery if either threshold met
            if recovery_pct >= MIN_RECOVERY or days_in_recovery >= MAX_RECOVERY_DAYS:
                self._state['in_recovery'] = False
                logger.info(f"Exiting recovery mode. Recovery: {recovery_pct:.2%}, Days: {days_in_recovery}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking recovery status: {str(e)}")
            return False

  
   
    def can_resume_trading(self, date):
        """Revised recovery logic with stronger market confirmation"""
        try:
            if not self._state.get('in_recovery', False):
                return True
                
            date = pd.to_datetime(date)
            recovery_start = pd.to_datetime(self._state['recovery_start_date'])
            days_in_recovery = (date - recovery_start).days
            
            logger.info(f"\nRecovery Analysis - Day {days_in_recovery}")
            
            # Don't even try for first 5 days
            if days_in_recovery < 5:
                logger.info("Initial waiting period")
                return False
            
            # Get market data
            try:
                # Get more history for better analysis
                market_data = yf.download('SPY', 
                                        start=(date - pd.Timedelta(days=90)),
                                        end=date + pd.Timedelta(days=1),
                                        progress=False)['Adj Close']
                
                # Calculate market metrics
                market_returns = market_data.pct_change()
                vol = market_returns.std() * np.sqrt(252)
                
                # Moving averages
                ma10 = market_data.rolling(10).mean()
                ma20 = market_data.rolling(20).mean()
                ma50 = market_data.rolling(50).mean()
                
                # Trend strength
                current_price = market_data.iloc[-1]
                trend_score = sum([
                    current_price > ma10.iloc[-1],
                    current_price > ma20.iloc[-1],
                    current_price > ma50.iloc[-1],
                    ma10.iloc[-1] > ma10.iloc[-5],
                    ma20.iloc[-1] > ma20.iloc[-5]
                ])
                
                # Recent performance
                last_5d_return = (market_data.iloc[-1] / market_data.iloc[-5] - 1)
                last_10d_return = (market_data.iloc[-1] / market_data.iloc[-10] - 1)
                
                logger.info("\nMarket Analysis:")
                logger.info(f"Volatility: {vol:.1%}")
                logger.info(f"5-day return: {last_5d_return:.1%}")
                logger.info(f"10-day return: {last_10d_return:.1%}")
                logger.info(f"Trend score (0-5): {trend_score}")
                
                # Phased re-entry conditions
                if days_in_recovery <= 10:
                    # Phase 1 (5-10 days): Need very strong confirmation
                    should_resume = (
                        trend_score >= 4 and
                        last_5d_return > 0.02 and
                        vol < 0.30
                    )
                    logger.info("Phase 1 - Strong Confirmation Required")
                    
                elif days_in_recovery <= 20:
                    # Phase 2 (11-20 days): Need good confirmation
                    should_resume = (
                        trend_score >= 3 and
                        last_5d_return > 0.01 and
                        vol < 0.35
                    )
                    logger.info("Phase 2 - Good Confirmation Required")
                    
                else:
                    # Phase 3 (21+ days): More relaxed conditions
                    should_resume = (
                        trend_score >= 2 and
                        vol < 0.40
                    )
                    logger.info("Phase 3 - Moderate Confirmation Required")
                
                if should_resume:
                    logger.info("\nRe-entry conditions met:")
                    logger.info(f"- Trend Score: {trend_score}")
                    logger.info(f"- Recent Return: {last_5d_return:.1%}")
                    logger.info(f"- Volatility: {vol:.1%}")
                    self._state['in_recovery'] = False
                    return True
                
                logger.info("\nMaintaining recovery mode")
                return False
                
            except Exception as e:
                logger.error(f"Market analysis error: {str(e)}")
                return False
        
        except Exception as e:
            logger.error(f"Recovery check error: {str(e)}")
            return False
                
    def monitor_positions(self, date, combined_prices):
        """Monitor existing positions for deteriorating conditions"""
        try:
            positions_to_exit = []
            
            for symbol, shares in self._state['holdings']['equities'].items():
                df = self._state['price_data'][symbol]
                current_price = self._state['combined_prices'].loc[date, symbol]
                
                returns = df['close'].pct_change()
                current_price = combined_prices.loc[date, symbol]
                entry_price = self._state.get('entry_prices', {}).get(symbol)
                
                if entry_price is not None:
                    position_return = (current_price - entry_price) / entry_price
                    
                    # Exit criteria
                    if (position_return < -0.1 or              # 10% loss
                        returns.tail(5).mean() < -0.02 or     # Recent negative trend
                        returns.std() * np.sqrt(252) > 0.4):  # High volatility
                        
                        positions_to_exit.append(symbol)
                        logger.warning(f"Flagging {symbol} for exit:")
                        logger.warning(f"Position Return: {position_return:.2%}")
                        logger.warning(f"Recent Trend: {returns.tail(5).mean():.2%}")
                        logger.warning(f"Volatility: {returns.std() * np.sqrt(252):.2%}")
            
            return positions_to_exit
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
            return []
        

    def _check_trading_conditions(self, date, current_value, combined_prices):
        """Check if trading conditions are met"""
        try:
            # Check if in recovery mode
            if self._state.get('in_recovery', False):
                recovery_pct = (current_value - self._state['recovery_start_value']) / self._state['recovery_start_value']
                days_in_recovery = (date - self._state['recovery_start_date']).days
                
                if recovery_pct < MIN_RECOVERY and days_in_recovery < MAX_RECOVERY_DAYS:
                    return False
                    
            # Calculate drawdown
            if self.peak_value is None:
                self.peak_value = current_value
            elif current_value > self.peak_value:
                self.peak_value = current_value
                
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > MAX_DRAWDOWN:
                logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
                return False
                
            # Calculate volatility if enough history
            if len(self._state['portfolio_value']) >= 20:
                returns = self._state['portfolio_value'].pct_change()
                volatility = returns.tail(20).std() * np.sqrt(252)
                if volatility > MAX_VOLATILITY:
                    logger.warning(f"Max volatility exceeded: {volatility:.2%}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {str(e)}")
            return False
    
    def log_portfolio_state(self):
        """Log current portfolio state"""
        try:
            logger.info("\nPortfolio State:")
            logger.info(f"Cash: ${self._state['cash']:,.2f}")
            logger.info(f"Total Value: ${self._state['total_value']:,.2f}")

            if self._state['holdings']['equities']:
                logger.info("\nEquity Holdings:")
                for symbol, shares in self._state['holdings']['equities'].items():
                    value = shares * float(self._state['combined_prices'].loc[self._state['combined_prices'].index[-1], symbol])
                            
            if self._state.get('in_recovery', False):
                logger.info("\nRecovery Status:")
                logger.info(f"Recovery start date: {self._state.get('recovery_start_date')}")
                logger.info(f"Recovery start value: ${self._state.get('recovery_start_value', 0):,.2f}")
                logger.info(f"Lowest value since exit: ${self._state.get('lowest_value_since_exit', 0):,.2f}")
                
            # Portfolio metrics if we have enough history
            if len(self._state['portfolio_value']) > 1:
                returns = self._state['portfolio_value'].pct_change()
                logger.info("\nPortfolio Metrics:")
                logger.info(f"Daily return: {returns.iloc[-1]:.2%}")
                logger.info(f"Volatility (20d): {returns.tail(20).std() * np.sqrt(252):.2%}")
                
                # Calculate drawdown
                peak = self._state['portfolio_value'].expanding().max()
                drawdown = (self._state['portfolio_value'] - peak) / peak
                logger.info(f"Current drawdown: {drawdown.iloc[-1]:.2%}")
                
        except Exception as e:
            logger.error(f"Error logging portfolio state: {str(e)}")
            logger.error(traceback.format_exc())

class PortfolioStateExtended(PortfolioState):
    def initialize_data(self, price_data, options_data, treasury_data, option_pricing_model):
        self._state.update({
            'price_data': price_data,
            'options_data': options_data,
            'treasury_data': treasury_data,
            'option_pricing_model': option_pricing_model,
            'options_enabled': option_pricing_model is not None and option_pricing_model.is_trained
        })


    def execute_rebalancing_trades(self, date: datetime,
                             allocations: Dict[str, float],
                             transaction_cost: float,
                             trackers: Dict,
                             allocation_ratios: Optional[Dict] = None) -> bool:
        """Execute rebalancing trades"""
        try:
            total_value = self._state['total_value']
            current_holdings = self._state['holdings']['equities']
            combined_prices = self._state['combined_prices']
            executed = False
            
            # Check ETF allocation limits if ratios provided
            if allocation_ratios and 'etfs' in allocation_ratios:
                etf_limit = min(
                    self.etf_constraints['max_etf_allocation'],
                    allocation_ratios['etfs']
                )
                # Filter out ETF allocations that would exceed limit
                allocations = self._filter_etf_allocations(allocations, etf_limit)

            # Sell positions not in new allocation
            for symbol in list(current_holdings.keys()):
                if symbol not in allocations:
                    shares = current_holdings[symbol]
                    price = combined_prices.loc[date, symbol]
                    sale_value = shares * price * (1 - transaction_cost)
                    trade_cost = shares * price * transaction_cost
                    
                    self._state['cash'] += sale_value
                    del current_holdings[symbol]
                    
                    trackers['trade'].record_trade(
                        date=date,
                        action='sell',
                        symbol=symbol,
                        shares=-shares,
                        price=price,
                        value=shares * price,
                        cost=trade_cost,
                        trade_type='equity' 
                    )
                    executed = True
                    
            # Execute new allocations
            for symbol, target_weight in allocations.items():
                price = combined_prices.loc[date, symbol]
                target_value = total_value * target_weight
                target_shares = int(target_value / price)
                current_shares = current_holdings.get(symbol, 0)
                
                if target_shares != current_shares:
                    shares_diff = target_shares - current_shares
                    trade_value = abs(shares_diff * price)
                    trade_cost = trade_value * transaction_cost
                    
                    if shares_diff > 0:  # Buy
                        if trade_value + trade_cost <= self._state['cash']:
                            self._state['cash'] -= (trade_value + trade_cost)
                            current_holdings[symbol] = target_shares
                            
                            trackers['trade'].record_trade(
                                date=date,
                                action='buy',
                                symbol=symbol,
                                shares=shares_diff,
                                price=price,
                                value=trade_value,
                                cost=trade_cost
                            )
                            executed = True
                            
                    else:  # Sell
                        sale_value = trade_value * (1 - transaction_cost)
                        self._state['cash'] += sale_value
                        current_holdings[symbol] = target_shares
                        
                        trackers['trade'].record_trade(
                            date=date,
                            action='sell',
                            symbol=symbol,
                            shares=shares_diff,
                            price=price,
                            value=trade_value,
                            cost=trade_cost
                        )
                        executed = True
            
            return executed
            
        except Exception as e:
            logger.error(f"Error executing rebalancing trades: {str(e)}")
            return False
        