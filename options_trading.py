# options_trading.py


import numpy as np
import pandas as pd
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
import yfinance as yf
from config import EQUITY_END_DATE, RISK_FREE_RATE
from portfolio_state import PortfolioState
from black_scholes import black_scholes_price
import logging
import traceback
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from column_utils import standardize_dataframe_columns, get_column_name


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EnhancedOptionsStrategy:

    
    
    def __init__(self, 
                 min_volume: int = 1000,
                 min_open_interest: int = 500,
                 min_implied_vol: float = 0.15,
                 max_implied_vol: float = 0.60,
                 min_days_to_expiry: int = 30,
                 max_days_to_expiry: int = 90,
                 min_mispricing: float = 0.02,
                 max_bid_ask_spread: float = 0.10):
        """
        Initialize options strategy with enhanced screening criteria
        """
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.min_implied_vol = min_implied_vol
        self.max_implied_vol = max_implied_vol
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.max_bid_ask_spread = max_bid_ask_spread
        self.min_mispricing = min_mispricing 
        self.active_positions = {}
        self.position_history = []
           
    
    
    def _calculate_option_value(self, option, current_price):
        """Calculate current option value"""
        try:
            # Calculate days to expiry
            expiry = pd.to_datetime(option['expiration'])
            days_to_expiry = (expiry - pd.Timestamp.now()).days
            
            if days_to_expiry <= 0:
                return self._calculate_expiration_value(option, current_price)
                
            # For active positions, use market price if available
            if 'market_price' in option:
                return option['market_price'] * option['contracts'] * 100
                
            # Otherwise calculate theoretical value
            if option['type'].lower() == 'call':
                intrinsic = max(0, current_price - option['strike'])
            else:
                intrinsic = max(0, option['strike'] - current_price)
                
            # Simple time decay approximation
            time_value = option['entry_price'] * (days_to_expiry / option.get('original_days', 30))
            option_value = (intrinsic + time_value) * option['contracts'] * 100
            
            return option_value
            
        except Exception as e:
            logger.error(f"Error calculating option value: {str(e)}")
            return 0
                
    def _calculate_expiration_value(self, option, current_price):
        """Calculate option value at expiration"""
        try:
            if option['type'].lower() == 'call':
                value = max(0, current_price - option['strike'])
            else:
                value = max(0, option['strike'] - current_price)
            return value * option['contracts'] * 100
        except Exception as e:
            logger.error(f"Error calculating expiration value: {str(e)}")
            return 0



    
    def _evaluate_options(self, symbol, options_data, current_price, shares_owned):
        """Evaluate options with profit calculations"""
        try:
            opportunities = []
            
            if shares_owned >= 100:  # For covered calls
                df, col_map = standardize_dataframe_columns(options_data)
                if df is None:
                    return opportunities
                    
                calls = df[df['optionType'] == 'call']
                if not calls.empty:
                    otm_calls = calls[calls['strike'] > current_price * 1.02]
                    if not otm_calls.empty:
                        otm_calls['premium_ratio'] = otm_calls['bid'] / otm_calls['strike']
                        otm_calls = otm_calls.sort_values('premium_ratio', ascending=False)
                        
                        for _, call in otm_calls.head(3).iterrows():
                            premium = call['bid']
                            strike = call['strike']
                            max_profit = premium * 100
                            
                            opp = {
                                'type': 'covered_call',
                                'strike': strike,
                                'expiration': call['expirationDate'],
                                'market_price': premium,
                                'premium_ratio': call['premium_ratio'],
                                'expected_profit': max_profit,
                                'max_loss': (current_price - strike + premium) * 100 if strike < current_price else 0
                            }
                            opportunities.append(opp)

            return opportunities

        except Exception as e:
            logger.error(f"Error evaluating options: {str(e)}")
            return []
        
    def _execute_trades(self, holdings, cash, options_data, price_data, date,
                       transaction_cost, trackers):
        """Execute options trades with improved position sizing"""
        try:
            executed_trades = []
            remaining_cash = cash
            
            # Only trade options on stocks we own
            for symbol in holdings['equities']:
                if symbol not in options_data:
                    continue
                    
                try:
                    current_price = price_data[symbol].loc[
                        price_data[symbol]['date'] == date, 'close'
                    ].iloc[0]
                    
                    opportunities = self._evaluate_options(
                        symbol=symbol,
                        options_data=options_data[symbol],
                        current_price=current_price
                    )
                    
                    if not opportunities:
                        continue
                        
                    # Take best opportunity
                    best_opp = opportunities[0]
                    
                    # Calculate position size - more aggressive
                    contract_cost = best_opp['market_price'] * 100
                    total_cost_per_contract = contract_cost * (1 + transaction_cost)
                    
                    # Allow up to 5% of portfolio per option position
                    max_allocation = remaining_cash * 0.05
                    max_contracts = min(
                        int(max_allocation / total_cost_per_contract),
                        5  # Maximum 5 contracts per position
                    )
                    
                    if max_contracts <= 0:
                        continue
                    
                    total_cost = contract_cost * max_contracts
                    trade_cost = total_cost * transaction_cost
                    
                    # Record trade
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'type': best_opp['type'],
                        'strike': best_opp['strike'],
                        'expiration': best_opp['expiration'],
                        'contracts': max_contracts,
                        'price': best_opp['market_price'],
                        'total_cost': total_cost,
                        'transaction_cost': trade_cost,
                        'action': 'buy_option',
                        'expected_profit': best_opp['expected_profit'] * max_contracts
                    }
                    
                    if trackers is not None:
                        trackers['trade'].record_trade(
                            date=date,
                            action='buy_option',
                            symbol=symbol,
                            quantity=max_contracts,
                            price=best_opp['market_price'],
                            value=total_cost,
                            cost=trade_cost
                        )
                    
                    executed_trades.append(trade)
                    remaining_cash -= (total_cost + trade_cost)
                    
                except Exception as e:
                    continue
                    
            return executed_trades
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            return []
        
    def _update_holdings(self, holdings, cash, trades, date):
        """Update holdings after trades"""
        try:
            for trade in trades:
                option_id = f"{trade['symbol']}_{trade['strike']}_{trade['expiration']}"
                holdings['options'][option_id] = {
                    'symbol': trade['symbol'],
                    'type': trade['type'],
                    'strike': trade['strike'],
                    'expiration': trade['expiration'],
                    'contracts': trade['contracts'],
                    'entry_price': trade['price'],
                    'entry_date': date,
                    'market_value': trade['total_cost']
                }
                cash -= (trade['total_cost'] + trade['transaction_cost'])
                
            return holdings, cash
            
        except Exception as e:
            logger.error(f"Error updating holdings: {str(e)}")
            return holdings, cash
                
    def _calculate_total_value(self, holdings, price_data, date):
        """Calculate total portfolio value"""
        total = holdings.get('cash', 0)
        
        for symbol, shares in holdings.get('equities', {}).items():
            try:
                price = price_data[symbol].loc[price_data[symbol]['date'] == date, 'close'].iloc[0]
                total += shares * price
            except Exception as e:
                logger.warning(f"Error getting price for {symbol}: {str(e)}")
                continue
            
        return total + self._get_current_options_value(holdings)
        
    def _get_current_options_value(self, holdings):
        """Get current value of options positions"""
        return sum(
            pos.get('market_value', 0) 
            for pos in holdings.get('options', {}).values()
        )

    def manage_option_positions(self, date, holdings, current_prices):
        """Manage existing option positions with risk controls"""
        try:
            state = PortfolioState(holdings)
            actions = []
            date = pd.to_datetime(date)
            
            for option_id, position in list(state._state['holdings']['options'].items()):
                try:
                    expiry = pd.to_datetime(position['expiration'])
                    
                    # Handle expiration
                    if date >= expiry:
                        actions.append({
                            'action': 'expire',
                            'option_id': option_id,
                            'position': position
                        })
                        continue
                        
                    # Check profit targets and time decay
                    days_to_expiry = (expiry - date).days
                    symbol = position['symbol']
                    current_price = current_prices.get(symbol)
                    
                    if current_price is not None:
                        value = self._calculate_option_value(position, current_price)
                        entry_value = position['entry_price'] * position['contracts'] * 100
                        
                        # Close profitable positions near expiry
                        if days_to_expiry <= 5 or (value > entry_value * 1.5):
                            actions.append({
                                'action': 'close',
                                'option_id': option_id,
                                'position': position
                            })
                            
                except Exception as e:
                    logger.error(f"Error managing position {option_id}: {str(e)}")
                    continue
                    
            return actions
            
        except Exception as e:
            logger.error(f"Error managing options positions: {str(e)}")
            return []
        
    def manage_positions(self, holdings, date, current_prices):
        """Manage existing option positions"""
        actions = []
        date = pd.to_datetime(date)
        
        for option_id, position in list(holdings['options'].items()):
            try:
                expiry = pd.to_datetime(position['expiration'])
                
                # Handle expiration
                if date >= expiry:
                    actions.append({
                        'action': 'expire',
                        'option_id': option_id,
                        'position': position
                    })
                    continue
                    
                # Check profit targets
                days_to_expiry = (expiry - date).days
                if days_to_expiry <= 5:  # Close positions near expiry
                    actions.append({
                        'action': 'close',
                        'option_id': option_id,
                        'position': position
                    })
                    
            except Exception as e:
                logger.error(f"Error managing position {option_id}: {str(e)}")
                continue
                
        return actions
    

    def _validate_options_date(self, date, options_data):
        """Validate options data for current trading options"""
        try:
            if not options_data:
                logger.warning("No options data provided")
                return False
                
            # Debug data structure
            logger.info("\nAnalyzing current options data:")
            logger.info(f"Number of symbols in options data: {len(options_data)}")
            sample_symbol = list(options_data.keys())[0]
            sample_data = options_data[sample_symbol]
            
            logger.info(f"\nSample data for {sample_symbol}:")
            logger.info(f"Data type: {type(sample_data)}")
            if isinstance(sample_data, pd.DataFrame):
                logger.info(f"Columns: {sample_data.columns.tolist()}")
                logger.info(f"Data shape: {sample_data.shape}")
                
                # Check expiration dates
                if 'expirationDate' in sample_data.columns:
                    expirations = pd.to_datetime(sample_data['expirationDate']).unique()
                    logger.info(f"\nAvailable expiration dates:")
                    for exp in sorted(expirations):
                        logger.info(f"- {exp}")
            
            # For current options trading, we care about:
            # 1. Having valid strikes
            # 2. Having valid bid/ask prices
            # 3. Having future expiration dates
            valid_count = 0
            total_count = len(options_data)
            current_date = pd.to_datetime(date)
            
            for symbol, df in options_data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                    
                # Check if we have valid options data
                if all(col in df.columns for col in ['strike', 'bid', 'ask', 'expirationDate']):
                    # Verify we have options with future expirations
                    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
                    if any(df['expirationDate'] > current_date):
                        valid_count += 1
            
            validity_ratio = valid_count / total_count if total_count > 0 else 0
            logger.info(f"\nOptions data validity:")
            logger.info(f"Total symbols: {total_count}")
            logger.info(f"Symbols with valid options: {valid_count}")
            logger.info(f"Validity ratio: {validity_ratio:.2%}")
            
            if validity_ratio < 0.5:
                logger.warning("Insufficient valid options data")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating options data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    #new enhancement for batch

    def screen_options(self, 
                      options_data: pd.DataFrame,
                      underlying_price: float) -> pd.DataFrame:
        """
        Screen options with comprehensive criteria
        """
        try:
            if options_data.empty:
                return pd.DataFrame()

            screened = options_data[
                # Liquidity criteria
                (options_data['volume'] >= self.min_volume) &
                (options_data['openInterest'] >= self.min_open_interest) &
                
                # Volatility criteria
                (options_data['impliedVolatility'] >= self.min_implied_vol) &
                (options_data['impliedVolatility'] <= self.max_implied_vol) &
                
                # Time criteria
                (options_data['daysToExpiration'] >= self.min_days_to_expiry) &
                (options_data['daysToExpiration'] <= self.max_days_to_expiry) &
                
                # Spread criteria
                ((options_data['ask'] - options_data['bid']) / 
                 ((options_data['bid'] + options_data['ask']) / 2) <= self.max_bid_ask_spread)
            ].copy()

            # Add moneyness calculation
            screened['moneyness'] = screened['strike'] / underlying_price
            
            # Filter based on moneyness
            screened = screened[
                (screened['moneyness'] >= 0.8) &
                (screened['moneyness'] <= 1.2)
            ]

            return screened

        except Exception as e:
            logger.error(f"Error screening options: {str(e)}")
            return pd.DataFrame()

    def calculate_option_score(self, 
                             option: pd.Series,
                             market_condition: str = 'normal') -> float:
        """
        Calculate comprehensive option score with market adjustment
        """
        try:
            # Base component scores
            volume_score = min(1.0, option['volume'] / 10000)
            oi_score = min(1.0, option['openInterest'] / 5000)
            spread_score = 1.0 - ((option['ask'] - option['bid']) / 
                                ((option['bid'] + option['ask']) / 2))

            # Volatility scoring
            iv = option['impliedVolatility']
            if market_condition == 'high_volatility':
                iv_target = 0.40
            elif market_condition == 'low_volatility':
                iv_target = 0.20
            else:
                iv_target = 0.30
            iv_score = 1.0 - abs(iv - iv_target) / iv_target

            # Time value scoring
            days = option['daysToExpiration']
            time_score = 1.0 - abs(days - 45) / 45  # Target 45 days

            # Moneyness scoring
            moneyness = option['moneyness']
            moneyness_score = 1.0 - abs(moneyness - 1.0)

            # Combine scores with market-adjusted weights
            if market_condition == 'high_volatility':
                weights = {
                    'volume': 0.25,
                    'oi': 0.20,
                    'spread': 0.20,
                    'iv': 0.15,
                    'time': 0.10,
                    'moneyness': 0.10
                }
            else:
                weights = {
                    'volume': 0.20,
                    'oi': 0.15,
                    'spread': 0.15,
                    'iv': 0.20,
                    'time': 0.15,
                    'moneyness': 0.15
                }

            total_score = (
                volume_score * weights['volume'] +
                oi_score * weights['oi'] +
                spread_score * weights['spread'] +
                iv_score * weights['iv'] +
                time_score * weights['time'] +
                moneyness_score * weights['moneyness']
            )

            return float(total_score)

        except Exception as e:
            logger.error(f"Error calculating option score: {str(e)}")
            return 0.0

    def find_option_opportunities(self, 
                                options_data: pd.DataFrame,
                                equity_price: float,
                                market_condition: str = 'normal') -> List[Dict]:
        """
        Find option trading opportunities with enhanced analytics
        """
        try:
            opportunities = []
            
            # Screen options
            screened_options = self.screen_options(options_data, equity_price)
            if screened_options.empty:
                return opportunities

            # Score each option
            for _, option in screened_options.iterrows():
                score = self.calculate_option_score(option, market_condition)
                
                if score > 0.7:  # Minimum quality threshold
                    mid_price = (option['bid'] + option['ask']) / 2
                    
                    # Calculate Greeks
                    option_greeks = self.calculate_greeks(
                        option, 
                        equity_price, 
                        RISK_FREE_RATE
                    )

                    opportunity = {
                        'symbol': option['symbol'],
                        'type': option['optionType'],
                        'strike': option['strike'],
                        'expiration': option['expirationDate'],
                        'market_price': mid_price,
                        'implied_vol': option['impliedVolatility'],
                        'score': score,
                        'greeks': option_greeks,
                        'bid_ask_spread': (option['ask'] - option['bid']) / mid_price,
                        'volume': option['volume'],
                        'open_interest': option['openInterest']
                    }
                    
                    opportunities.append(opportunity)

            # Sort by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            return opportunities

        except Exception as e:
            logger.error(f"Error finding opportunities: {str(e)}")
            return []

    def calculate_greeks(self, 
                        option: pd.Series,
                        stock_price: float,
                        risk_free_rate: float) -> Dict[str, float]:
        """
        Calculate option Greeks
        """
        try:
            flag = 'c' if option['optionType'].lower() == 'call' else 'p'
            S = stock_price
            K = option['strike']
            T = option['daysToExpiration'] / 365
            r = risk_free_rate
            sigma = option['impliedVolatility']

            return {
                'delta': delta(flag, S, K, T, r, sigma),
                'gamma': gamma(flag, S, K, T, r, sigma),
                'theta': theta(flag, S, K, T, r, sigma),
                'vega': vega(flag, S, K, T, r, sigma),
                'rho': rho(flag, S, K, T, r, sigma)
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

    def trade_options(self, holdings, cash, options_data, price_data, date, allocation_ratios, transaction_cost, trackers=None, treasury_data=None, option_model=None):
        """
        Execute options trades based on strategy
        """
        try:
            logger.info(f"\nTrading options for {date}")
            
            # Calculate total portfolio value
            total_value = cash
            for symbol, shares in holdings['equities'].items():
                try:
                    price = price_data[symbol].loc[
                        price_data[symbol]['date'] == date, 'close'
                    ].iloc[0]
                    total_value += shares * price
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {str(e)}")
                    continue

            # Get options allocation
            options_allocation = allocation_ratios.get('options', 0.10)
            max_options_value = total_value * options_allocation

            # Calculate current options exposure
            current_options_value = sum(
                pos.get('market_value', 0)
                for pos in holdings.get('options', {}).values()
            )

            logger.info(f"Portfolio Value: ${total_value:,.2f}")
            logger.info(f"Options Allocation: {options_allocation:.1%}")
            logger.info(f"Max Options Value: ${max_options_value:,.2f}")
            logger.info(f"Current Options Value: ${current_options_value:,.2f}")

            if current_options_value >= max_options_value:
                logger.info("Already at maximum options allocation")
                return holdings, cash, []

            # Calculate available allocation
            available_allocation = min(
                max_options_value - current_options_value,
                cash * 0.5
            )

            executed_trades = []
            market_condition = self._assess_market_condition(price_data, date)

            for symbol, shares in holdings['equities'].items():
                if symbol not in options_data or shares < 100:
                    continue

                try:
                    current_price = price_data[symbol].loc[
                        price_data[symbol]['date'] == date, 'close'
                    ].iloc[0]

                    opportunities = self.find_option_opportunities(
                        options_data[symbol],
                        current_price,
                        market_condition
                    )

                    for opp in opportunities[:3]:  # Consider top 3 opportunities
                        contract_cost = opp['market_price'] * 100
                        total_cost = contract_cost * (1 + transaction_cost)

                        if total_cost <= available_allocation:
                            trade = {
                                'date': date,
                                'symbol': symbol,
                                'type': opp['type'],
                                'strike': opp['strike'],
                                'expiration': opp['expiration'],
                                'contracts': 1,
                                'price': opp['market_price'],
                                'total_cost': total_cost,
                                'transaction_cost': total_cost * transaction_cost,
                                'expected_profit': opp['score'] * contract_cost * 0.1,
                                'greeks': opp['greeks']
                            }

                            executed_trades.append(trade)
                            
                            if trackers:
                                self._record_trade(trade, trackers)

                            # Update holdings and allocation
                            option_key = f"{symbol}_{opp['type']}_{opp['strike']}_{opp['expiration']}"
                            if 'options' not in holdings:
                                holdings['options'] = {}

                            holdings['options'][option_key] = {
                                'symbol': symbol,
                                'type': opp['type'],
                                'strike': opp['strike'],
                                'expiration': opp['expiration'],
                                'contracts': 1,
                                'entry_price': opp['market_price'],
                                'market_value': contract_cost,
                                'expected_profit': opp['score'] * contract_cost * 0.1,
                                'greeks': opp['greeks']
                            }

                            cash += (contract_cost - total_cost * transaction_cost)
                            available_allocation -= total_cost

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue

            return holdings, cash, executed_trades

        except Exception as e:
            logger.error(f"Error in options trading: {str(e)}")
            return holdings, cash, []

    def _assess_market_condition(self, 
                               price_data: Dict,
                               date: datetime,
                               lookback_days: int = 20) -> str:
        """
        Assess market condition for options trading
        """
        try:
            # Use SPY as market proxy if available
            if 'SPY' in price_data:
                market_data = price_data['SPY']
                market_data = market_data[market_data['date'] <= date].tail(lookback_days)
                
                if len(market_data) >= lookback_days:
                    returns = market_data['close'].pct_change()
                    volatility = returns.std() * np.sqrt(252)
                    
                    if volatility > 0.25:
                        return 'high_volatility'
                    elif volatility < 0.15:
                        return 'low_volatility'
                    
            return 'normal'
            
        except Exception as e:
            logger.error(f"Error assessing market condition: {str(e)}")
            return 'normal'

    def _record_trade(self, trade: Dict, trackers: Dict):
        """Record option trade in tracking system"""
        try:
            trackers['trade'].record_trade(
                date=trade['date'],
                action=f"sell_{trade['type']}",
                symbol=trade['symbol'],
                quantity=trade['contracts'],
                price=trade['price'],
                value=trade['total_cost'],
                cost=trade['transaction_cost'],
                expected_profit=trade['expected_profit']
            )
            
            trackers['transactions'].append(trade)
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
