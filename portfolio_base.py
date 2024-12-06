# portfolio_base.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class PortfolioState:
   """Base portfolio state class"""
   def __init__(self, initial_capital: float, combined_prices: pd.DataFrame):
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
           'options_enabled': False,
           'in_recovery': False
       }

       self.peak_value = initial_capital
       logger.info(f"Initialized portfolio with ${initial_capital:,.2f}")
       
   @property
   def total_value(self) -> float:
       """Get total portfolio value"""
       return self._state['total_value']
       
   @property
   def holdings(self) -> Dict:
       """Get holdings"""
       return self._state['holdings']
       
   @property
   def cash(self) -> float:
       """Get cash"""
       return self._state['cash']
       
   @property
   def portfolio_value(self) -> pd.Series:
       """Get portfolio value series"""
       return self._state['portfolio_value']
       
   @property
   def combined_prices(self) -> pd.DataFrame:
       """Get combined prices"""
       return self._state['combined_prices']

   def __getattr__(self, name):
       """Fallback for getting attributes from _state"""
       if name in self._state:
           return self._state[name]
       raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

   def update(self, date: datetime, combined_prices: pd.DataFrame):
       """Update portfolio state with current values"""
       try:
           # Calculate equity value
           equity_value = sum(
               shares * float(combined_prices.loc[date, symbol])
               for symbol, shares in self._state['holdings']['equities'].items()
           )
           
           # Calculate options value 
           options_value = sum(
               pos.get('market_value', 0)
               for pos in self._state['holdings']['options'].values()
           )
           
           # Update state values
           self._state['equity_value'] = equity_value
           self._state['options_value'] = options_value
           self._state['total_value'] = self._state['cash'] + equity_value + options_value
           
           # Update portfolio value series
           if not isinstance(self._state['portfolio_value'], pd.Series):
               self._state['portfolio_value'] = pd.Series(dtype=float)
           self._state['portfolio_value'].loc[date] = self._state['total_value']
           
           self._state['previous_value'] = self._state['total_value']
           
       except Exception as e:
           logger.error(f"Error updating state: {str(e)}")

   def __repr__(self):
       return f"PortfolioState(total_value=${self._state['total_value']:,.2f}, holdings={len(self._state['holdings']['equities'])} equities)"