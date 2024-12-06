# config.py
from datetime import datetime, timedelta
import pandas as pd

# Data parameters
EQUITY_START_DATE = "2020-01-01"
BACKTEST_START_DATE = "2015-01-01"  # Actual backtest start after having enough history
EQUITY_END_DATE = "2024-11-12"
#EQUITY_END_DATE = datetime.now().strftime('%Y-%m-%d')  # Current date or specify
MINIMUM_HISTORY_YEARS = 3  # Need 3 years of history before making predictions
HISTORY_YEARS = 3                 # Years of history needed
DATA_START_DATE = pd.to_datetime(EQUITY_START_DATE) - pd.DateOffset(years=HISTORY_YEARS)

# Date range for options data (fixed to last 2 years)
OPTIONS_START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years ago
OPTIONS_END_DATE = datetime.now().strftime('%Y-%m-%d')


WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 1
SEQUENCE_LENGTH = 40
RISK_FREE_RATE = 0.03
TOP_N = 30
INITIAL_CAPITAL = 2000000
REBALANCE_FREQUENCY = 'Q'
TRANSACTION_COST = 0.001
STOP_LOSS = 0.15  # 15% stop loss
TRAILING_STOP = 0.20
RE_ENTRY_THRESHOLD = 0.03  # 5% re-entry threshold
TOP_N_HOLDINGS = 10  # Limit to top 5 holdings per ETF

MAX_OUT_OF_MARKET_DAYS = 20  # Maximum days to stay out of market
FORCE_REENTRY_RECOVERY = 0.8  # Force re-entry if recovered 80% from bottom

MAX_POSITION_SIZE = 0.07  # 5% maximum position size
MIN_POSITION_SIZE = 0.015  # 2% minimum position
MIN_POSITIONS = 5 # Minimum number of positions
MAX_POSITIONS = 50  # Maximum number of positions
MIN_EXPECTED_RETURN = 0.02  # 5% minimum expected return threshold
REQUIRED_COVERAGE_RATIO = 0.5         # Minimum ratio of symbols with sufficient history

MAX_SECTOR_ALLOCATION = 0.25  # 40% maximum sector allocation
VOLATILITY_THRESHOLD = 0.40  # 40% maximum annualized volatility
MOMENTUM_WINDOW = 120  # 60-day momentum lookback
COVARIANCE_SHRINKAGE = 0.40  # 30% shrinkage to diagonal matrix

VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5

CACHE_SECTOR_INFO = True      # Cache sector info to reduce API calls

VOLATILITY_LOOKBACK = 30  # Days to calculate volatility
TARGET_PORTFOLIO_VOLATILITY = 0.18  # 15% annual volatility target
MAX_LEVERAGE = 1.0  # No leverage
MIN_HOLDING_PERIOD = 20  # Minimum days to hold a position


ALLOCATION_RATIOS = {'equities': 0.90, 'options': 0.10}  # 90% equities, 10% options

MAX_DRAWDOWN = 0.30           # 25% maximum drawdown
MAX_VOLATILITY = 0.35        # 30% annualized volatility
VOL_WINDOW = 20               # 20-day window for volatility calculation
RISK_CHECK_FREQUENCY = 'D'    # Daily risk checks
FORCE_LIQUIDATION = False   # Whether to force liquidation on risk limit breach
MIN_RECOVERY = 0.02           # 5% minimum recovery before re-entry
MAX_CONSECUTIVE_LOSSES = 5    # Maximum consecutive losing days
MAX_RECOVERY_DAYS = 15       # Maximum days to wait for recovery
FORCE_REENTRY_RECOVERY = 0.03  # Re-enter after 5% recovery from bottom

MIN_TRADING_CAPITAL = 0.85    # Keep trading if above 90% of initial capital

# Path to the output file for ETF weights after each rebalancing
ETF_OUTPUT_FILE = 'etf_weights_rebalancing.csv'

OPTION_FEATURES = ['moneyness', 'timeToExpiration', 'impliedVolatility', 'optionType']
OPTION_TARGET = 'lastPrice'


# Features and target for the LSTM and GCN models
FEATURES = [
    'equity_return',
    'equity_ma_5',
    'equity_ma_10',
    'volume',
    'volume_ma_5',
    'volume_ma_10',
    'rsi_14',
    'macd',
    'macd_signal',
    'bollinger_mavg',
    'bollinger_hband',
    'bollinger_lband',
    # Add any additional features you are calculating in `calculate_features`
]

TARGET = 'equity_return'  # Or another target variable suitable for your model

#Treasury data settings
TREASURY_USE_INTERPOLATION = False  # Whether to use interpolation or simple matching
TREASURY_UPDATE_FREQUENCY = 'D'     # How often to fetch new rates ('D' for daily)
TREASURY_CACHE_FILE = 'treasury_rates.csv'  # Where to cache treasury rates

# If treasury fetch fails, use these fallback rates
TREASURY_FALLBACK_RATES = {
    '1M': 0.03,
    '3M': 0.035,
    '6M': 0.04,
    '1Y': 0.045,
    '2Y': 0.05,
    '3Y': 0.055,
    '5Y': 0.06,
    '7Y': 0.065,
    '10Y': 0.07,
    '20Y': 0.075,
    '30Y': 0.08
}


ETF_FEATURES = [
    'dollar_volume',
    'spread',
    'illiquidity',
    'momentum_1m',
    'volume_stability'
]

PREDICTION_THRESHOLD = 0.5  # Base threshold for ETF allocation adjustment
# API keys
ALPHAVANTAGE_API_KEY = "80HLC8UDC38Z6HVE"

