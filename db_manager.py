# db_manager.py
import sqlite3
import pandas as pd

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path='symbol_data.db'):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            # Batch processing table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS batch_runs (
                    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT,
                    end_date TEXT,
                    processed_count INTEGER,
                    selected_count INTEGER,
                    batch_score REAL,
                    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Symbol tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS symbol_tracking (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT,
                    last_batch_id INTEGER,
                    times_selected INTEGER DEFAULT 0,
                    current_score REAL,
                    last_processed TIMESTAMP,
                    FOREIGN KEY(last_batch_id) REFERENCES batch_runs(batch_id)
                )
            ''')

            # Price data table
            conn.execute('''
                 symbol TEXT,
                    date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    "Adj Close" REAL, 
                    Volume REAL,
                    FOREIGN KEY(symbol) REFERENCES symbol_tracking(symbol)
                )
            ''')

            # Options data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS options_data (
                    symbol TEXT,
                    expiration_date TEXT,
                    strike REAL,
                    option_type TEXT,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    FOREIGN KEY(symbol) REFERENCES symbol_tracking(symbol)
                )
            ''')

    def start_batch(self, start_date, end_date):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO batch_runs (start_date, end_date, processed_count)
                VALUES (?, ?, 0)
                RETURNING batch_id
            ''', (start_date, end_date))
            return cursor.fetchone()[0]

    def store_price_data(self, symbol_data, batch_id):
        with sqlite3.connect(self.db_path) as conn:
            for symbol, df in symbol_data.items():
                df['symbol'] = symbol
                df.to_sql('price_data', conn, if_exists='append', index=False)
                
                conn.execute('''
                    INSERT OR REPLACE INTO symbol_tracking (symbol, last_batch_id, last_processed)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (symbol, batch_id))

    def store_options_data(self, options_data):
        with sqlite3.connect(self.db_path) as conn:
            for symbol, df in options_data.items():
                df['symbol'] = symbol
                df.to_sql('options_data', conn, if_exists='append', index=False)

    def get_batch_data(self, batch_id):
        query = '''
            SELECT p.*, s.sector 
            FROM price_data p
            JOIN symbol_tracking s ON p.symbol = s.symbol
            WHERE s.last_batch_id = ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=[batch_id])

    def update_batch_results(self, batch_id, selected_symbols, batch_score):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE batch_runs 
                SET selected_count = ?, batch_score = ?
                WHERE batch_id = ?
            ''', (len(selected_symbols), batch_score, batch_id))
            
            for symbol in selected_symbols:
                conn.execute('''
                    UPDATE symbol_tracking 
                    SET times_selected = times_selected + 1
                    WHERE symbol = ?
                ''', (symbol,))

    def get_selected_symbols(self, min_selections=1):
        query = '''
            SELECT symbol, sector, times_selected, current_score
            FROM symbol_tracking
            WHERE times_selected >= ?
            ORDER BY current_score DESC
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=[min_selections])
        
    
class ETFDatabaseManager:
    """Enhanced database manager for ETF data management"""
    
    def __init__(self, db_path: str = 'etf_data.db'):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Setup enhanced database schema for ETF management"""
        with sqlite3.connect(self.db_path) as conn:
            # ETF Constituents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS etf_constituents (
                    symbol TEXT,
                    date TEXT,
                    weight REAL,
                    sector TEXT,
                    market_cap REAL,
                    score REAL,
                    inclusion_date TEXT,
                    removal_date TEXT,
                    reason TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # Sector Allocation table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sector_allocation (
                    date TEXT,
                    sector TEXT,
                    weight REAL,
                    num_constituents INTEGER,
                    avg_market_cap REAL,
                    target_weight REAL,
                    tracking_error REAL,
                    PRIMARY KEY (date, sector)
                )
            ''')

            # Price Data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    "Adj Close" REAL,
                    Volume REAL,
                    market_cap REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # ETF Analytics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS etf_analytics (
                    date TEXT PRIMARY KEY,
                    nav REAL,
                    tracking_error REAL,
                    sector_deviation REAL,
                    turnover REAL,
                    liquidity_score REAL,
                    concentration_score REAL,
                    rebalance_flag INTEGER,
                    num_constituents INTEGER
                )
            ''')

            # Rebalancing Events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS rebalancing_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    type TEXT,
                    symbols_added TEXT,
                    symbols_removed TEXT,
                    total_turnover REAL,
                    sector_impact TEXT,
                    reason TEXT
                )
            ''')

            # Risk Events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    type TEXT,
                    severity TEXT,
                    metrics TEXT,
                    description TEXT,
                    action_taken TEXT
                )
            ''')

            logger.info("Database schema setup complete")

    def store_etf_constituents(self, constituents: pd.DataFrame):
        """Store ETF constituent data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                constituents.to_sql('etf_constituents', conn, 
                                  if_exists='append', index=False)
                
            logger.info(f"Stored {len(constituents)} constituent records")
            
        except Exception as e:
            logger.error(f"Error storing constituents: {str(e)}")

    def store_sector_allocation(self, allocations: Dict[str, Dict]):
        """Store sector allocation data"""
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            records = []
            for sector, data in allocations.items():
                record = {
                    'date': current_date,
                    'sector': sector,
                    'weight': data.get('weight', 0),
                    'num_constituents': data.get('num_constituents', 0),
                    'avg_market_cap': data.get('avg_market_cap', 0),
                    'target_weight': data.get('target_weight', 0),
                    'tracking_error': data.get('tracking_error', 0)
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql('sector_allocation', conn, 
                         if_exists='append', index=False)
                         
            logger.info(f"Stored sector allocation data for {len(records)} sectors")
            
        except Exception as e:
            logger.error(f"Error storing sector allocation: {str(e)}")

    def store_price_data(self, price_data: Dict[str, pd.DataFrame], batch_id: Optional[int] = None):
        """Store price data with market cap"""
        try:
            total_records = 0
            
            with sqlite3.connect(self.db_path) as conn:
                for symbol, df in price_data.items():
                    if df is None or df.empty:
                        continue
                        
                    df = df.copy()  # Avoid modifying original
                    df['symbol'] = symbol
                    
                    # Ensure proper date format
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    
                    df.to_sql('price_data', conn, 
                             if_exists='append', index=False)
                             
                    total_records += len(df)
            
            logger.info(f"Stored {total_records} price records for {len(price_data)} symbols")
            
            if batch_id:
                self._update_batch_stats(batch_id, total_records)
                
        except Exception as e:
            logger.error(f"Error storing price data: {str(e)}")

    def store_etf_analytics(self, analytics: Dict[str, float]):
        """Store ETF analytics data"""
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO etf_analytics (
                        date, nav, tracking_error, sector_deviation,
                        turnover, liquidity_score, concentration_score,
                        rebalance_flag, num_constituents
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_date,
                    analytics.get('nav', 0),
                    analytics.get('tracking_error', 0),
                    analytics.get('sector_deviation', 0),
                    analytics.get('turnover', 0),
                    analytics.get('liquidity_score', 0),
                    analytics.get('concentration_score', 0),
                    analytics.get('rebalance_flag', 0),
                    analytics.get('num_constituents', 0)
                ))
                
            logger.info(f"Stored ETF analytics for {current_date}")
            
        except Exception as e:
            logger.error(f"Error storing ETF analytics: {str(e)}")

    def record_rebalancing_event(self, 
                               event_type: str,
                               symbols_added: List[str],
                               symbols_removed: List[str],
                               turnover: float,
                               sector_impact: Dict[str, float],
                               reason: str):
        """Record rebalancing event details"""
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO rebalancing_events (
                        date, type, symbols_added, symbols_removed,
                        total_turnover, sector_impact, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_date,
                    event_type,
                    ','.join(symbols_added),
                    ','.join(symbols_removed),
                    turnover,
                    str(sector_impact),
                    reason
                ))
                
            logger.info(f"Recorded rebalancing event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error recording rebalancing event: {str(e)}")

    def record_risk_event(self, 
                         event_type: str,
                         severity: str,
                         metrics: Dict[str, float],
                         description: str,
                         action_taken: str):
        """Record risk event with details"""
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO risk_events (
                        date, type, severity, metrics,
                        description, action_taken
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    current_date,
                    event_type,
                    severity,
                    str(metrics),
                    description,
                    action_taken
                ))
                
            logger.info(f"Recorded risk event: {event_type} ({severity})")
            
        except Exception as e:
            logger.error(f"Error recording risk event: {str(e)}")

    def get_etf_composition(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get ETF composition for a given date"""
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
                
            query = '''
                SELECT c.*, p.close, p.volume, p.market_cap
                FROM etf_constituents c
                LEFT JOIN price_data p 
                    ON c.symbol = p.symbol 
                    AND c.date = p.date
                WHERE c.date = ?
                AND c.removal_date IS NULL
            '''
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=[date])
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting ETF composition: {str(e)}")
            return pd.DataFrame()

    def get_sector_history(self, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get sector allocation history"""
        try:
            query = '''
                SELECT *
                FROM sector_allocation
                WHERE 1=1
            '''
            
            params = []
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
                
            query += ' ORDER BY date, sector'
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting sector history: {str(e)}")
            return pd.DataFrame()

    def get_etf_analytics(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Get ETF analytics history"""
        try:
            query = '''
                SELECT *
                FROM etf_analytics
                WHERE 1=1
            '''
            
            params = []
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
                
            query += ' ORDER BY date'
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting ETF analytics: {str(e)}")
            return pd.DataFrame()

    def get_rebalancing_history(self, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """Get rebalancing event history"""
        try:
            query = '''
                SELECT *
                FROM rebalancing_events
                WHERE 1=1
            '''
            
            params = []
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
                
            query += ' ORDER BY date DESC'
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting rebalancing history: {str(e)}")
            return pd.DataFrame()

    def get_risk_events(self,
                       event_type: Optional[str] = None,
                       min_severity: Optional[str] = None,
                       limit: int = 100) -> pd.DataFrame:
        """Get risk event history with filtering"""
        try:
            query = '''
                SELECT *
                FROM risk_events
                WHERE 1=1
            '''
            
            params = []
            if event_type:
                query += ' AND type = ?'
                params.append(event_type)
            if min_severity:
                query += ' AND severity >= ?'
                params.append(min_severity)
                
            query += ' ORDER BY date DESC LIMIT ?'
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting risk events: {str(e)}")
            return pd.DataFrame()

    def _update_batch_stats(self, batch_id: int, records_processed: int):
        """Update batch processing statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE batch_runs
                    SET processed_count = ?
                    WHERE batch_id = ?
                ''', (records_processed, batch_id))
                
        except Exception as e:
            logger.error(f"Error updating batch stats: {str(e)}")

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data while maintaining essential history"""
        try:
            cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete old price data
                conn.execute('''
                    DELETE FROM price_data
                    WHERE date < ?
                ''', (cutoff_date,))
                
                # Delete old analytics
                conn.execute('''
                    DELETE FROM etf_analytics
                    WHERE date < ?
                ''', (cutoff_date,))
                
                # Keep all constituent history
                
                logger.info(f"Cleaned up data older than {cutoff_date}")
                
        except Exception as e:
           
            logger.error(f"Error cleaning up old data: {str(e)}")

    def export_etf_summary(self, output_path: str):
        """Export ETF summary report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current composition
                current_composition = self.get_etf_composition()
                
                # Get latest analytics
                analytics = pd.read_sql('''
                    SELECT *
                    FROM etf_analytics
                    ORDER BY date DESC
                    LIMIT 1
                ''', conn)
                
                # Get sector allocations
                sector_alloc = pd.read_sql('''
                    SELECT sector, weight, num_constituents
                    FROM sector_allocation
                    ORDER BY date DESC
                ''', conn)
                
                # Get recent risk events
                risk_events = pd.read_sql('''
                    SELECT date, type, severity, description
                    FROM risk_events
                    ORDER BY date DESC
                    LIMIT 10
                ''', conn)
                
                # Create summary report
                with pd.ExcelWriter(output_path) as writer:
                    current_composition.to_excel(writer, sheet_name='Composition', index=False)
                    analytics.to_excel(writer, sheet_name='Analytics', index=False)
                    sector_alloc.to_excel(writer, sheet_name='Sectors', index=False)
                    risk_events.to_excel(writer, sheet_name='Recent Risks', index=False)
                
                logger.info(f"ETF summary exported to {output_path}")
                
        except Exception as e:
            logger.error(f"Error exporting ETF summary: {str(e)}")

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive ETF performance metrics"""
        try:
            metrics = {}
            
            with sqlite3.connect(self.db_path) as conn:
                # Get tracking error history
                tracking_errors = pd.read_sql('''
                    SELECT date, tracking_error
                    FROM etf_analytics
                    ORDER BY date
                ''', conn)
                
                if not tracking_errors.empty:
                    metrics['avg_tracking_error'] = tracking_errors['tracking_error'].mean()
                    metrics['max_tracking_error'] = tracking_errors['tracking_error'].max()
                
                # Get turnover metrics
                turnover = pd.read_sql('''
                    SELECT date, total_turnover
                    FROM rebalancing_events
                    ORDER BY date
                ''', conn)
                
                if not turnover.empty:
                    metrics['total_turnover'] = turnover['total_turnover'].sum()
                    metrics['avg_turnover'] = turnover['total_turnover'].mean()
                
                # Get composition stability
                composition_changes = pd.read_sql('''
                    SELECT COUNT(*) as changes
                    FROM rebalancing_events
                    WHERE date >= date('now', '-30 days')
                ''', conn)
                
                metrics['recent_changes'] = int(composition_changes['changes'].iloc[0])
                
                # Get sector deviation
                sector_dev = pd.read_sql('''
                    SELECT MAX(sector_deviation) as max_dev
                    FROM etf_analytics
                    WHERE date >= date('now', '-30 days')
                ''', conn)
                
                metrics['max_sector_deviation'] = float(sector_dev['max_dev'].iloc[0])
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

    def analyze_constituent_stability(self) -> pd.DataFrame:
        """Analyze constituent stability over time"""
        try:
            query = '''
                WITH constituent_history AS (
                    SELECT 
                        symbol,
                        MIN(inclusion_date) as first_inclusion,
                        MAX(CASE WHEN removal_date IS NULL THEN date 
                             ELSE removal_date END) as last_date,
                        COUNT(DISTINCT inclusion_date) as num_inclusions,
                        COUNT(DISTINCT removal_date) as num_removals
                    FROM etf_constituents
                    GROUP BY symbol
                )
                SELECT
                    h.*,
                    s.sector,
                    JULIANDAY(h.last_date) - JULIANDAY(h.first_inclusion) as days_tracked,
                    CAST(num_removals AS FLOAT) / 
                        (JULIANDAY(h.last_date) - JULIANDAY(h.first_inclusion)) * 365 
                        as annual_turnover_rate
                FROM constituent_history h
                LEFT JOIN sector_allocation s ON h.symbol = s.symbol
                WHERE h.last_date >= date('now', '-365 days')
                ORDER BY annual_turnover_rate DESC
            '''
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn)
                
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing constituent stability: {str(e)}")
            return pd.DataFrame()

    def get_sector_transitions(self, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """Analyze sector weight transitions"""
        try:
            query = '''
                SELECT 
                    a1.sector,
                    a1.date as start_date,
                    a2.date as end_date,
                    a1.weight as start_weight,
                    a2.weight as end_weight,
                    a2.weight - a1.weight as weight_change,
                    a1.num_constituents as start_constituents,
                    a2.num_constituents as end_constituents
                FROM sector_allocation a1
                JOIN sector_allocation a2 
                    ON a1.sector = a2.sector
                    AND a2.date > a1.date
                WHERE 1=1
            '''
            
            params = []
            if start_date:
                query += ' AND a1.date >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND a2.date <= ?'
                params.append(end_date)
                
            query += ' ORDER BY a1.sector, a1.date'
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting sector transitions: {str(e)}")
            return pd.DataFrame()