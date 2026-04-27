import pandas as pd
import numpy as np


MODEL_FEATURE_COLUMNS = [
    'time_to_expiry',
    'distance_to_strike',
    'obi',
    'log_return_10s',
    'volatility_10s',
    'log_return_30s',
    'volatility_30s',
    'log_return_60s',
    'volatility_60s',
    'log_return_300s',
    'volatility_300s',
]


class FeatureEngineer:
    def __init__(self):
        self.price_history = []
        
    def compute_obi(self, bids, asks):
        """Order Book Imbalance (OBI) based on top 20 levels"""
        if not bids or not asks:
            return 0.0
        bid_vol = sum(v for p, v in bids.items())
        ask_vol = sum(v for p, v in asks.items())
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

    def extract_realtime_features(self, current_time, btc_price, S0, bids, asks, p_mkt, history_df, round_expiry=None):
        """
        Gera o vetor de features para inferência no Forward Test.
        """
        features = {}
        
        # 1. Time-to-Expiry (T) em segundos
        if round_expiry is None:
            # Fallback local. No forward test real, prefira o endDate vindo do Gamma.
            next_expiry = current_time.replace(second=0, microsecond=0)
            minutes_to_add = 5 - (current_time.minute % 5)
            next_expiry = next_expiry + pd.Timedelta(minutes=minutes_to_add)
            if next_expiry == current_time:
                next_expiry = next_expiry + pd.Timedelta(minutes=5)
        else:
            next_expiry = round_expiry

        features['time_to_expiry'] = max(0.0, (next_expiry - current_time).total_seconds())
        
        # 2. Distance-to-Strike (Δ)
        if S0 is None:
            features['distance_to_strike'] = 0.0
        else:
            features['distance_to_strike'] = (btc_price - S0) / S0
            
        # 3. Microstructure: OBI
        features['obi'] = self.compute_obi(bids, asks)
        
        # 4. Implied Probability (P_mkt)
        features['implied_prob'] = p_mkt
        
        # 5. Log-Returns & Realized Volatility
        windows = [10, 30, 60, 300]
        for w in windows:
            if len(history_df) >= w:
                past_price = history_df['price'].iloc[-w]
                ret = np.log(btc_price / past_price)
                features[f'log_return_{w}s'] = ret
                
                # Volatilidade: desvio padrão dos retornos segundo a segundo na janela
                recent_returns = np.log(history_df['price'].tail(w) / history_df['price'].tail(w).shift(1)).dropna()
                if len(recent_returns) > 1:
                    vol = recent_returns.std() * np.sqrt(w)
                else:
                    vol = 0.0
                features[f'volatility_{w}s'] = vol
            else:
                features[f'log_return_{w}s'] = 0.0
                features[f'volatility_{w}s'] = 0.0
                
        return pd.Series(features)
        
    def build_backtest_features(self, df, strike_series, expiry_series):
        """
        Processamento vetorizado para Walk-Forward Backtest
        df: dataframe de ticks agrupados por segundo
        strike_series: Pandas Series indicando o S0 vigente em cada segundo
        expiry_series: Pandas Series com o datetime da expiração da rodada atual
        """
        df = df.copy()
        
        df['time_to_expiry'] = (expiry_series - df.index).dt.total_seconds()
        df['distance_to_strike'] = (df['price'] - strike_series) / strike_series
        df['obi'] = (df['bid_vol'] - df['ask_vol']) / (df['bid_vol'] + df['ask_vol'] + 1e-8)
        df['implied_prob'] = df['polymarket_price']
        
        windows = [10, 30, 60, 300]
        for w in windows:
            df[f'log_return_{w}s'] = np.log(df['price'] / df['price'].shift(w))
            ret_1s = np.log(df['price'] / df['price'].shift(1))
            df[f'volatility_{w}s'] = ret_1s.rolling(window=w).std() * np.sqrt(w)
            
        return df.dropna()
