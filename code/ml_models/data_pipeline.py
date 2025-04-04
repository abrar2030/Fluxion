import pandas as pd
from sklearn.preprocessing import RobustScaler

class DataPipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.features = [
            'volume', 'price', 'liquidity', 
            'volatility', 'arbitrage_opportunities'
        ]

    def transform(self, raw_data):
        df = pd.DataFrame(raw_data)
        
        # Advanced feature engineering
        df['price_velocity'] = df['price'].pct_change()
        df['liquidity_zscore'] = (df['liquidity'] - df['liquidity'].mean())/df['liquidity'].std()
        
        # Handle outliers
        df = df[(df['volume'] > 0) & (df['price'] > 0)]
        
        # Scale features
        scaled = self.scaler.fit_transform(df[self.features])
        return pd.DataFrame(scaled, columns=self.features)