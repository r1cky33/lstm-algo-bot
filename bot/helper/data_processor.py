import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

class data_processor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pct_data = None
        self.original_data = None

    def load_data(self):
        self.original_data = pd.read_json(self.filepath)
        self.original_data['timestamp'] = pd.to_datetime(self.original_data['timestamp'])
        self.original_data.set_index('timestamp', inplace=True)

    def compute_pct_changes(self):
        self.pct_data = self.original_data.copy()

        exclude_columns = ['rsi', 'atr', 'BBP_20_2.0', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
        pct_change_columns = self.pct_data.columns.difference(exclude_columns)

        self.pct_data[pct_change_columns] = self.pct_data[pct_change_columns].pct_change()
        self.pct_data.dropna(inplace=True)
        self.original_data = self.original_data.loc[self.pct_data.index]

    def replace_inf_values(self):
        self.pct_data['volume'].replace([np.inf, -np.inf], 0, inplace=True)

    def calculate_indicators(self):
        self.original_data['rsi'] = ta.rsi(self.original_data.close, length=16)
        self.original_data['atr'] = ta.atr(self.original_data['high'], self.original_data['low'], self.original_data['close'], length=14)

        bollinger = ta.bbands(self.original_data['close'], length=20, std=2)
        self.original_data = pd.concat([self.original_data, bollinger], axis=1)

        self.original_data['obv'] = ta.obv(self.original_data['close'], self.original_data['volume'])
        macd = self.original_data.ta.macd(close='close', fast=12, slow=26, signal=9)
        self.original_data = self.original_data.join(macd)

        self.original_data['sma_5'] = ta.sma(self.original_data['close'], length=5)
        self.original_data['sma_60'] = ta.sma(self.original_data['close'], length=60)
        self.original_data['sma_250'] = ta.sma(self.original_data['close'], length=250)

        self.clean_bb_columns(self.original_data)

    def clean_bb_columns(self, dataset):
        columns_to_delete = ['BBM_20_2.0', 'BBB_20_2.0']
        dataset.drop(columns=columns_to_delete, inplace=True)
        dataset.dropna(inplace=True)
        #dataset = self.dataset.loc[self.pct_data.index]

    def prep_target_for_model(self):
        self.pct_data['target'] = 0

        print('[+] calculating targets in dataset... this might take a moment')

        if len(self.pct_data) != len(self.original_data):
            raise Exception('[!] len() mismatch from pct_data to original_data!')

        for i in range(len(self.pct_data) - 13):
            current_time = self.pct_data.index[i]
            next_13_times = self.pct_data.index[i + 1: i + 14]

            if len(next_13_times) < 13:
                break

            current_close_price = self.original_data.loc[current_time, 'close']
            current_atr = self.original_data.loc[current_time, 'atr']

            min_close_next_13 = self.original_data.loc[next_13_times, 'close'].min()
            max_close_next_13 = self.original_data.loc[next_13_times, 'close'].max()

            if min_close_next_13 > current_close_price - (current_atr * 1.5) and max_close_next_13 > current_close_price + (current_atr * 2):
                self.pct_data.at[current_time, 'target'] = 1

    def save_data(self, filename="../data/pct_data.csv"):
        self.pct_data.to_csv(filename)

    def scale_features(self):
        features_to_scale = [col for col in self.pct_data.columns if 'target' not in col]

        features = self.pct_data[features_to_scale].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        self.pct_data[features_to_scale] = features_scaled
        return features_scaled, self.pct_data['target'].values


# if __name__ == "__main__":
#     processor = BTCUSDDataProcessor('/home/richarch/PycharmProjects/ccxt_fetcher/btc_usdt_hourly_filled.json')
#     processor.load_data()
#     processor.compute_pct_changes()
#     processor.replace_inf_values()
#     processor.calculate_indicators()
#     processor.prepare_data_for_model()
#     processor.scale_features()
#     processor.save_data()
