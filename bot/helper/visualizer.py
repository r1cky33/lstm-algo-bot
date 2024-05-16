import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt


class visualizer:
    def __init__(self, trade_indices, dataset, sequence_length, forecast_candle_len=13):
        self.trade_indices = trade_indices
        self.dataset = dataset
        self.prediction_directory='predictions'
        self.sequence_length = sequence_length
        self.forecast_candle_len = forecast_candle_len

    def generate_trade_imgs(self):
        last_fail = 0
        loosers = 0
        winners = 0
        last_pos_end_index = 0

        for index in self.trade_indices:
            prediction_index = index + self.sequence_length
            data_sequence = self.dataset.iloc[index:prediction_index]
            last_close_price = data_sequence['close'].iloc[-1]

            if prediction_index < last_fail + 15 or prediction_index < last_pos_end_index:
                continue

            atr = data_sequence['atr'].iloc[-1]
            tp_price = last_close_price + atr * 2 * 1.5
            sl_price = last_close_price - atr * 2

            forecast_times = self.dataset.index[index + 1: index + 1 + self.forecast_candle_len]
            min_forecast = self.dataset.loc[forecast_times, 'low'].min()
            max_forecast = self.dataset.loc[forecast_times, 'high'].max()

            hit = "None"
            for i in range(prediction_index + 1, len(self.dataset)):
                last_pos_end_index = i
                if self.dataset.iloc[i]['low'] <= sl_price:
                    last_fail = i
                    hit = "SL"
                    loosers += 1
                    break
                elif self.dataset.iloc[i]['high'] >= tp_price:
                    hit = "TP"
                    winners += 1
                    break

            candles_to_end = last_pos_end_index - prediction_index
            df_extended = self.dataset.iloc[index:prediction_index + candles_to_end]

            if prediction_index + 1 + self.forecast_candle_len >= len(self.dataset):
                continue  # Skip this iteration if the forecast would go beyond the available data

            # Ensure we don't try to access beyond the dataset
            end_index = min(len(self.dataset), prediction_index + candles_to_end + 1)
            df_extended = self.dataset.iloc[index:end_index]
            if df_extended.empty:
                print('skipping for safety')
                continue

            fig, axes = mpf.plot(df_extended, type='candle', style='charles', volume=True,
                                 title=f'Sequence {index} with Prediction', figsize=(20, 6), returnfig=True)

            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [last_close_price, last_close_price], color='orange', linewidth=2)
            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [tp_price, tp_price], color='green', linewidth=2)
            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [sl_price, sl_price], color='red', linewidth=2)

            os.makedirs(self.prediction_directory, exist_ok=True)
            plt.savefig(os.path.join(self.prediction_directory, f'prediction_{index}.png'))
            plt.close(fig)

        print(f"Loosers: {loosers}")
        print(f"Winners: {winners}")

    def plot_signal_distribution(self):
        self.dataset.plot.line(y="close", use_index=True, legend=True)

        x_values = [self.dataset.index[i + self.sequence_length] for i in self.trade_indices]
        y_values = [self.dataset.iloc[i + self.sequence_length]["close"] for i in self.trade_indices]

        plt.scatter(x_values, y_values, color='red', s=10)
        plt.savefig(os.path.join(self.prediction_directory, 'trade_distribution.png'))
        plt.show()
