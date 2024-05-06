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

        for sequence_index in self.trade_indices:
            prediction_index = sequence_index + self.sequence_length
            data_sequence = self.dataset.iloc[sequence_index:prediction_index]
            last_close_price = data_sequence['close'].iloc[-1]

            if prediction_index < last_fail + 15 or prediction_index < last_pos_end_index:
                continue

            atr = data_sequence['atr'].iloc[-1]
            tp_price = last_close_price + atr * 2 * 1.5
            sl_price = last_close_price - atr * 2

            forecast_times = self.dataset.index[sequence_index + 1: sequence_index + 1 + self.forecast_candle_len]
            min_close_forecast = self.dataset.loc[forecast_times, 'close'].min()
            max_close_forecast = self.dataset.loc[forecast_times, 'close'].max()

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

            candles_to_end = last_pos_end_index - prediction_index + 1
            df_extended = self.dataset.iloc[sequence_index:prediction_index + candles_to_end]
            fig, axes = mpf.plot(df_extended, type='candle', style='charles', volume=True,
                                 title=f'Sequence {sequence_index} with Prediction', figsize=(20, 6), returnfig=True)

            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [last_close_price, last_close_price], color='orange', linewidth=2)
            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [tp_price, tp_price], color='green', linewidth=2)
            axes[0].plot([self.sequence_length, self.sequence_length + candles_to_end], [sl_price, sl_price], color='red', linewidth=2)

            plt.savefig(os.path.join(self.prediction_directory, f'prediction_{sequence_index}.png'))
            plt.close(fig)
