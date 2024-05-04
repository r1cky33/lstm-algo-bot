import unittest
import pandas as pd
import json

class TestCandlestickDataFromFile(unittest.TestCase):
    def setUp(self):
        with open('../../data/btc_usdt_hourly_filled.json', 'r') as file:
            data = json.load(file)

        self.df = pd.DataFrame(data)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)

    def test_no_gaps_in_candlestick_data(self):
        expected_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='h')
        missing_times = expected_range.difference(self.df.index)
        self.assertTrue(missing_times.empty, f"Es fehlen Daten für folgende Zeiten: {missing_times}")

if __name__ == '__main__':
    unittest.main()
