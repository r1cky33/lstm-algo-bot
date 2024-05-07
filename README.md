First attempt in implementing a more or less simple LSTM model for forecasting BTCUSD price action.

The dataset consists of OHLCV candlestick data and several calculated technical indicators. The absolute indicators get relativated in order to not harm the model. (Otherwise it might only trade in price ranges where most of the training happened)
