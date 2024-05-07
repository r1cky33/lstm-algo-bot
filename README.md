### Machine Learning for BTCUSD forecasting ###
First attempt in implementing a more or less simple LSTM model for forecasting BTCUSD price action. The model is trained with sequences (length=144) and a binary target set to "1", if the following price saw a increase (e. g. last `close` + ATR (period=14) * 2).

The dataset consists of OHLCV candlestick data and several calculated technical indicators. The absolute indicators get relativated in order to not harm the model. (Otherwise it might only trade in price ranges where most of the training happened)

### Signal distribution ###
![image](https://github.com/r1cky33/lstm-algo-bot/assets/49491951/f09317d3-37f4-45b1-8053-b5dd97cc8962)

### Example trade ###
![image](https://github.com/r1cky33/lstm-algo-bot/assets/49491951/41eda0b3-d956-492e-8eee-df49ce4d6564)

### Returns ###
![image](https://github.com/r1cky33/lstm-algo-bot/assets/49491951/03596178-af12-4565-98db-55f58a1b459d)
~ 54% winrate mit 1.5:1 R/R-Ration. 
