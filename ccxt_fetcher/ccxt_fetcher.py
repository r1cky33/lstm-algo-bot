import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def fetch_single_candle(exchange, symbol, timeframe, timestamp):
    max_retries = 3  # Anzahl der maximalen Wiederholungsversuche
    for attempt in range(max_retries):
        try:
            candle = exchange.fetch_ohlcv(symbol, timeframe, since=timestamp, limit=1)
            if candle:
                return candle
        except ccxt.BaseError as e:
            print(f"Error fetching data for {datetime.utcfromtimestamp(timestamp / 1000)}: {str(e)}")
            time.sleep(5)  # Wartezeit zwischen Versuchen
    return None

def check_and_fill_missing_candles(df, exchange, symbol, timeframe):
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)

    # Sicherstellen, dass jede Stunde vertreten ist
    all_hours = df.resample('1h').asfreq()

    # Identifiziere fehlende Zeitstempel
    missing_times = all_hours[all_hours['open'].isna()].index

    # Fehlende Daten abrufen und auffüllen
    for missing_time in missing_times:
        print(f"Missing data at {missing_time}")  # Log fehlende Stunden
        timestamp = int(missing_time.timestamp()) * 1000
        fetched_candle = fetch_single_candle(exchange, symbol, timeframe, timestamp)
        if fetched_candle:
            candle_df = pd.DataFrame(fetched_candle, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            candle_df['datetime'] = pd.to_datetime(candle_df['timestamp'], unit='ms')
            candle_df.set_index('datetime', inplace=True)
            all_hours.loc[missing_time] = candle_df.iloc[0]
            print(f"Inserted missing data for {missing_time}")  # Log erfolgreich eingefügte Stunden

    return all_hours

def fetch_historical_data(symbol, timeframe, since, exchange):
    all_candles = []
    end_time = datetime.utcnow()  # Beendet die Schleife beim aktuellen Datum und Uhrzeit
    since = exchange.parse8601(since)

    while since < end_time.timestamp() * 1000:
        print(f"Fetching data starting from {datetime.utcfromtimestamp(since / 1000)}")
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=500)
            if not candles:
                print("No more data returned by the exchange.")
                break
            since = candles[-1][0] + 60 * 60 * 1000
            all_candles.extend(candles)
            time.sleep(exchange.rateLimit / 1000)  # Beachtung der Ratenbegrenzung
        except ccxt.BaseError as e:
            print(f"An error occurred: {str(e)}")
            break

    return all_candles

def save_to_json(data, filename, exchange, symbol, timeframe):
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_filled = check_and_fill_missing_candles(df, exchange, symbol, timeframe)

    # Zurückkonvertieren in das ursprüngliche Format und in JSON speichern
    df_filled.reset_index(inplace=True)
    df_filled['timestamp'] = df_filled['datetime'].astype(int) // 10 ** 6
    result_json = df_filled[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')

    with open(filename, 'w') as file:
        json.dump(result_json, file)

# Setup and initial data fetch
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True
})
symbol = 'BTC/USDT'
timeframe = '1h'
years = 1
since = (datetime.utcnow() - pd.DateOffset(years=years)).isoformat()

historical_data = fetch_historical_data(symbol, timeframe, since, exchange)
if historical_data:
    save_to_json(historical_data, '../data/btc_usdt_hourly_filled.json', exchange, symbol, timeframe)
    print("Data fetch complete and saved to 'btc_usdt_hourly_filled.json'.")
else:
    print("No data fetched. Please check the input parameters and network connection.")
