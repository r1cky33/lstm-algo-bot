#from tensorflow.keras.models import Sequential
import tensorflow.python.keras.models
from sklearn.metrics import precision_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout

import numpy as np
from helper.data_processor import data_processor

def get_sequences(features, targets, train_samples, sequence_length=80):
    X = []
    y = []

    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    train_samples -= sequence_length
    return X[:train_samples], y[:train_samples], X[train_samples:], y[train_samples:]

def train_and_save_model(X_train, y_train):
    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(25, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='tanh'),
        Dense(1, activation='sigmoid')  # Use 'linear' for regression tasks
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    model.save('lstm_model.h5')
    return model

def get_precision_score(preds, confidence_treshhold=0.12):
    predicted_classes = (preds > confidence_treshhold).astype(int)
    precision = precision_score(y_test, predicted_classes)
    return precision, predicted_classes

if __name__ == "__main__":
    print("[i] TensorFlow version: ", tf.__version__)

    # prepare data
    processor = data_processor('../data/btc_usdt_hourly_filled.json')

    processor.load_data()
    processor.calculate_indicators()

    processor.compute_pct_changes()
    processor.replace_inf_values()
    processor.prep_target_for_model()

    features_scaled, targets = processor.scale_features()
    processor.save_data('../data/pct_change.csv')

    # get sequences
    X_train, y_train, X_test, y_test = get_sequences(features_scaled, targets, 43000)

    # train and save
    model = train_and_save_model(X_train, y_train)

    # predict
    preds = model.predict(X_test)
    precision_score, predicted_classes = get_precision_score(preds, confidence_treshhold=0.10)
    print(f"[+] precision_score: {precision_score}")

    high_confidence_preds = [1 if p > 0.10 else 0 for p in preds.flatten()]