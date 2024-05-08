from collections import Counter

from keras.src.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
from sklearn.metrics import precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout

import tensorflow as tf
import numpy as np

from helper.data_processor import data_processor
from helper.visualizer import visualizer

MODEL_PATH='../data/lstm_model.h5'

def get_sequences(dataset, targets, train_samples_count, sequence_length=144):
    X = []
    y = []

    for i in range(len(dataset) - sequence_length):
        X.append(dataset[i:i + sequence_length])
        y.append(targets[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    #train_samples_count -= sequence_length
    return X[:train_samples_count], y[:train_samples_count], X[train_samples_count:], y[train_samples_count:], train_samples_count

def train_and_save_model(X_train, y_train, model_path):
    model = Sequential([
        LSTM(100, activation='tanh',return_sequences=True, kernel_regularizer='l2', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(50, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')  # Use 'linear' for regression tasks
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=12, batch_size=32, verbose=1, callbacks=[tensorboard])
    model.save(model_path)
    return model

def get_precision_score(preds, confidence_threshold=0.12):
    predicted_classes = (preds > confidence_threshold).astype(int)
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

    dataset_scaled, targets = processor.scale_features()
    processor.save_data('../data/pct_change.csv')

    # get sequences
    sequence_length=80
    train_sample_count = 49000
    X_train, y_train, X_test, y_test, train_sample_count = get_sequences(dataset_scaled, targets, train_sample_count, sequence_length=sequence_length)

    # train and save
    model = train_and_save_model(X_train, y_train, model_path=MODEL_PATH)
    #model = load_model(MODEL_PATH)

    # predict
    preds = model.predict(X_test)

    confidence_threshold = np.percentile(preds, 90)
    precision_score, predicted_classes = get_precision_score(preds, confidence_threshold=confidence_threshold)
    high_confidence_indices = [i for i, confidence in enumerate(preds) if confidence > confidence_threshold]
    print(f"[+] precision_score: {precision_score} with confidence_threshold: {confidence_threshold}")

    high_confidence_preds = [1 if p > confidence_threshold else 0 for p in preds.flatten()]
    prediction_counts = Counter(high_confidence_preds)

    print(f"Number of 0s (Below Threshold or Predicting Lower): {prediction_counts[0]}")
    print(f"Number of 1s (Above Threshold and Predicting Higher): {prediction_counts[1]}")

    # visualisation
    visualizer = visualizer(high_confidence_indices, processor.original_data.iloc[train_sample_count:], sequence_length=sequence_length, forecast_candle_len=13)
    visualizer.generate_trade_imgs()
    visualizer.plot_signal_distribution()
    i = 0