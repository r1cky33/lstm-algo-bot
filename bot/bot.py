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


if __name__ == "__main__":

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
    lol = 1