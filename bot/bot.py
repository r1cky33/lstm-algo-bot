from helper.data_processor import data_processor

if __name__ == "__main__":
    processor = data_processor('../data/btc_usdt_hourly_filled.json')

    processor.load_data()
    processor.calculate_indicators()

    processor.compute_pct_changes()
    processor.replace_inf_values()
    processor.prepare_data_for_model()
    processor.scale_features()
    processor.save_data('output.csv')
