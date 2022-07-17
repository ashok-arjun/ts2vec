import pandas as pd

SOURCE = "./datasets/bahia_windspeed.csv"

TARGET_COL = 'windspeed'
TARGET_FEATURES = "./datasets/bahia_windspeed_feats.csv"
TARGET_WITH_TARGET = "./datasets/ETTm1_with_target.csv"

source_df = pd.read_csv(SOURCE)
target_features_df = source_df.loc[:, source_df.columns.drop([TARGET_COL])]
target_features_df.set_index('date', inplace=True)
target_features_df.to_csv(TARGET_FEATURES)

# target_with_target_df = source_df.filter(['date',TARGET_COL], axis=1)
# target_with_target_df.set_index('date', inplace=True)

# target_with_target_df.to_csv(TARGET_WITH_TARGET)