import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Read the data
raw_data = pd.read_csv("data/raw/ratings_Beauty.csv")

# Split the data into train-valid-test split

df_train_valid, df_test = train_test_split(raw_data, test_size=0.2, random_state=1115)
df_train, df_valid = train_test_split(df_train_valid, test_size=0.25, random_state=1115)

# Save the three datasets
df_train.to_csv("data/clean/df_train.csv")
df_valid.to_csv("data/clean/df_valid.csv")
df_test.to_csv("data/clean/df_test.csv")