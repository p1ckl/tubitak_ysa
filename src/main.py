import pandas
from os import getcwd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


data = str(getcwd()) + "/data/Bias_correction_ucl.csv" 

ds = pandas.read_csv(data).dropna()

train_data = ds.sample(frac = 0.8, random_state = 1)
test_data = ds.drop(train_data.index)

min_temp_train_label = train_data.copy().pop("Next_Tmin")
max_temp_train_label = train_data.copy().drop("Next_Tmax")

min_temp_test_label = test_data.copy().drop("Next_Tmin")
max_temp_test_label = test_data.copy().drop("Next_Tmax")


normalizer = tf.keras.layers.Normalization(axis = -1)
normalizer.adapt(np.array(train_data.copy()))

min_temp_model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation = "relu"),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1)
    ])

max_temp_model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation = "relu"),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1)
    ])


min_temp_model.compile(loss = "mean_absolute_error", optimizer = keras.optimizers.Adam(0.001))
max_temp_model.compile(loss = "mean_absolute_error", optimizer = keras.optimizers.Adam(0.001))
