import sys
sys.path.append("/Users/zhanghangsheng/others_code/time-series/darts")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import FFT
from darts.metrics import mae
from darts.datasets import TemperatureDataset
from darts.utils.missing_values import fill_missing_values

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

ts = TemperatureDataset().load()

# time_=ts.univariate_values()
# print(time_.shape)
# fft_values = np.fft.fft(time_)

train, val = ts.split_after(pd.Timestamp("19850701"))
train = fill_missing_values(train)
z=train.univariate_values()
fft_values = np.fft.fft(z)

fft_=TimeSeries.from_dataframe(pd.DataFrame(fft_values))
fft_.plot(label="ttt")

train.plot(label="train")
# fft_values.plot(label="val")

model = FFT(required_matches=set(), nr_freqs_to_keep=None)
model.fit(train)
pred_val = model.predict(len(val))
pred_val.plot(label="pred_val")
print(pred_val)

# train.plot(label="train")
# val.plot(label="val")
# pred_val.plot(label="prediction")
print("MAE:", mae(pred_val, val))