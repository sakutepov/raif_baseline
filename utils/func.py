import re
import typing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1


def deviation_metric_vec(y_true: np.array, y_pred: np.array) -> float:
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)

    metr = deviation * 0.0 + 9

    metr[np.abs(deviation) <= THRESHOLD] = 0

    metr[deviation <= - 4 * THRESHOLD] = 9 * NEGATIVE_WEIGHT

    mask = (-4 * THRESHOLD < deviation) & (deviation < -THRESHOLD)
    metr[mask] = NEGATIVE_WEIGHT * ((deviation[mask] / THRESHOLD) + 1) ** 2

    mask = (deviation < 4 * THRESHOLD) & (deviation > THRESHOLD)
    metr[mask] = ((deviation[mask] / THRESHOLD) - 1) ** 2

    return metr.mean()


def floor_process(series):
    for i, s in enumerate(series):
        print(i, end='\r')
        if pd.isna(s):
            continue
        if type(s) in [float, int]:
            series[i] = int(s)
        else:
            if '.' in s:
                try:
                    series[i] = int(s.split('.')[0])
                except:
                    series[i] = s.lower()
            else:
                try:
                    series[i] = int(s)
                except:
                    series[i] = s.lower()

    floor_category = series.fillna('unknown')

    for i, s in enumerate(floor_category):
        if s != 'unknown':
            if type(s) is int:
                if s <= 0:
                    floor_category[i] = 'basement'
                elif s == 1:
                    floor_category[i] = 'first_floor'
                else:
                    floor_category[i] = 'higher_floor'
            elif ',' in s or bool(re.search('\d-\d', s)):
                floor_category[i] = 'multy_level'
            elif bool(re.search('(цоколь)|(подва[л])', s)):
                floor_category[i] = 'basement'
            else:
                f = re.findall('\d', s)
                if not f:
                    floor_category[i] = 'higher_floor'
                else:
                    f = int(f[0])
                    if f <= 0:
                        floor_category[i] = 'basement'
                    elif f == 1:
                        floor_category[i] = 'first_floor'
                    else:
                        floor_category[i] = 'higher_floor'
    return floor_category.replace('unknown', np.nan)


def create_extra_features(df):
    df['floor'] = floor_process(df['floor'])
    return df


def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    return np.median(np.abs(y_pred - y_true) / y_true)


def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric_vec(y_true, y_pred)
    return {'mape': mape, 'mdape': mdape, 'rmse': rmse, 'r2': r2, 'raif_metric': raif_metric}
