from bisect import bisect_left

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(0)


class Cell:
    def __init__(self, id_, lat, lon,
                 size_lat, size_lon):
        self.id = id_
        self.latitude = lat
        self.longitude = lon
        self.size_lat = size_lat
        self.size_lon = size_lon
        self.coord = [[lon, lat],
                      [lon + size_lon, lat],
                      [lon + size_lon, lat + size_lat],
                      [lon, lat + size_lat]]

    def to_dict(self):
        return {
            'id': self.id,
            'lat': self.latitude,
            'lon': self.longitude,
            'coord': self.coord,
        }


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def find_closest_coords(df_, size_lat, size_lon):
    df = df_.copy()

    latitudes = 55.38 + size_lat * np.array(range(1, 71))
    longitudes = 37.04 + size_lon * np.array(range(1, 65))

    closest = []
    for lat in df["lat"]:
        closest.append(take_closest(latitudes, lat))
    df['lat_closest'] = closest

    closest = []
    for lon in df["lon"]:
        closest.append(take_closest(longitudes, lon))
    df['lon_closest'] = closest

    return df


def create_cells(df_, size_lat, size_lon):
    df = df_.copy()

    cells = [Cell(i, lat, lon, size_lat, size_lon) for i, (lat, lon) in enumerate(
        [(lat, lon) for lat, lon in zip(df.lat_closest, df.lon_closest)]
    )]

    df['coord'] = [c.to_dict()['coord'] for c in cells]

    return df


def process(df_, size_lat, size_lon):
    df = df_.copy()
    df.rename(columns={"lng": "lon"}, inplace=True)
    df = find_closest_coords(df, size_lat, size_lon)
    df = create_cells(df, size_lat, size_lon)
    oe = OrdinalEncoder()
    df["coord_str"] = df["coord"].apply(str)
    df["coord_idx"] = oe.fit_transform(
        df["coord_str"].values.reshape(-1, 1))

    df.drop(columns={"coord_str"}, inplace=True)
    return df


def create_cell_train_n_test(_train_df, _test_df,
                             size_lat=0.009, size_lon=0.016,
                             target="per_square_meter_price"):
    train_df = _train_df.copy()
    test_df = _test_df.copy()
    train_df["train"] = 1
    test_df["train"] = 0

    merged = pd.concat((train_df[test_df.columns], test_df))

    merged = process(merged, size_lat, size_lon)

    train_fin_df = merged[merged.train == 1]
    train_fin_df[target] = train_df[target]

    return train_fin_df.drop(columns={"train"}), merged[merged.train == 0].drop(columns={"train"})
