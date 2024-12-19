from os import walk
from functools import reduce
import pandas as pd
import numpy as np
import datetime as dt


def fill_column(df):
    df = df.copy()
    non_nans = df[~df["close"].apply(np.isnan)]
    start, end = non_nans.index[0], non_nans.index[-1]
    df[start:end] = df[start:end].ffill()
    return df


def main():
    hd_path = "data/historical/"

    # idx = pd.date_range("2018-01-01", "2024-11-08", freq="d")
    # ts = pd.Series(range(len(idx)), index=idx)
    # df = pd.DataFrame(data={"date": ts})
    # df.index = pd.DatetimeIndex(df["date"]) # no hacer
    # df.to_csv("data/dates_range.csv")
    # df1 = pd.read_table("data/dates_range.csv", sep=",")
    # df1 = df1.rename(columns={"date": "date_old", "Unnamed: 0": "date"})

    dates_df = pd.read_table("data/dates_range.csv", sep=",")
    dates_df = dates_df.rename(columns={"date": "date_old", "Unnamed: 0": "date"})

    empty_symbols = []

    def generate_df_prices():
        symbols = list(next(walk(hd_path), (None, None, []))[2])
        for symbol in symbols:
            print(symbol)
            try:
                df = pd.read_table(hd_path + symbol, sep=",")
            except pd.errors.EmptyDataError:
                empty_symbols.append(symbol)
            else:
                df = pd.merge(dates_df, df, on=["date"], how="outer")
                # df = fill_column(df)
                df = df.rename(columns={"close": symbol})[[symbol, "date"]]
                yield df

    dfs = generate_df_prices()

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=["date"], how="outer"), dfs
    )

    df_merged = df_merged.rename(columns={"date": "Date"})
    nan_value = float("NaN")

    # remove columns if their last row is equal to null
    df_merged = df_merged.loc[:, ~df_merged.iloc[-1].isna()]
    df_merged.to_csv("data/symbols_hd_prices.csv", index=False)

    print(df_merged.isnull().values.sum())  # 144351
    print(empty_symbols)


if __name__ == "__main__":
    main()
