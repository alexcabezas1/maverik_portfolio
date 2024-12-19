import os
from pathlib import Path
import csv
import requests


def create_csv(data, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(data[0].keys()))
        for entry in data:
            writer.writerow(entry.values())


def get_stock_data(symbol, start_date, end_date, api_key):
    url = "https://eodhd.com/api/eod/{}".format(symbol)
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": start_date,
        "to": end_date,
        "period": "d",
        "order": "a",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("API request failed with status code {}".format(response.status_code))
        return None


def main():
    api_key = os.environ.get("EODHD_APIKEY")
    historical_data_path = "data/historical"
    from_date = "2018-01-01"
    to_date = "2024-11-09"

    with open("data/symbols.csv", mode="r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            symbol = row["symbol"]
            filepath = Path(historical_data_path) / symbol
            if filepath.exists():
                continue

            print("getting historial data: symbol={} ...".format(symbol))
            resp = get_stock_data(
                symbol=symbol,
                start_date=from_date,
                end_date=to_date,
                api_key=api_key,
            )
            if resp:
                create_csv(resp, str(filepath))


if __name__ == "__main__":
    main()
