from typing import List
from enum import Enum
import datetime as dt
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from pandas import DataFrame
import joblib

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.exceptions import NonPositiveVarianceError
from skfolio.preprocessing import prices_to_returns
from sklearn.model_selection import train_test_split

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maverik_portfolio import schemas


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "min_risk": None,
    "max_return": None,
    "max_ratio": None,
}
models_path = "maverik_portfolio/models"

estimator_class = MeanRisk
estimator_details = {
    "min_risk": {
        "objective_function": ObjectiveFunction.MINIMIZE_RISK,
        # expected loss of a portfolio in the worst case scenarios
        # beyond confidential level, conditional value at risk
        "risk_measure": RiskMeasure.CVAR,
    },
    "max_return": {
        "objective_function": ObjectiveFunction.MAXIMIZE_RETURN,
        "risk_measure": RiskMeasure.VARIANCE,
    },
    "max_sharpe": {
        "objective_function": ObjectiveFunction.MAXIMIZE_RATIO,
        "risk_measure": RiskMeasure.VARIANCE,
    },
}


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BOLD = "bold"


risk_profile_to_estimator_schema = {
    RiskProfile.CONSERVATIVE: "min_risk",
    RiskProfile.MODERATE: "max_sharpe",
    RiskProfile.BOLD: "max_return",
}


def load_model():
    model_names = list(models.keys())

    for name in model_names:
        models[name] = joblib.load("{}/{}_model".format(models_path, name))


def get_stocks_with_prices(number_of_stocks: int = 10) -> DataFrame:
    prices_df = pd.read_table("data/symbols_hd_prices.csv", sep=",")
    prices_df["Date"] = prices_df.apply(
        lambda row: dt.datetime.strptime(row.Date, "%Y-%m-%d"),
        axis=1,
    )
    prices_df["Date"] = pd.to_datetime(prices_df.Date)
    prices_df.index = pd.DatetimeIndex(prices_df.Date)
    prices_df = prices_df.drop("Date", axis=1)
    prices_df = prices_df.sample(n=10, axis="columns")

    return prices_to_returns(prices_df)


def generate(risk_profile: RiskProfile) -> schemas.Portfolio:
    portfolio_type_name = risk_profile_to_estimator_schema[risk_profile]

    mdetails = estimator_details[portfolio_type_name]

    model = estimator_class(
        objective_function=mdetails["objective_function"],
        risk_measure=mdetails["risk_measure"],
        portfolio_params=dict(name=portfolio_type_name),
    )

    X = get_stocks_with_prices()
    X_train, X_test = train_test_split(X, test_size=0.30, shuffle=False)

    try:
        model.fit(X_train)
    except NonPositiveVarianceError:
        X = get_stocks_with_prices()
        X_train, X_test = train_test_split(X, test_size=0.30, shuffle=False)
        model.fit(X_train)

    # portfolio = model.predict(X_test)
    # print((1 + X).cumprod())

    return schemas.Portfolio(
        risk_profile=risk_profile.value,
        assets=X.columns.to_list(),
        weights=[float(w * 100) for w in model.weights_.tolist()],
        commulative_return=[],
    )


# pool = ProcessPoolExecutor(max_workers=1, initializer=load_model)


@app.post("/portfolio/generate/{risk_profile}", response_model=schemas.Portfolio)
async def portfolio_generate(risk_profile: str):
    # loop = asyncio.get_event_loop()
    # portfolio = await loop.run_in_executor(pool, partial(generate, model_name))

    return generate(RiskProfile(risk_profile))
