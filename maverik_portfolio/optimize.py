import numpy as np
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
import joblib


if __name__ == "__main__":
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.30, shuffle=False)
    # print(X_train.head())

    min_risk_model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
        portfolio_params=dict(name="Min CVaR"),
    )

    min_risk_model.fit(X_train)
    min_risk_portfolio = min_risk_model.predict(X_test)
    print(type(min_risk_model.weights_), min_risk_model.weights_)
    print(min_risk_portfolio.cumulative_returns)

    max_return_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
        portfolio_params=dict(name="Max Return"),
    )

    max_return_model.fit(X_train)
    max_return_portfolio = max_return_model.predict(X_test)
    print(max_return_model.weights_)
    print(max_return_portfolio.cumulative_returns)

    max_ratio_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
        portfolio_params=dict(name="Max Sharpe"),
    )

    max_ratio_model.fit(X_train)
    max_ratio_portfolio = max_ratio_model.predict(X_test)
    print(max_ratio_model.weights_)
    print(max_ratio_portfolio.cumulative_returns)

    models_path = "maverik_portfolio/models"

    joblib.dump(min_risk_model, "{}/min_risk_model".format(models_path))
    joblib.dump(max_return_model, "{}/max_return_model".format(models_path))
    joblib.dump(max_ratio_model, "{}/max_ratio_model".format(models_path))
