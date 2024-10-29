import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import joblib
from typing import List
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from maverik_portfolio import schemas

from fastapi import FastAPI

app = FastAPI()
models = {
    "min_risk": None,
    "max_return": None,
    "max_ratio": None,
}
models_path = "maverik_portfolio/models"


def load_model():
    model_names = list(models.keys())

    for name in model_names:
        models[name] = joblib.load("{}/{}_model".format(models_path, name))


def generate(model_name: str) -> List[float]:
    model = models[model_name]
    print(model)
    prices = load_sp500_dataset()

    X = prices_to_returns(prices)

    weights = [float(e) for e in model.weights_]

    return schemas.Portfolio(
        assets=[prices.columns], weights=weights, commulative_return=[]
    )


# pool = ProcessPoolExecutor(max_workers=1, initializer=load_model)
load_model()


@app.post("/portfolio/generate/{model_name}", response_model=schemas.Portfolio)
async def portfolio_generate(model_name: str):
    loop = asyncio.get_event_loop()

    # portfolio = await loop.run_in_executor(pool, partial(generate, model_name))

    return partial(generate, model_name)()
