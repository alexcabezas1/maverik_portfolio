from pydantic import BaseModel


class Portfolio(BaseModel):
    assets: list[str]
    weights: list[float]
    commulative_return: list[float]
