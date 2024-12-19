from pydantic import BaseModel


class Portfolio(BaseModel):
    risk_profile: str
    assets: list[str]
    weights: list[float]
    commulative_return: list[float]
