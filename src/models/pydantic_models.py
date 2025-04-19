from pydantic import BaseModel


class FitParams(BaseModel):
    is_refresh: bool = False
