from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from src.inference.inference_main import predict

app = FastAPI()


class Features(BaseModel):
    values: list


@app.post("/predict")
def predict_endpoint(payload: Features):
    try:
        result = predict(payload.values)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"prediction": result}
