from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import get_prediction

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    result = get_prediction(data.text)
    result = "Positive" if result == 1 else "Negative"
    return {"sentiment": result}
