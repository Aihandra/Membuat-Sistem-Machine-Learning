from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import uvicorn
import json

app = FastAPI()

model_path = r"C:\Users\ADVAN\Downloads\MLProject\mlruns\492799901531716253\d594a7f01ccc4266854f8207108d2ec3\artifacts\model"
model = mlflow.pyfunc.load_model(model_path)

@app.post("/v2/models/mlflow-model/infer")
async def infer(request: Request):
    body = await request.json()
    data = body["inputs"]
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
