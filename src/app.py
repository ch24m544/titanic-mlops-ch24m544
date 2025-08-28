from fastapi import FastAPI
import mlflow.pyfunc
from typing import List
import pandas as pd
from pydantic import BaseModel  # ✅ This is required

class TitanicRecord(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    
    
app = FastAPI()

# Point to MLflow Tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load from MLflow Model Registry
model_uri = "models:/Titanic_Model_Serving_Demo1_26aug/latest"  # or specify version: "models:/TitanicAutoGluonModel/1"
print(f"🔗 Loading model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)
print("used model====",model)
@app.post("/predict")
def predict(data: List[TitanicRecord]):
    df = pd.DataFrame([record.dict() for record in data])
    pred = model.predict(df)
    return {"prediction": pred.tolist()}


# --- Sample data for testing ---
if __name__ == "__main__":
    import uvicorn

    # Example input to test prediction directly
    sample_data = [
        TitanicRecord(Pclass=3, Sex="male", Age=22, SibSp=1, Parch=0, Fare=7.25, Embarked='S'),
        TitanicRecord(Pclass=1, Sex="female", Age=38, SibSp=1, Parch=0, Fare=71.2833, Embarked='S')
    ]
    
    df_sample = pd.DataFrame([record.dict() for record in sample_data])
    sample_pred = model.predict(df_sample)
    print("Sample Predictions:", sample_pred.tolist())

    # Run FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
