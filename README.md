

---



\## ⚙️ Setup Instructions



\### 1. Clone the repository

```bash

git clone https://github.com/ch24m544/titanic-mlops-ch24m544.git

cd titanic-mlops-ch24m544

pip install -r requirements.txt

python src/preprocess\_spark1.py

python -m mlflow server --backend-store-uri sqlite:///mlflow.db \\

&nbsp;   --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000



python src/train3.py

python src/app.py

uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

python src/test.py

python src/drift.py

python src/retrain\_pipeline.py
python src/autorun_pipeline.py


docker build -t titanic-mlops .

docker run -p 8000:8000 titanic-mlops



