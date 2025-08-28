import requests

url = "http://127.0.0.1:8000/predict"

data = [
    {"Pclass":3,"Sex":"female","Age":28,"SibSp":0,"Parch":0,"Fare":7.225,"Embarked":"C"},
    {"Pclass":1,"Sex":"female","Age":26,"SibSp":0,"Parch":0,"Fare":7.925,"Embarked":"S"}
]

response = requests.post(url, json=data)
print(response.json())
