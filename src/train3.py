import mlflow
import mlflow.pyfunc
import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


experiment_name = "Titanic_Model_Serving_Demo1_26aug"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name)

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Concatenate target and features for AutoGluon
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

target = "Survived"
features = [col for col in train_df.columns if col != target]

class AutoGluonWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from autogluon.tabular import TabularPredictor
        self.model = TabularPredictor.load(context.artifacts["autogluon_model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

with mlflow.start_run() as run:
    # Train AutoGluon predictor
    predictor = TabularPredictor(label=target, eval_metric='accuracy',path="models/autogluon_model").fit(
        train_data=train_df, 
        time_limit=120,  # seconds
        presets='best_quality'
    )
    # Save model locally first,bundle of all trained models, not a single estimator.a pointer to the best model
    predictor.save("models/autogluon_model") #AutoGluon saves:All candidate models,Their metadata + leaderboard,A pointer to the best model to use for prediction
    print(f"✅ AutoGluon model saved locally at 'models/autogluon_model'")
    
    # Make predictions
    preds = predictor.predict(test_df[features]) # Predictions (default = best model)
    y_true = y_test.values.ravel()

 
    # Evaluate
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    print(f"✅ AutoGluon Predictor trained")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    all_models = predictor.model_names()
    # best_model = predictor.best_model()
    print(f"📂 Models trained: {all_models}")
    # print(f"⭐ Best model selected: {best_model}")
    leaderboard = predictor.leaderboard(silent=True)
    print("Leaderboard for all models metrics data=\n",leaderboard)


    # Log parameters & metrics
    mlflow.log_param("AutoGluon_time_limit", 120)
    mlflow.log_param("AutoGluon_presets", "best_quality")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
       
      
    cm = confusion_matrix(y_true, preds)
    print("confusion_matrix ************",cm) 
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    if hasattr(predictor, "coef_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.coef_[0]
        })
        fi_path = "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
 
    print("Log model ************")    #log all models LightGBM, CatBoost, RandomForest, NeuralNet, ensembles, etc
    mlflow.pyfunc.log_model(
        artifact_path="autogluon_model",
        python_model=AutoGluonWrapper(),
        artifacts={"autogluon_model": "models/autogluon_model"},
        registered_model_name=experiment_name    )
    
