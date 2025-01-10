import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from zenml.client import Client
# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for Model Evaluation
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, X_test, y_test):
        pass

class ModelEvaluation(ModelEvaluationStrategy):
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, y_pred)
        model_f1 = f1_score(y_test, y_pred, average='weighted')
        # model_roc = roc_auc_score(y_test, y_pred, multi_class='ovr')

        logging.info(f'Accuracy: {model_acc:.2f}')
        logging.info(f'F1 Score: {model_f1:.2f}')
        # logging.info(f'ROC AUC Score: {model_roc:.2f}')
        return {
            "accuracy": model_acc,
            "f1_score": model_f1,
            # "roc_auc_score": model_roc
        }

if __name__ == "__main__":
    pass