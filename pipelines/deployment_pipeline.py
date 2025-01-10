import pickle
import json
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'trained_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(data):
    try:
        data = json.loads(data)
        data = np.array(data['data'])
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})