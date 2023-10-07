from sklearn.ensemble import RandomForestClassifier
import logging
import pickle
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobServiceClient

class CustomData:
    def __init__(self, promoted: int, review: float, projects: int, salary: int, 
                 tenure: float, satisfaction: float, bonus: int, avg_hrs_month: float):
        self.promoted = promoted
        self.review = review
        self.projecs = projects
        self.salary = salary
        self.tenure = tenure
        self.satisfaction = satisfaction
        self.bonus = bonus
        self.avg_hrs_month = avg_hrs_month

    def get_data_as_data_frame(self):
        input_dict = {
            "promoted": [self.promoted],
            "review": [self.review],
            "projects": [self.projecs],
            "salary": [self.salary],
            "tenure": [self.tenure],
            "satisfaction": [self.satisfaction],
            "bonus": [self.bonus],
            "avg_hrs_month": [self.avg_hrs_month]
        }
        return pd.DataFrame(input_dict)

def get_pickle_models(storage_url, access_key, container_name):
    # Create a BlobServiceClient using the storage account URL and key
    blob_service_client = BlobServiceClient(account_url=storage_url, credential=access_key)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    blobs = container_client.list_blobs()

    # Create a list to store pickled models
    pickles = dict()

    # Download each blob to the local directory
    for blob in blobs:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
        blob_data = blob_client.download_blob()
        pickle_item = pickle.loads(blob_data.readall())
        pickles[blob.name] = pickle_item
    return pickles


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get Inputs
    try:
        columns = req.get_json()

        data = CustomData(
            promoted = int(columns.get('promoted')),
            review = float(columns.get('review')),
            projects = int(columns.get('projects')),
            salary = int(columns.get('salary')),
            tenure = float(columns.get('tenure')),
            satisfaction = float(columns.get('satisfaction')),
            bonus = int(columns.get('bonus')),
            avg_hrs_month = float(columns.get('avg_hrs_month')),
        )

        pred_df = data.get_data_as_data_frame()
        
    except ValueError:
        return func.HttpResponse("Invalid request format.", status_code=400)

    # Load Pickles
    try:
        STORAGE_URL = "https://azureturnover.blob.core.windows.net/"
        ACCESS_KEY = "5B6//F2xo0oIH4TINjMOVUvOluPmmRAJ6+p3ma+6qByhpTGBtWuNTMlifKSQm+fnVHDu0HDJOrm3+AStPEgDkw=="
        CONTAINER_NAME = "pickle-models"
        pickle_models = get_pickle_models(storage_url=STORAGE_URL, access_key=ACCESS_KEY, container_name=CONTAINER_NAME)
        model = pickle_models['RFC_Model.sav']
        scaler = pickle_models['StandardScaler.sav']
    except Exception as e:
        return func.HttpResponse(f"Problem at reading pickled models {e}", status_code=400)
    
    try:
        scaled_data = scaler.transform(pred_df)
        prediction = model.predict(scaled_data)
        if prediction[0].tolist() == 0:
            return func.HttpResponse("The Employee will stay")
        elif prediction[0].tolist() == 1:
            return func.HttpResponse("The Employee will left")
    except Exception as e:
        func.HttpResponse(f"Bro, thats not working!!! The error: {e}")
