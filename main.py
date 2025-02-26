# Host our FastAPI app 
from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np

# Load trained model 
model = joblib.load('model/model.pkl')

# Create instance of fastapi 
app = FastAPI()

# Define the request body for input data (features or independent variables) 
class PredictRequest(BaseModel): 
     sepal_length: float 
     sepal_width: float 
     petal_length: float 
     petal_width: float 


# Define prediction endpoint 
@app.post("/predict") # Goes to predict page
def predict(request: PredictRequest): # Assign it the format of request body defined earlier
    data = np.array([[request.sepal_length, 
                      request.sepal_width, 
                      request.petal_width, 
                      request.petal_width]]) 
    prediction = model.predict(data) # Make predictions on our data 
    species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"} # To map numerical class names back to text
    # This is the response body 
    return {"prediction": species_mapping[int(prediction[0])]} # Return the mapped predictions


# Define home 
@app.get("/")
def read_root(): 
     return {"Welcome to the Iris CLassification API"}