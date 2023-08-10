from typing import Optional , Text
from fastapi import FastAPI  
from src.stages.test_model import Test
import yaml

#get configurations
with open('params.yaml') as obj :
    configs = yaml.safe_load(obj)

# create FastAPI app and load model
app = FastAPI()  
model = Test(configs)

# create an endpoint that receives POST requests
# and returns predictions
@app.post("/predict/")
def predict(features : Text):  
    predictions = model.predict(features)
    return predictions