import requests
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
# Load the model and scaler once at startup
loaded_model = joblib.load('svm_water_classification_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

@app.get("/predict")
async def predict_water_classification():
    # URL to fetch the data from ThingSpeak
    url = "https://api.thingspeak.com/channels/2689751/feeds.json?api_key=7PFAGFIDJUH5G5JM&results=2"
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        data = response.json()

        # Extract feeds (the actual data)
        feeds = data['feeds']
        results = []

        # Iterate over the feeds
        for feed in feeds:
            entry_id = feed['entry_id']
            created_at = feed['created_at']
            temperature = float(feed.get('field1', 'N/A'))  # Temperature from field1
            moisture = float(feed.get('field2', 'N/A'))      # Moisture from field2
            
            # Prepare input for prediction
            input_data = np.array([[temperature, moisture]])
            input_scaled = loaded_scaler.transform(input_data)

            # Predict the class for the single input
            predicted_class = loaded_model.predict(input_scaled)[0]

            # Determine water needs based on predicted class
            if predicted_class == "Low":
                water = 80
            elif predicted_class == "Medium":
                water = 60
            else:
                water = 0
            
            results.append({
                "entry_id": entry_id,
                "created_at": created_at,
                "temperature": temperature,
                "moisture": moisture,
                "predicted_class": predicted_class,
                "water_needed": water
            })

        return JSONResponse(content={"status": "success", "data": results}, status_code=200)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from ThingSpeak: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# To run the application, use the command:
# uvicorn app:app --reload
