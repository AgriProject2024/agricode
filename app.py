import requests
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from pydantic import BaseModel
# Database setup
def create_user_table():
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Create the users table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            contact TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Call the function to create the table
create_user_table()

class UserRegister(BaseModel):
    username: str
    email: str  
    password: str
    contact: str  


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

loaded_model = joblib.load('svm_water_classification_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')
class UserLogin(BaseModel):
    email: str
    password: str
    

@app.post("/login")
async def login_user(user: UserLogin):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT username, email, contact FROM users WHERE email = ? AND password = ?
        ''', (user.email, user.password))
        
        user_data = cursor.fetchone()
        conn.close()

        if user_data:
            # Return user details with a 200 status code
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "data": {
                        "username": user_data[0],
                        "email": user_data[1],
                        "contact": user_data[2]
                    }
                }
            )
        else:
            # Return empty if user not found
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/register")
async def register_user(user: UserRegister):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, email, password, contact)
            VALUES (?, ?, ?, ?)
        ''', (user.username, user.email, user.password, user.contact))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/predict")
async def predict_water_classification():
    url = "https://api.thingspeak.com/channels/2689751/feeds.json?api_key=7PFAGFIDJUH5G5JM&results=2"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        feeds = data['feeds']
        results = []
        for feed in feeds:
            entry_id = feed['entry_id']
            created_at = feed['created_at']
            temperature = float(feed.get('field1', 'N/A'))  
            moisture = float(feed.get('field2', 'N/A'))      
            input_data = np.array([[temperature, moisture]])
            input_scaled = loaded_scaler.transform(input_data)
            predicted_class = loaded_model.predict(input_scaled)[0]
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


# uvicorn app:app --reload --host 0.0.0.0 --port 8000

