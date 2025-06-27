from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
from io import BytesIO
from app.model_utils import predict
import os

app = FastAPI()

# CORS (optional, can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Default route - serve index.html
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Prediction route
@app.post("/predict")
async def detect_vehicle(file: UploadFile = File(...)):
    print(1)  

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    result = predict(image)

    return result
