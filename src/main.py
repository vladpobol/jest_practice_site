from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from models.flower_classifier import FlowerClassifier
import shutil
from pathlib import Path

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize the model
MODEL_PATH = "src/flowers_efficientv2_acc_0.97.pth"
classifier = FlowerClassifier(MODEL_PATH)

# Mount static files
app.mount("/site", StaticFiles(directory="site"), name="site")

@app.get("/")
async def read_root():
    return FileResponse("site/index.html")

@app.get("/{path:path}")
async def read_static(path: str):
    return FileResponse(f"site/{path}")

@app.post("/classify")
async def classify_flower(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = UPLOAD_DIR / file.filename
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Get predictions
        results = classifier.predict(str(temp_path))
        return results
    finally:
        # Clean up the temporary file
        temp_path.unlink()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)