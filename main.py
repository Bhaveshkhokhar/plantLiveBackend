from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import uuid
from datetime import datetime

app = FastAPI(
    title="Major Project API",
    description="Backend API for the Major Project App",
    version="1.0.0"
)

# Enable CORS for React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Plant(BaseModel):
    id: Optional[int] = None
    name: str
    species: Optional[str] = None
    health_status: Optional[str] = "healthy"
    image_url: Optional[str] = None

class PlantCheckRequest(BaseModel):
    image_base64: Optional[str] = None
    plant_id: Optional[int] = None

class PlantCheckResponse(BaseModel):
    status: str
    message: str
    health_score: int
    recommendations: List[str]

# In-memory database (replace with real database later)
plants_db: List[Plant] = []
plant_id_counter = 1

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Major Project API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/plants", response_model=List[Plant])
async def get_plants():
    return plants_db

@app.get("/api/plants/{plant_id}", response_model=Plant)
async def get_plant(plant_id: int):
    for plant in plants_db:
        if plant.id == plant_id:
            return plant
    raise HTTPException(status_code=404, detail="Plant not found")

@app.post("/api/plants", response_model=Plant)
async def create_plant(plant: Plant):
    global plant_id_counter
    plant.id = plant_id_counter
    plant_id_counter += 1
    plants_db.append(plant)
    return plant

@app.put("/api/plants/{plant_id}", response_model=Plant)
async def update_plant(plant_id: int, updated_plant: Plant):
    for i, plant in enumerate(plants_db):
        if plant.id == plant_id:
            updated_plant.id = plant_id
            plants_db[i] = updated_plant
            return updated_plant
    raise HTTPException(status_code=404, detail="Plant not found")

@app.delete("/api/plants/{plant_id}")
async def delete_plant(plant_id: int):
    for i, plant in enumerate(plants_db):
        if plant.id == plant_id:
            plants_db.pop(i)
            return {"message": "Plant deleted successfully"}
    raise HTTPException(status_code=404, detail="Plant not found")

@app.post("/api/check-plant", response_model=PlantCheckResponse)
async def check_plant(request: PlantCheckRequest):
    """
    Analyze plant health from image or plant ID
    This is a placeholder - integrate with ML model later
    """
    # Placeholder response - replace with actual ML analysis
    return PlantCheckResponse(
        status="success",
        message="Plant analysis complete",
        health_score=85,
        recommendations=[
            "Water the plant every 2-3 days",
            "Ensure adequate sunlight",
            "Check for pests regularly"
        ]
    )

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file
    Returns the saved file path and metadata
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {allowed_types}"
        )
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return {
        "status": "success",
        "filename": unique_filename,
        "original_filename": file.filename,
        "file_path": file_path,
        "content_type": file.content_type,
        "size": len(contents)
    }

@app.post("/api/analyze-plant-image")
async def analyze_plant_image(file: UploadFile = File(...)):
    """
    Upload a plant image and analyze its health
    Returns analysis results with recommendations
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {allowed_types}"
        )
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"plant_{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # TODO: Add actual ML model analysis here
    # For now, return placeholder analysis
    
    return {
        "status": "success",
        "message": "Plant image analyzed successfully",
        "image": {
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": file_path,
            "size": len(contents)
        },
        "analysis": {
            "plant_name": "Unknown Plant",
            "confidence": 0.85,
            "health_score": 78,
            "health_status": "Good",
            "issues_detected": [
                "Slight yellowing on leaves",
                "Possible overwatering"
            ],
            "recommendations": [
                "Reduce watering frequency",
                "Ensure proper drainage",
                "Place in indirect sunlight",
                "Check soil moisture before watering"
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
