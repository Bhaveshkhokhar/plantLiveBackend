from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/image")
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
