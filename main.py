from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from test import predict_from_bytes

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

@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get plant disease prediction
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {allowed_types}"
        )
    
    # Read image bytes
    contents = await file.read()
    
    # Get prediction from model
    prediction = predict_from_bytes(contents)
    
    return {
        "prediction": prediction
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
