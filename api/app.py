from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import traceback
from fingerprint_enhancer import enhance_Fingerprint
from pathlib import Path
import numpy as np

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://fingers-app.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to process the image
def process_image(img_array: np.ndarray) -> str:
    try:
        # Flip the image horizontally
        img = cv2.flip(img_array, 1)

        # Enhance fingerprint
        out = enhance_Fingerprint(img)

        # Encode to JPEG and convert to base64
        _, img_encoded = cv2.imencode('.jpeg', out)

        # Convert the NumPy array (img_encoded) to bytes
        img_bytes = img_encoded.tobytes()

        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# API endpoint to upload an image and get processed fingerprint
@app.post("/process-fingerprint")
async def process_fingerprint(file: UploadFile = File(...)):
    try:
        # Read the file as bytes
        file_bytes = await file.read()

        # Convert bytes to numpy array
        np_arr = np.frombuffer(file_bytes, np.uint8)

        # Decode the image (assuming the file is an image)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode the image. Please upload a valid image.")

        # Process the image and return base64 string
        processed_image = process_image(img)
        
        return JSONResponse(content={"processedImage": processed_image})
    except Exception as e:
        traceback.print_exc()
        return HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
        

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
# Run the server using uvicorn (for testing purposes)
# You can run the app with: uvicorn your_filename:app --reload
