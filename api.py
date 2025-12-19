"""
FastAPI REST API for Multi-Face Swapping
Provides endpoints to swap up to 4 faces in a single image
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import uuid
import os
import shutil
from pydantic import BaseModel

from face_swapper import MultiFaceSwapper, save_image, enhance_image_for_hijab

# Initialize face swapper (singleton)
face_swapper = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Initialize the face swapper
    try:
        global face_swapper
        if face_swapper is None:
            face_swapper = MultiFaceSwapper(enable_hijab_mode=True)
        print("✓ Face swapper initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize face swapper: {str(e)}")
    
    yield
    
    # Shutdown: cleanup if needed
    print("Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multi-Face Swap API",
    description="API for swapping up to 4 faces simultaneously in an image",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize face swapper (singleton)
face_swapper = None

# Directories for temporary files
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SwapResponse(BaseModel):
    """Response model for face swap operation"""
    success: bool
    message: str
    output_image_url: Optional[str] = None
    swap_info: Optional[dict] = None


def get_face_swapper():
    """Lazy initialization of face swapper with hijab mode enabled"""
    global face_swapper
    if face_swapper is None:
        # Initialize with hijab mode enabled for better detection
        face_swapper = MultiFaceSwapper(enable_hijab_mode=True)
    return face_swapper


def image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """
    Convert uploaded file to numpy array (BGR format)
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Image as numpy array in BGR format
    """
    contents = upload_file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload_file.filename}")
    return image


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Face Swap API",
        "version": "1.0.0",
        "endpoints": {
            "/swap": "POST - Swap faces in an image",
            "/swap-specific": "POST - Swap faces with specific target indices",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        swapper = get_face_swapper()
        return {
            "status": "healthy",
            "face_swapper": "initialized"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/swap", response_model=SwapResponse)
async def swap_faces(
    base_image: UploadFile = File(..., description="Base image containing faces to be replaced"),
    face_1: UploadFile = File(..., description="First source face image"),
    face_2: Optional[UploadFile] = File(None, description="Second source face image (optional)"),
    face_3: Optional[UploadFile] = File(None, description="Third source face image (optional)"),
    face_4: Optional[UploadFile] = File(None, description="Fourth source face image (optional)"),
    enhance_hijab: bool = Form(True, description="Apply preprocessing for images with hijab/headwear (default: True)")
):
    """
    Swap faces in the base image. Faces are swapped from left to right.
    
    - **base_image**: Image with 1-4 faces to be replaced
    - **face_1**: Required first source face
    - **face_2**: Optional second source face (can use hijab)
    - **face_3**: Optional third source face (can use hijab)
    - **face_4**: Optional fourth source face (can use hijab)
    - **enhance_hijab**: Apply preprocessing for hijab images (default: True for better detection)
    
    **Note**: Enhancement is enabled by default for optimal hijab detection.
    Set to false only if you have detection issues with non-hijab images.
    
    Returns the image with swapped faces and information about the swap operation.
    """
    try:
        swapper = get_face_swapper()
        
        print(f"\n{'='*60}")
        print(f"🎭 Starting face swap operation")
        print(f"{'='*60}")
        print(f"Enhancement mode: {'✓ Enabled' if enhance_hijab else '✗ Disabled'}")
        
        # Load base image
        base_img = image_from_upload(base_image)
        
        # Load source face images with optional hijab enhancement
        face_images = [image_from_upload(face_1)]
        
        # Apply enhancement for hijab if requested
        if enhance_hijab:
            print("Applying hijab-friendly preprocessing to source images...")
            face_images[0] = enhance_image_for_hijab(face_images[0])
        
        if face_2:
            img = image_from_upload(face_2)
            if enhance_hijab:
                img = enhance_image_for_hijab(img)
            face_images.append(img)
        if face_3:
            img = image_from_upload(face_3)
            if enhance_hijab:
                img = enhance_image_for_hijab(img)
            face_images.append(img)
        if face_4:
            img = image_from_upload(face_4)
            if enhance_hijab:
                img = enhance_image_for_hijab(img)
            face_images.append(img)
        
        # Perform face swap
        result_image, swap_info = swapper.swap_faces(base_img, face_images)
        
        # Save result
        output_filename = f"swap_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_image(result_image, output_path)
        
        return SwapResponse(
            success=True,
            message=f"Successfully swapped {len(face_images)} face(s)",
            output_image_url=f"/output/{output_filename}",
            swap_info=swap_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/swap-specific", response_model=SwapResponse)
async def swap_faces_specific(
    base_image: UploadFile = File(..., description="Base image containing faces to be replaced"),
    face_1: UploadFile = File(..., description="First source face image"),
    target_1: int = Form(..., description="Target face index for face_1 (0-based)"),
    face_2: Optional[UploadFile] = File(None, description="Second source face image (optional)"),
    target_2: Optional[int] = Form(None, description="Target face index for face_2 (0-based)"),
    face_3: Optional[UploadFile] = File(None, description="Third source face image (optional)"),
    target_3: Optional[int] = Form(None, description="Target face index for face_3 (0-based)"),
    face_4: Optional[UploadFile] = File(None, description="Fourth source face image (optional)"),
    target_4: Optional[int] = Form(None, description="Target face index for face_4 (0-based)")
):
    """
    Swap faces with explicit target indices.
    
    Allows you to specify which face in the base image should be replaced by each source face.
    Faces in the base image are indexed from 0 (leftmost) to N-1.
    
    Example: If base image has 3 faces and you want to replace only the middle and rightmost:
    - face_1 with target_1=1 (middle face)
    - face_2 with target_2=2 (rightmost face)
    """
    try:
        swapper = get_face_swapper()
        
        # Load base image
        base_img = image_from_upload(base_image)
        
        # Build face mappings
        face_mappings = [(image_from_upload(face_1), target_1)]
        
        if face_2 and target_2 is not None:
            face_mappings.append((image_from_upload(face_2), target_2))
        if face_3 and target_3 is not None:
            face_mappings.append((image_from_upload(face_3), target_3))
        if face_4 and target_4 is not None:
            face_mappings.append((image_from_upload(face_4), target_4))
        
        # Perform face swap
        result_image, swap_info = swapper.swap_specific_faces(base_img, face_mappings)
        
        # Save result
        output_filename = f"swap_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_image(result_image, output_path)
        
        return SwapResponse(
            success=True,
            message=f"Successfully swapped {len(face_mappings)} face(s)",
            output_image_url=f"/output/{output_filename}",
            swap_info=swap_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/output/{filename}")
async def get_output_image(filename: str):
    """
    Retrieve a generated output image
    
    Args:
        filename: Name of the output file
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output image not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.delete("/cleanup")
async def cleanup_files():
    """
    Clean up temporary upload and output files
    """
    try:
        # Clean upload directory
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clean output directory
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
