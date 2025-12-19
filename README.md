# Multi-Face Swap API 🎭

A powerful and smooth API for swapping up to 4 faces simultaneously in a single image using InsightFace and FastAPI.

## Features ✨

- **Multi-Face Support**: Swap up to 4 faces in one operation
- **Two Swap Modes**: 
  - Sequential: Replaces faces from left to right automatically
  - Specific: Choose exactly which face to replace with each source
- **Hijab-Friendly Mode**: 🧕 Special handling for faces with hijab/headwear
  - Dual detection mode (standard + sensitive)
  - Automatic image enhancement for better detection
  - Preprocessing functions for optimal results
- **REST API**: Easy-to-use HTTP API with FastAPI
- **Python Module**: Direct usage in Python code
- **High Quality**: Uses InsightFace's state-of-the-art face swapping model
- **GPU Support**: Automatically uses GPU if available (CUDA)

## Installation 🚀

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for speed)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd swap-multi-face
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: First run will download InsightFace models (~500MB) automatically.

## Usage 📖

### Option 1: REST API (Recommended for Integration)

#### Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

#### API Documentation

Once the server is running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **API info**: http://localhost:8000

#### Example API Requests

**1. Sequential Face Swap (replaces faces left to right)**

```bash
curl -X POST http://localhost:8000/swap \
  -F "base_image=@base_image.jpg" \
  -F "face_1=@person1.jpg" \
  -F "face_2=@person2.jpg"
```

**2. Specific Face Swap (choose which face to replace)**

```bash
curl -X POST http://localhost:8000/swap-specific \
  -F "base_image=@base_image.jpg" \
  -F "face_1=@person1.jpg" \
  -F "target_1=1" \
  -F "face_2=@person2.jpg" \
  -F "target_2=0"
```

In this example:
- `target_1=1` means replace the 2nd face (index 1) with person1.jpg
- `target_2=0` means replace the 1st face (index 0) with person2.jpg

**3. Face Swap with Hijab (enhanced detection) 🧕**

```bash
curl -X POST http://localhost:8000/swap \
  -F "base_image=@base_image.jpg" \
  -F "face_1=@person1.jpg" \
  -F "face_2=@person_hijab.jpg" \
  -F "enhance_hijab=true"
```

**Note**: The API automatically handles hijab detection. Use `enhance_hijab=true` for better results.
For complete hijab guide, see [HIJAB_GUIDE.md](HIJAB_GUIDE.md)

**4. Download Result**

```bash
curl http://localhost:8000/output/swap_xxxxx.jpg -o result.jpg
```

#### Python API Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Open your images
with open("base_image.jpg", "rb") as base, \
     open("face1.jpg", "rb") as f1, \
     open("face2.jpg", "rb") as f2:
    
    files = {
        "base_image": base,
        "face_1": f1,
        "face_2": f2
    }
    
    response = requests.post(f"{API_URL}/swap", files=files)
    result = response.json()
    
    print(f"Success: {result['message']}")
    print(f"Output: {result['output_image_url']}")
    
    # Download result
    output_url = f"{API_URL}{result['output_image_url']}"
    img_response = requests.get(output_url)
    with open("result.jpg", "wb") as f:
        f.write(img_response.content)
```

### Option 2: Direct Python Usage

#### Simple Command Line Script

```bash
python simple_swap.py base_image.jpg face1.jpg face2.jpg
```

This will:
1. Load `base_image.jpg` (should contain 2+ faces)
2. Replace faces with `face1.jpg` and `face2.jpg` (left to right)
3. Save result as `output.jpg`

#### Python Code

```python
from face_swapper import MultiFaceSwapper, load_image, save_image

# Initialize (hijab mode enabled by default)
swapper = MultiFaceSwapper(enable_hijab_mode=True)

# Load images
base_image = load_image("base_image.jpg")
face_1 = load_image("person1.jpg")
face_2 = load_image("person2.jpg")

# Method 1: Sequential swap (left to right)
result, info = swapper.swap_faces(base_image, [face_1, face_2])
save_image(result, "output_sequential.jpg")

# Method 2: Specific face indices
result, info = swapper.swap_faces(
    base_image, 
    [face_1, face_2],
    target_face_indices=[1, 0]  # Swap to 2nd and 1st face
)
save_image(result, "output_specific.jpg")

print(f"Swapped {info['swapped_count']} faces")
print(f"Total faces in base: {info['total_base_faces']}")
```

#### Python Code with Hijab 🧕

```python
from face_swapper import MultiFaceSwapper, load_image, preprocess_hijab_image, save_image

# Initialize with hijab mode
swapper = MultiFaceSwapper(enable_hijab_mode=True)

# Load images
base_image = load_image("base_image.jpg")
face_1 = load_image("person1.jpg")

# Use preprocessing for hijab image
face_2_hijab = preprocess_hijab_image("person2_hijab.jpg")  # 🧕

# Perform swap
result, info = swapper.swap_faces(base_image, [face_1, face_2_hijab])
save_image(result, "output_with_hijab.jpg")

print(f"✓ Successfully swapped {info['swapped_count']} faces")
```

**See [HIJAB_GUIDE.md](HIJAB_GUIDE.md) for complete hijab handling documentation.**

## API Endpoints 🔌

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/swap` | POST | Sequential face swap (left to right) |
| `/swap-specific` | POST | Swap with explicit target indices |
| `/output/{filename}` | GET | Download result image |
| `/cleanup` | DELETE | Clean temporary files |

## How It Works 🔬

1. **Face Detection**: Uses InsightFace's SCRFD detector to find all faces in images
2. **Face Analysis**: Extracts facial features and landmarks
3. **Face Swapping**: Uses the InSwapper model to seamlessly swap faces
4. **Ordering**: Faces in base image are sorted left-to-right for consistent indexing

### Face Indexing

Faces are indexed from **left to right** starting at **0**:

```
Base Image with 3 faces:
  [Face 0]  [Face 1]  [Face 2]
   (left)   (middle)   (right)
```

## Parameters 📋

### `/swap` endpoint

- `base_image` (required): Image with faces to replace
- `face_1` (required): First source face
- `face_2` (optional): Second source face
- `face_3` (optional): Third source face
- `face_4` (optional): Fourth source face

### `/swap-specific` endpoint

- `base_image` (required): Image with faces to replace
- `face_1` (required): First source face
- `target_1` (required): Target index for face_1
- `face_2` (optional): Second source face
- `target_2` (optional): Target index for face_2
- `face_3` (optional): Third source face
- `target_3` (optional): Target index for face_3
- `face_4` (optional): Fourth source face
- `target_4` (optional): Target index for face_4

## Response Format 📤

```json
{
  "success": true,
  "message": "Successfully swapped 2 face(s)",
  "output_image_url": "/output/swap_abc123.jpg",
  "swap_info": {
    "total_base_faces": 3,
    "swapped_count": 2,
    "swaps": [
      {
        "target_index": 0,
        "target_bbox": [100, 150, 250, 300],
        "source_bbox": [50, 75, 200, 225]
      }
    ]
  }
}
```

## Examples 💡

### Example 1: Group Photo - Replace 2 People

You have a group photo with 4 people and want to replace 2 of them:

```bash
curl -X POST http://localhost:8000/swap \
  -F "base_image=@group_photo.jpg" \
  -F "face_1=@new_person1.jpg" \
  -F "face_2=@new_person2.jpg"
```

This replaces the two leftmost faces.

### Example 2: Replace Specific People in Photo

You have a photo with 3 people and want to replace only the middle and rightmost person:

```bash
curl -X POST http://localhost:8000/swap-specific \
  -F "base_image=@photo.jpg" \
  -F "face_1=@replacement1.jpg" \
  -F "target_1=1" \
  -F "face_2=@replacement2.jpg" \
  -F "target_2=2"
```

### Example 3: Swap All 4 Faces

```python
import requests

with open("base_4_faces.jpg", "rb") as base, \
     open("face1.jpg", "rb") as f1, \
     open("face2.jpg", "rb") as f2, \
     open("face3.jpg", "rb") as f3, \
     open("face4.jpg", "rb") as f4:
    
    files = {
        "base_image": base,
        "face_1": f1,
        "face_2": f2,
        "face_3": f3,
        "face_4": f4
    }
    
    response = requests.post("http://localhost:8000/swap", files=files)
    result = response.json()
```

## Performance 🚀

- **GPU (CUDA)**: ~1-2 seconds per image
- **CPU**: ~5-10 seconds per image

Performance depends on image resolution and number of faces.

## Troubleshooting 🔧

### Common Issues

**1. "No faces detected in base image"**
- Ensure the image has clear, visible faces
- Check image quality and lighting
- Try a different image

**2. Models downloading slowly**
- First run downloads ~500MB of models
- Be patient, this is one-time only
- Check your internet connection

**3. Out of memory errors**
- Reduce image resolution
- Process fewer faces at once
- Use CPU instead of GPU

**4. API not starting**
- Check if port 8000 is available
- Try a different port: `uvicorn api:app --port 8001`
- Check if all dependencies are installed

### Changing to CPU-only

Edit `requirements.txt` and change:
```
onnxruntime-gpu==1.16.3
```
to:
```
onnxruntime==1.16.3
```

Then reinstall: `pip install -r requirements.txt --force-reinstall`

## Project Structure 📁

```
swap-multi-face/
├── api.py                 # FastAPI REST API server
├── face_swapper.py        # Core face swapping logic
├── simple_swap.py         # Simple CLI script
├── example_usage.py       # Example code and curl commands
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # This file
├── models/               # InsightFace models (auto-downloaded)
├── uploads/              # Temporary upload files
└── outputs/              # Generated output images
```

## Requirements 📦

- fastapi
- uvicorn
- python-multipart
- insightface
- onnxruntime-gpu (or onnxruntime for CPU)
- opencv-python
- numpy
- Pillow

## License 📄

This project uses InsightFace models which are subject to their own licenses.

## Credits 🙏

- **InsightFace**: State-of-the-art face analysis toolkit
- **FastAPI**: Modern web framework for building APIs

## Support 💬

For issues or questions, please check:
1. This README
2. The example files
3. API documentation at `/docs` endpoint

---

Made with ❤️ for smooth multi-face swapping
