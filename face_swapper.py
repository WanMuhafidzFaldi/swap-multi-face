"""
Multi-Face Swapper Module
Supports swapping up to 4 faces simultaneously in a single image
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os


class MultiFaceSwapper:
    """
    A class to handle multiple face swapping operations
    """
    
    def __init__(self, model_path: str = None, enable_hijab_mode: bool = True):
        """
        Initialize the face swapper with InsightFace models
        
        Args:
            model_path: Path to store/load InsightFace models
            enable_hijab_mode: Enable special handling for faces with hijab/headwear
        """
        self.model_path = model_path or './models'
        self.enable_hijab_mode = enable_hijab_mode
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize face analysis app for detection with higher sensitivity
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            root=self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # Use larger det_size for better detection of partially covered faces
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize secondary detector with different settings for hijab detection
        if self.enable_hijab_mode:
            self.face_app_sensitive = FaceAnalysis(
                name='buffalo_l',
                root=self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Lower detection threshold for faces with coverings (0.2 lebih agresif)
            self.face_app_sensitive.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)
        
        # Initialize face swapper model
        self.swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx',
            download=True,
            download_zip=True,
            root=self.model_path
        )
        
    def detect_faces(self, image: np.ndarray, use_sensitive: bool = False) -> List:
        """
        Detect all faces in an image with optional sensitive mode for hijab/headwear
        
        Args:
            image: Input image as numpy array (BGR format)
            use_sensitive: Use more sensitive detection for faces with coverings
            
        Returns:
            List of detected face objects
        """
        faces = []
        
        # Try standard detection first
        if not use_sensitive:
            faces = self.face_app.get(image)
        
        # If no faces found and hijab mode enabled, try sensitive detection
        if len(faces) == 0 and self.enable_hijab_mode and hasattr(self, 'face_app_sensitive'):
            faces = self.face_app_sensitive.get(image)
            if len(faces) > 0:
                print("  ✓ Face detected using sensitive mode (hijab-friendly)")
        
        # Sort faces by x-coordinate (left to right) for consistent ordering
        faces = sorted(faces, key=lambda x: x.bbox[0])
        return faces
    
    def swap_faces(
        self, 
        base_image: np.ndarray, 
        face_images: List[np.ndarray],
        target_face_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Swap multiple faces in the base image
        
        Args:
            base_image: The base image containing faces to be replaced (BGR)
            face_images: List of images containing source faces (BGR)
            target_face_indices: Optional list specifying which face in base_image 
                               to replace with each source face. If None, replaces
                               faces in order from left to right.
                               
        Returns:
            Tuple of (result_image, info_dict)
            - result_image: Image with swapped faces
            - info_dict: Dictionary with swap information
        """
        if len(face_images) > 4:
            raise ValueError("Maximum 4 faces can be swapped at once")
        
        # Detect faces in base image
        base_faces = self.detect_faces(base_image)
        
        if len(base_faces) == 0:
            raise ValueError("No faces detected in base image")
        
        # Extract source faces from provided images
        source_faces = []
        for i, face_img in enumerate(face_images):
            print(f"\n🔍 Processing source image {i+1}...")
            
            # Apply enhancement first for better detection (especially for hijab)
            enhanced_img = enhance_image_for_hijab(face_img)
            
            # Try standard detection first with enhanced image
            faces = self.detect_faces(enhanced_img, use_sensitive=False)
            
            # If no face detected, try with original image
            if len(faces) == 0:
                print(f"  ⚠ No face detected with enhanced image, trying original...")
                faces = self.detect_faces(face_img, use_sensitive=False)
            
            # If still no face, try sensitive mode with enhanced image
            if len(faces) == 0 and self.enable_hijab_mode:
                print(f"  ⚠ Trying sensitive detection mode (hijab-friendly)...")
                faces = self.detect_faces(enhanced_img, use_sensitive=True)
            
            # Last resort: try sensitive mode with original
            if len(faces) == 0 and self.enable_hijab_mode:
                print(f"  ⚠ Trying sensitive mode with original image...")
                faces = self.detect_faces(face_img, use_sensitive=True)
            
            print(f"  {'✓' if len(faces) > 0 else '✗'} Detected {len(faces)} face(s) in source image {i+1}")
            
            if len(faces) == 0:
                raise ValueError(
                    f"❌ No face detected in source image {i+1}. "
                    f"\n\nTips untuk foto hijab:"
                    f"\n  1. Pastikan wajah menghadap kamera (frontal)"
                    f"\n  2. Pencahayaan harus baik dan merata"
                    f"\n  3. Area wajah (mata, hidung, mulut) terlihat jelas"
                    f"\n  4. Hindari bayangan di wajah"
                    f"\n  5. Foto tidak blur atau terlalu gelap"
                    f"\n  6. Resolusi foto minimal 512x512 pixel"
                )
            
            # Use the largest face if multiple faces detected
            source_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Additional quality check for face with low confidence
            if hasattr(source_face, 'det_score'):
                confidence = source_face.det_score
                print(f"  📊 Detection confidence: {confidence:.2f}")
                if confidence < 0.5:
                    print(f"  ⚠ Warning: Low confidence. Result quality may vary.")
                elif confidence >= 0.7:
                    print(f"  ✨ Good confidence! Expected good results.")
            
            source_faces.append(source_face)
        
        # Determine which faces to swap
        if target_face_indices is None:
            # Default: swap faces in order (left to right)
            target_face_indices = list(range(len(source_faces)))
        
        if len(target_face_indices) != len(source_faces):
            raise ValueError("Number of target indices must match number of source faces")
        
        # Validate indices
        for idx in target_face_indices:
            if idx >= len(base_faces):
                raise ValueError(f"Target face index {idx} out of range. Base image has {len(base_faces)} faces.")
        
        # Perform face swapping
        result_image = base_image.copy()
        swap_info = {
            'total_base_faces': len(base_faces),
            'swapped_count': len(source_faces),
            'swaps': []
        }
        
        for source_face, target_idx in zip(source_faces, target_face_indices):
            target_face = base_faces[target_idx]
            result_image = self.swapper.get(result_image, target_face, source_face, paste_back=True)
            
            swap_info['swaps'].append({
                'target_index': target_idx,
                'target_bbox': target_face.bbox.tolist(),
                'source_bbox': source_face.bbox.tolist()
            })
        
        return result_image, swap_info
    
    def swap_specific_faces(
        self,
        base_image: np.ndarray,
        face_mappings: List[Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, dict]:
        """
        Swap faces with explicit mapping
        
        Args:
            base_image: The base image containing faces to be replaced
            face_mappings: List of tuples (source_face_image, target_face_index)
            
        Returns:
            Tuple of (result_image, info_dict)
        """
        face_images = [mapping[0] for mapping in face_mappings]
        target_indices = [mapping[1] for mapping in face_mappings]
        
        return self.swap_faces(base_image, face_images, target_indices)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def enhance_image_for_hijab(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better face detection when wearing hijab/headwear
    Applies contrast enhancement and brightness adjustment
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        Enhanced image as numpy array in BGR format
    """
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def preprocess_hijab_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image with hijab for optimal face detection
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image as numpy array in BGR format
    """
    image = load_image(image_path)
    enhanced = enhance_image_for_hijab(image)
    return enhanced


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save image to file
    
    Args:
        image: Image as numpy array in BGR format
        output_path: Path where to save the image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
