if __name__=='__main__':
    import sys,os; cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))
from util_and_constant import *
from pathlib import Path
import cv2
import numpy as np
from typing import Union, List, Optional, Dict, Any
from natsort import natsorted
import glob
from lmk_util.mp_utils import LMKExtractor  
from lmk_util.draw_utils import FaceMeshVisualizer
from PIL import Image
from skimage.io import imsave
import torch


def pil_to_cv2(pil_img):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_img).astype(np.uint8), cv2.COLOR_RGB2BGR)
def cv2_to_pil(cv2_img):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB).astype(np.uint8))

class LandmarkExtractor:
    """
    A wrapper class for face landmark extraction.
    
    This class provides an interface to extract facial landmarks from images.
    """
    
    def __init__(self, include_visualizer: bool=False, fps: int = 25, **kw_of_vis):
        """
        Initialize the landmark extractor.
        
        Args:
            fps: Frames per second for video processing (default: 25)
        """
        self.lmk_extractor = LMKExtractor(FPS=fps)
        if include_visualizer:
            self.visualizer = LandmarkVisualizer(**kw_of_vis)
    
    def extract_single(self, image: np.ndarray, only_main_lmk = False) -> np.ndarray:
        """
        Extract landmarks from a single image.
        
        Args:
            image: Input image as numpy array (h, w, 3)
            
        Returns:
            Landmarks as numpy array (N, 2) with absolute coordinates, or None if detection failed
        """
        if 0:
            save_dir = Path("4debug/LandmarkExtractor-extract_single")
            save_dir.mkdir(parents=0, exist_ok=True)
            save_path = save_dir / f"{str_t_pid()}.jpg"
            imsave(str(save_path), image)
            print(f"{save_path=}")
        # Extract landmarks
        result = self.lmk_extractor(image)
        
        if result is None:
            if 0:
                save_dir = Path("4debug/LandmarkExtractor-extract_single--no-result")
                save_dir.mkdir(parents=0, exist_ok=True)
                save_path = save_dir / f"{str_t_pid()}.jpg"
                imsave(str(save_path), image)
                print(f"Landmark not detected: {save_path}")
            # assert 0, (image.shape, save_path, image)
            return None
        
        # Extract 2D landmarks and convert to absolute coordinates
        lmks = result["lmks"]  # Shape: (num_landmarks, 3), normalized coordinates
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to absolute pixel coordinates
        absolute_coords = lmks[:, :2] * [w, h]  # (N, 2)
        if only_main_lmk:
            absolute_coords = lmkAll_2_lmkMain(absolute_coords)
        
        return absolute_coords

class LandmarkVisualizer:
    """
    A class for visualizing facial landmarks on images.
    """
    
    def __init__(self,img_256_mode=True):
        """Initialize the landmark visualizer."""
        self.visualizer = FaceMeshVisualizer(
            draw_iris=False, 
            draw_mouse=True, 
            draw_eye=True, 
            draw_nose=True, 
            draw_eyebrow=True, 
            draw_pupil=True,
            line_thickness=2,
            img_256_mode=img_256_mode,
        )
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                          target_size: tuple = (512, 512), use_connections: bool = True) -> np.ndarray:
        """
        Visualize landmarks on an image.
        
        Args:
            image: Input image as numpy array (h, w, 3)
            landmarks: Landmark coordinates as numpy array (N, 2) with absolute coordinates
            target_size: Target image size for visualization
            use_connections: Whether to use MediaPipe connections (only works with 468+ landmarks)
            
        Returns:
            Image with landmarks drawn as numpy array (BGR format)
        """
        image = image.copy()
        img_resized = cv2.resize(image, target_size)
        
        if use_connections and landmarks.shape[0] >= 468:
            # Use original MediaPipe visualizer with connections
            # Convert absolute coordinates to normalized coordinates for visualizer
            h, w = target_size
            normalized_lmks = landmarks / [image.shape[1], image.shape[0]]  # Normalize by original image size
            
            # Add dummy z coordinate
            lmks_3d = np.column_stack([normalized_lmks, np.zeros(len(normalized_lmks))])  # (N, 3)
            
            # Draw landmarks
            if 0:
                landmark_img = self.visualizer.draw_landmarks(target_size, lmks_3d, normed=True)
                combined = (img_resized * 0.5 + landmark_img * 0.5).clip(0, 255).astype(np.uint8)
            else:
                combined = self.visualizer.draw_landmarks(target_size, lmks_3d, normed=True, image=img_resized, )
        else:
            # Draw simple points for smaller landmark sets
            combined = img_resized.copy()
            
            # Convert coordinates to target size
            scale_x = target_size[0] / image.shape[1]
            scale_y = target_size[1] / image.shape[0]
            scaled_landmarks = landmarks * [scale_x, scale_y]
            
            # Draw each landmark as a colored circle
            for i, (x, y) in enumerate(scaled_landmarks):
                x, y = int(x), int(y)
                # Use different colors for different regions
                if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
                    cv2.circle(combined, (x, y), 2, (255, 0 , 0), -1)  # red/blue (depends on RGB/BGR) points
        
        return combined
    
    def save_landmark_visualization(self, image: np.ndarray, landmarks: np.ndarray,
                                  output_path: Union[str, Path],
                                  target_size: tuple = (512, 512), use_connections: bool = True) -> None:
        """
        Save landmark visualization to file.
        
        Args:
            image: Input image as numpy array (h, w, 3)
            landmarks: Landmark coordinates as numpy array (N, 2) with absolute coordinates
            output_path: Output file path
            target_size: Target image size for visualization
            use_connections: Whether to use MediaPipe connections (only works with 468+ landmarks)
        """
        vis_img = self.visualize_landmarks(image, landmarks, target_size, use_connections)
        imsave(str(output_path), vis_img) 
        print(f"Saved visualization to: {output_path}")
from functools import lru_cache
@lru_cache(maxsize=None)
def get_lmkMain_indices(include_face_oval: bool, return_tensor: bool = False):
    # Main landmark indices based on MediaPipe face mesh
    # These indices are from FaceMeshVisualizer connections
    
    # Left eye landmarks (based on FACEMESH_LEFT_EYE connections)
    left_eye_indices = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,  # outer contour
        466, 388, 387, 386, 385, 384, 398  # inner contour
    ]
    
    # Right eye landmarks (based on FACEMESH_RIGHT_EYE connections)  
    right_eye_indices = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,  # outer contour
        246, 161, 160, 159, 158, 157, 173  # inner contour
    ]
    
    # Left eyebrow landmarks (based on FACEMESH_LEFT_EYEBROW connections)
    left_eyebrow_indices = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    
    # Right eyebrow landmarks (based on FACEMESH_RIGHT_EYEBROW connections)
    right_eyebrow_indices = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    
    # Lip landmarks (based on LIPS definition in draw_utils.py)
    lips_outer_bottom_left = [61, 146, 91, 181, 84, 17]
    lips_outer_bottom_right = [17, 314, 405, 321, 375, 291]
    lips_inner_bottom_left = [78, 95, 88, 178, 87, 14]
    lips_inner_bottom_right = [14, 317, 402, 318, 324, 308]
    lips_outer_top_left = [61, 185, 40, 39, 37, 0]
    lips_outer_top_right = [0, 267, 269, 270, 409, 291]
    lips_inner_top_left = [78, 191, 80, 81, 82, 13]
    lips_inner_top_right = [13, 312, 311, 310, 415, 308]
    
    # Nose landmarks
    nose_indices = [4]  # nose tip, defined in draw_utils.py nose_landmark_spec
    
    # Pupil landmarks (gaze landmarks)
    pupil_indices = [468, 473]  # 468: right iris center, 473: left iris center
    
    # Face contour landmarks (based on MediaPipe FACEMESH_FACE_OVAL)
    face_oval_indices = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 
                         234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
    
    # Merge all main landmark indices
    main_indices = set()
    
    # Add eye landmarks
    main_indices.update(left_eye_indices)
    main_indices.update(right_eye_indices)
    
    # Add eyebrow landmarks
    main_indices.update(left_eyebrow_indices) 
    main_indices.update(right_eyebrow_indices)
    
    # Add lip landmarks
    main_indices.update(lips_outer_bottom_left)
    main_indices.update(lips_outer_bottom_right)
    main_indices.update(lips_inner_bottom_left)
    main_indices.update(lips_inner_bottom_right)
    main_indices.update(lips_outer_top_left)
    main_indices.update(lips_outer_top_right)
    main_indices.update(lips_inner_top_left)
    main_indices.update(lips_inner_top_right)
    
    # Add nose landmarks
    main_indices.update(nose_indices)
    
    # Add pupil landmarks (gaze landmarks)
    main_indices.update(pupil_indices)
    
    # Add face contour landmarks if requested
    if include_face_oval:
        main_indices.update(face_oval_indices)
    
    indices = sorted(main_indices)
    if return_tensor:
        return torch.as_tensor(indices, dtype=torch.long)
    return indices
def lmkAll_2_lmkMain(lmks468or478: np.ndarray, include_face_oval: bool = False) -> np.ndarray:
    """
    Convert 468/478 landmarks to a main landmark set.
    Based on MediaPipe visualization, extract key landmarks for eyes, eyebrows, lips, nose, pupils, etc.
    
    Args:
        lmks468or478: 468/478 landmark coordinates, shape (468, 2)
        include_face_oval: whether to include face contour landmarks (default: False)
        
    Returns:
        Main landmark coordinates, shape (N, 2)
    """
    if len(lmks468or478)<473:
        raise Exception(lmks468or478.shape)
    
    main_indices = get_lmkMain_indices(include_face_oval)
    # Filter indices out of range (e.g., iris points 468, 473 exist only in refined mode)
    valid_indices = [idx for idx in main_indices if idx < lmks468or478.shape[0]]
    
    # Sort by index for consistency
    valid_indices = sorted(valid_indices)
    
    # Extract the corresponding landmarks
    main_landmarks = lmks468or478[valid_indices]
    
    return main_landmarks
if __name__=='__main__':
    """Test the landmark extractor functionality."""
    print("Testing LandmarkExtractor...")
    
    # Initialize extractor and visualizer
    extractor = LandmarkExtractor()
    visualizer = LandmarkVisualizer()
    
    
    if not test_images:
        print(f"No test images found in {test_img_dir}")
        exit(0)
    
    print(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path("4debug/landmark_test_output")
    output_dir.mkdir(exist_ok=True)
    print(f"{output_dir=}")
    
    # Test single image extraction
    print("\n=== Testing single image extraction ===")
    for i, img_path in enumerate(test_images):
        print(f"Processing {img_path}")
        
        # Load image
        img_cv2 = cv2.imread(str(img_path))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        if img_cv2 is None:
            print(f"  ❌ Could not load image from {img_path}")
            continue
        
        # Extract landmarks
        landmarks = extractor.extract_single(img_cv2)
        print(f"{landmarks[:4]=}")
        
        if landmarks is None:
            print(f"  ❌ Failed to extract landmarks from {img_path}")
            continue
        
        print(f"  ✅ Extracted {landmarks.shape[0]} landmarks")
        print(f"     Landmark shape: {landmarks.shape}")  # (N, 2)
        
        # Test main landmark extraction (without face contour)
        main_landmarks = lmkAll_2_lmkMain(landmarks, include_face_oval=False)
        print(f"Extracted {main_landmarks.shape[0]} main landmarks (without face oval) from {landmarks.shape[0]} total landmarks")
        
        # Test main landmark extraction (with face contour)
        main_landmarks_with_oval = lmkAll_2_lmkMain(landmarks, include_face_oval=True)
        print(f"Extracted {main_landmarks_with_oval.shape[0]} main landmarks (with face oval) from {landmarks.shape[0]} total landmarks")
        
        # Visualize and save original landmarks
        output_path = output_dir / f"landmark_vis_{i+1}_{Path(img_path).stem}_all468.jpg"
        visualizer.save_landmark_visualization(img_cv2, landmarks, output_path)
        print(f"  📁 Saved all 468 landmarks visualization to {output_path}")
        
        # Visualize and save main landmarks only (use simple points, not connections)
        output_path_main = output_dir / f"landmark_vis_{i+1}_{Path(img_path).stem}_main.jpg"
        visualizer.save_landmark_visualization(img_cv2, main_landmarks, output_path_main, use_connections=False)
        print(f"  📁 Saved main landmarks (without face oval) visualization to {output_path_main}")
        
        # Visualize and save main landmarks with face oval
        output_path_main_oval = output_dir / f"landmark_vis_{i+1}_{Path(img_path).stem}_main_with_oval.jpg"
        visualizer.save_landmark_visualization(img_cv2, main_landmarks_with_oval, output_path_main_oval, use_connections=False)
        print(f"  📁 Saved main landmarks (with face oval) visualization to {output_path_main_oval}")
