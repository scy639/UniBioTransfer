import numpy as np
import random
from torchvision import transforms as T
from torch import Tensor
from PIL import Image
import torchvision, torch, cv2

def get_tensor(normalize=True, toTensor=True,
    mean = (0.5, 0.5, 0.5),
    std = (0.5, 0.5, 0.5),
):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize(mean,std)]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True,
    mean = (0.48145466, 0.4578275, 0.40821073),
    std = (0.26862954, 0.26130258, 0.27577711),
):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize(mean,std)]
    return torchvision.transforms.Compose(transform_list)

def mask_after_npisin__2__tensor(mask_after_npisin: np.ndarray) -> Tensor:
    converted_mask = np.zeros_like(mask_after_npisin)
    converted_mask[mask_after_npisin] = 255
    mask_tensor = Image.fromarray(converted_mask).convert('L')
    mask_tensor = get_tensor(normalize=False, toTensor=True)(mask_tensor)
    mask_tensor = T.Resize([512, 512])(mask_tensor)
    return mask_tensor

# Implement perspective warp for reference images
def apply_perspective_warp(img, mask, deg_x, deg_y,
    # border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR, 
    border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, 
    constant_border_value=(0,0,0),
    fix_edge_artifacts=False, # no noticeable difference
):
    """
    Apply a perspective warp transformation to an image and mask
    
    Args:
        img: numpy array of shape (H, W, C)
        mask: numpy array of shape (H, W)
        max_deg: maximum rotation degree
        border_mode: border handling mode (cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT, etc.)
        interpolation: interpolation method (cv2.INTER_LINEAR, cv2.INTER_CUBIC, etc.)
        constant_border_value: border color to use with BORDER_CONSTANT
        fix_edge_artifacts: Whether to apply additional processing to fix edge artifacts
    
    Returns:
        transformed_img, transformed_mask
    """
    h, w = img.shape[:2]
    assert img.shape[:2] == mask.shape[:2], f"img shape {img.shape[:2]} != mask shape {mask.shape[:2]}"
    
    # Convert degrees to radians
    rad_x = np.deg2rad(deg_x)
    rad_y = np.deg2rad(deg_y)
    
    # Calculate perspective transform matrix
    d = np.sqrt(h**2 + w**2)
    eye_to_center = d / (2 * np.tan(np.pi/8))  # approx distance from eye to image center
    
    # Define the transformation matrix
    transform = np.eye(3)
    
    # Apply rotation around X axis (vertical)
    transform = transform @ np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    
    # Apply rotation around Y axis (horizontal)
    transform = transform @ np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    # Project 3D points onto 2D plane
    pts_3d = np.array([
        [-w/2, -h/2, 0],
        [w/2, -h/2, 0],
        [w/2, h/2, 0],
        [-w/2, h/2, 0]
    ])
    
    # Apply transformation
    pts_3d_transformed = pts_3d @ transform.T
    
    # Project to 2D
    pts_3d_transformed[:, 0] = pts_3d_transformed[:, 0] * eye_to_center / (eye_to_center + pts_3d_transformed[:, 2]) + w/2
    pts_3d_transformed[:, 1] = pts_3d_transformed[:, 1] * eye_to_center / (eye_to_center + pts_3d_transformed[:, 2]) + h/2
    
    src_pts = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)
    
    dst_pts = np.array([
        [pts_3d_transformed[0, 0], pts_3d_transformed[0, 1]],
        [pts_3d_transformed[1, 0], pts_3d_transformed[1, 1]],
        [pts_3d_transformed[2, 0], pts_3d_transformed[2, 1]],
        [pts_3d_transformed[3, 0], pts_3d_transformed[3, 1]]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply perspective transformation with specified border mode and interpolation
    transformed_img = cv2.warpPerspective(img, M, (w, h), flags=interpolation, 
                                          borderMode=border_mode, 
                                          borderValue=constant_border_value)
    
    # For mask, always use nearest neighbor interpolation
    transformed_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, 
                                           borderMode=border_mode, 
                                           borderValue=0)
    
    # Additional processing to fix edge artifacts
    if fix_edge_artifacts:
        # Calculate edge detection mask to find problematic areas
        edge_mask = np.ones((h, w), dtype=np.uint8)
        warped_edge_mask = cv2.warpPerspective(edge_mask, M, (w, h), flags=cv2.INTER_NEAREST, 
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Create transition region mask with larger dilation for better handling of edge artifacts
        kernel = np.ones((7, 7), np.uint8)
        inner_edge = cv2.erode(warped_edge_mask, kernel)
        transition_mask = warped_edge_mask - inner_edge
        
        # Only focus on vertical edges where artifacts are most common
        left_margin = 20
        right_margin = 20
        vertical_edge_mask = np.zeros_like(transition_mask)
        vertical_edge_mask[:, :left_margin] = transition_mask[:, :left_margin]
        vertical_edge_mask[:, -right_margin:] = transition_mask[:, -right_margin:]
        
        # Apply stronger smoothing specifically to vertical edges
        if len(transformed_img.shape) == 3:
            # Create a smooth blend from interior to exterior
            for i in range(3):  # For each color channel
                if np.sum(vertical_edge_mask) > 0:
                    # Apply a stronger blur to vertical edges
                    blurred = cv2.GaussianBlur(transformed_img, (9, 9), 0)
                    vertical_edge_mask_3d = np.stack([vertical_edge_mask] * 3, axis=2) / 255.0
                    transformed_img = transformed_img * (1 - vertical_edge_mask_3d) + blurred * vertical_edge_mask_3d
            
            # Apply general edge blending as well
            edge_blurred = cv2.GaussianBlur(transformed_img, (3, 3), 0)
            transition_mask_3d = np.stack([transition_mask] * 3, axis=2) / 255.0
            transformed_img = transformed_img * (1 - transition_mask_3d) + edge_blurred * transition_mask_3d
    
    # Ensure the output image is uint8
    if transformed_img.dtype != np.uint8:
        transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
    
    # Ensure the output mask is uint8
    if transformed_mask.dtype != np.uint8:
        transformed_mask = np.clip(transformed_mask, 0, 255).astype(np.uint8)
    
    return transformed_img, transformed_mask
