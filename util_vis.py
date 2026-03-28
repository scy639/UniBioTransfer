from pathlib import Path
import cv2
import numpy as np

def vis_tensors_A(l_tensor_or_named_tensor, path_grid, vis_batch_size=4, layout='auto'):
    """Visualize a list of tensors in a grid layout.
    Args:
        l_tensor_or_named_tensor: [tensor | (name, tensor), ..]. each tensor: B,(C,)H,W is in [-1,1] range
        path_grid: Path object for saving the grid visualization
        vis_batch_size: number of samples to visualize
        layout: 'BxI' (batch x images) or 'IxB' (images x batch) or 'auto'
    """
    import torch
    from torchvision.utils import make_grid, save_image
    path_grid = Path(path_grid)
    path_grid.parent.mkdir(parents=0, exist_ok=True)
    # Helper function to unnormalize and prepare images for saving
    def prepare_for_vis(tensor, ):
        if tensor is None:
            return None
        shape = tensor.shape
        assert shape[1]<=3
        if len(shape)==3 or shape[1]==1:
            is_mask = True
        else:  is_mask = False
        if is_mask:
            return tensor.repeat(1, 3, 1, 1).cpu()  # Expand mask to 3 channels
        else:
            return (tensor * 0.5 + 0.5).cpu()  # Unnormalize from [-1, 1] to [0, 1]
    named_tensors = []
    for tensor_or_named_tensor in l_tensor_or_named_tensor:
        if isinstance(tensor_or_named_tensor, tuple):
            name, tensor = tensor_or_named_tensor
        else:
            name = ""
            tensor = tensor_or_named_tensor
        if tensor is not None:
            named_tensors.append((name, prepare_for_vis(tensor.detach()[:vis_batch_size], )))
    # Make sure all tensors have the same spatial dimensions  
    all_shapes = [img.shape[2:] for _, img in named_tensors if img is not None]
    if len(set(all_shapes)) > 1:  # Pad images to match the largest dimensions
        max_h = max(shape[0] for shape in all_shapes)
        max_w = max(shape[1] for shape in all_shapes)
        for i in range(len(named_tensors)):
            name, img = named_tensors[i]
            if img is None:
                continue
            if img.shape[2] == max_h and img.shape[3] == max_w:
                continue
            pad_h = max_h - img.shape[2]
            pad_w = max_w - img.shape[3]
            named_tensors[i] = (name, torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0))
    tensors = []
    for _, (name, tensor) in enumerate(named_tensors):
        tensor = tensor.detach()
        if name:
            for b in range(tensor.shape[0]):
                # Convert tensor to numpy for OpenCV
                img = tensor[b].permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8).copy()  # Make contiguous copy for OpenCV
                # Add text
                cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 1, cv2.LINE_AA)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0 # Convert back to tensor
                tensors.append(img_tensor)
        else:
            for b in range(tensor.shape[0]):
                tensors.append(tensor[b])
    if tensors: # I*B,3,..
        all_images_flat = torch.stack(tensors) # I*B,3,..
        I = len(named_tensors)
        B = vis_batch_size
        if layout == 'auto':
            if B/I > 0.8:
                layout = 'IxB'
            else:
                layout = 'BxI'
        if layout == 'BxI':
            all_images_nonflat = all_images_flat.reshape(I, B, *all_images_flat.shape[1:])
            all_images_nonflat = all_images_nonflat.permute(1, 0, 2, 3, 4)
            all_images_flat = all_images_nonflat.reshape(-1, *all_images_flat.shape[1:])
            nrow = I
        else:  # 'IxB'
            nrow = B
        save_image(make_grid(all_images_flat, nrow=nrow), path_grid)
        print(f"{path_grid=}")

def visualize_landmarks(image, landmarks, save_path):
    """
    Draw landmarks on an image and save the result.
    
    Args:
        image: Input image as a numpy array (H,W,3) with values in [0,255]
        landmarks: Numpy array of shape (136,) or (68,2) containing 68 keypoint coordinates
        save_path: Path where the annotated image should be written
    """
    # Clone the image and ensure uint8 type
    image = image.copy().astype(np.uint8)
    
    # Ensure the image buffer is contiguous
    image = np.ascontiguousarray(image)
    
    # Reshape landmarks into (68,2) if needed
    if landmarks.shape[0] == 136:
        landmarks = landmarks.reshape(68, 2)
    
    # Draw each landmark point
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # Save the annotated image
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def visualize_headPose(img_path, yaw, pitch, roll, save_path):
    """Visualize pose angles on image using arrows
    Args:
        img_path: Path to input image
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    center = (w//2, h//2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Yaw (left-right)
    yaw_rad = np.radians(yaw)
    yaw_end = (center[0] + int(100 * np.sin(yaw_rad)), 
               center[1] - int(100 * np.cos(yaw_rad)))
    plt.arrow(center[0], center[1], yaw_end[0]-center[0], yaw_end[1]-center[1],
              color='r', width=2, head_width=20, label=f'Yaw: {yaw:.1f}°')
    
    # Pitch (up-down)
    pitch_rad = np.radians(pitch)
    pitch_end = (center[0] + int(100 * np.sin(pitch_rad)),
                 center[1] - int(100 * np.cos(pitch_rad)))
    plt.arrow(center[0], center[1], pitch_end[0]-center[0], pitch_end[1]-center[1],
              color='g', width=2, head_width=20, label=f'Pitch: {pitch:.1f}°')
    
    # Roll (tilt)
    roll_rad = np.radians(roll)
    roll_end = (center[0] + int(100 * np.cos(roll_rad)),
                center[1] + int(100 * np.sin(roll_rad)))
    plt.arrow(center[0], center[1], roll_end[0]-center[0], roll_end[1]-center[1],
              color='b', width=2, head_width=20, label=f'Roll: {roll:.1f}°')
    
    plt.legend()
    plt.axis('off')
    
    # Save visualization
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"{save_path=}")
