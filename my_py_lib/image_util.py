import numpy as np
import os,sys,cv2
import PIL
from PIL import Image
from pathlib import Path

def to__image_in_npArr(img):
    """
    convert PIL/np.ndarray type image to np.ndarray
    Equivalent to misc_util.to_ndarray
    """
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, PIL.Image.Image):
        return np.array(img)
    import torch
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    raise TypeError("got {}".format(type(img)))
def imgArr_2_objXminYminXmaxYmax(imgArr, bg_color, THRES=5, coarse_bbox=None,diff_type='A'):
    """
    param:
        imgArr: np.array
        bg_color: background color in the form of a tuple (R, G, B)
        coarse_bbox: find bbox inside the coarse_bbox
    return:
        xmin,ymin,xmax,ymax (type= primitive int,NOT np int)
    """
    img_array = imgArr
    if coarse_bbox is not None:
        xmin_coarse, ymin_coarse, xmax_coarse, ymax_coarse = coarse_bbox
        img_array = img_array[ymin_coarse:ymax_coarse, xmin_coarse:xmax_coarse]

    if diff_type=='A':
        # Extract pixels from the image that are different from the background color
        diff_pixels = np.any(np.abs(img_array - np.array(bg_color)) > THRES, axis=2)
    elif diff_type=='B':
        # Extract pixels from the image that are different from the background color
        diff_pixels =( np.sum(np.abs(img_array - np.array(bg_color)) , axis=2)> THRES)

    # Calculate the bounding box of the object
    rows = np.any(diff_pixels, axis=1)
    cols = np.any(diff_pixels, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin=xmin.item()
    ymin=ymin.item()
    xmax=xmax.item()
    ymax=ymax.item()
    if coarse_bbox is not None:
        xmin += xmin_coarse
        ymin += ymin_coarse
        xmax += xmin_coarse
        ymax += ymin_coarse

    return xmin, ymin, xmax, ymax
def draw_bbox(img, bbox, color=None, thickness=2,bbox_type='x0y0wh'):
    """
    xmin,ymin,xmax,ymax
    """
    img = np.copy(img)
    if color is not None:
        color = [int(c) for c in color]
    else:
        color = (0, 255, 0)
    if bbox_type=='x0y0wh':
        left = int(round(bbox[0]))
        top = int(round(bbox[1]))
        width = int(round(bbox[2]))
        height = int(round(bbox[3]))
    elif bbox_type=='x0y0x1y1':
        left,top,right,bottom=bbox
        width = right-left
        height = bottom-top
    img = cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
    return img




def print_image_statistics(
    image,
    reduce_line:bool = 1, # reduce printed lines by condensing multi-line output
    #
    return_:bool = False,
    print_:bool = True,
):
    """
    Print image statistics:
        type
        dtype and shape
        min, max, mean, median, unique values for each channel
    """
    string = "----[statistics]----\n"
    string += f"type = {type(image)}\n"
    image = to__image_in_npArr(image)
    string += f"dtype = {image.dtype}\n"
    string += f"shape = {image.shape}\n"

    if image.shape[0]==3 or image.shape[0]==4 or image.shape[0]==1:
        if image.shape[1] > 13:
            print("Assuming the first axis is channel", end=' ')
            if len(image.shape) == 2:
                raise NotImplementedError
            image = image.transpose(1, 2, 0)
            print(f"transposed {image.shape=}")
        else:
            print("[warning] the first axis might be the channel dimension")
    if len(image.shape) == 2:
        channels = [image]
    else:
        # channels = np.split(image, image.shape[-1], axis=-1)#poe generated, I cannot understand easily
        channels = [image[:, :, i] for i in range(image.shape[-1])]

    for i, channel in enumerate(channels):
        uniques=np.unique(channel)
        _N=6
        if len(uniques)>_N:
            s_uniques = " ".join([f"{x:.3f}" for x in uniques[:_N//2]])# Format the first half with two decimals
            s_uniques+=' .. '
            s_uniques += " ".join([f"{x:.3f}" for x in uniques[-_N//2:]])
        else:
            s_uniques = " ".join([f"{x:.3f}" for x in uniques])
        if not reduce_line:
            string += f"\nChannel {i }:\n"
            string += f"  Min: {np.min(channel)}\n"
            string += f"  Max: {np.max(channel)}\n"
            string += f"  Mean: {np.mean(channel)}\n"
            string += f"  Median: {np.median(channel)}\n"
            string += f"  Unique values: {s_uniques}\n"
        else:
            string += f"Channel {i}: Min={np.min(channel):<8.2f} Max={np.max(channel):<8.2f} Mean={np.mean(channel):<8.2f} Median={np.median(channel):<8.2f} Unique={s_uniques}\n"
    if reduce_line: # remove the first few newline characters from string
        def remove_first_n_char(text, char, n=3):
            modified = text
            for _ in range(n):
                modified = modified.replace(char, '', 1)
            return modified
        string = remove_first_n_char(string,'\n')
    string=string.replace('\n','\n|')
    string += "----[statistics]over----\n"
    if print_:
        print(string)
    if return_:
        return string

def pad_around_center(img, new_size,  ):
    """
    Pad image to a new size with fill color around image center.
    pad with white (255)
    """
    img = to__image_in_npArr(img)
    assert len(img.shape) == 3
    assert len(new_size) == 2

    # compute padding
    height, width, _ = img.shape
    new_height, new_width = new_size
    assert new_height >= height
    assert new_width >= width
    pad_height = new_height - height
    pad_width = new_width - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # pad image
    img = np.pad(
        img,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return img



def norm_min0max255_image_per_channel(image,  ):
    """
    norm to 0-255 for each channel (min=0, max=255 for each channel)
    
    Args:
        image: file path (str) or PIL Image object
    
    Returns:
        normalized PIL Image
    """
    if isinstance(image, str):
        img_pil = Image.open(image).convert('RGB')
    else:
        img_pil = image.convert('RGB')
    
    img_array = np.array(img_pil).astype(np.float32)
    
    for channel in range(3):
        channel_data = img_array[:, :, channel]
        c_min = np.min(channel_data)
        c_max = np.max(channel_data)
        
        if c_max > c_min:
            img_array[:, :, channel] = (channel_data - c_min) * (255.0 / (c_max - c_min))
        else:  # fallback when all channel values are identical
            pass
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    if 1:
        for c in range(3):
            channel_data = img_array[:, :, c]
            c_min = np.min(channel_data)
            c_max = np.max(channel_data)
            # allow up to ±3 absolute error
            if abs(c_min-0)>3 or abs(c_max-255)>3:
                print_image_statistics(img_array)
                assert 0
    img_pil = Image.fromarray(img_array)
    return img_pil
            
def imgs_2_grid_A(
    imgs, # list of RGB images (PIL or numpy arrays)
    # any provided mask makes masked pixels lighter across images
    masks=None,
    # if provided, save grid
    grid_path=None,
    # other settings
    downsample=1, # downsample factor for the grid
    inv_mask:bool=False,
    resize_mode:str=None, # None | 'mask_to_img' | 'img_to_mask' (resize img to match mask shape)
    grid_layout:str="row", # 'row' | 'column' |'square'
    auto_pad_if_not_same_size=True,
    verbose :int = 1,
):
    """
    Create a grid of images from paths, optionally with masks overlaid.
    """
    from pathlib import Path
    import PIL.Image
    import numpy as np
    import torchvision.utils as vutils
    import torch
    
    images = []
    for i, img in enumerate(imgs):
        if isinstance(img, PIL.Image.Image):
            pass
        else:
            if verbose>0:
                print(f"{img.shape=}")
            img = to__image_in_npArr(img)
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        # else:
        #     raise TypeError(f"Images must be PIL Image or numpy array{type(img)}")
        
        if not isinstance(img, PIL.Image.Image):
            raise TypeError(f"Images must be PIL Image or numpy array{type(img)}")
            
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)) / 255.0
        
        
        if masks is not None:
            mask = masks[i]
            if isinstance(mask, np.ndarray):
                mask = PIL.Image.fromarray(mask)
                
            if not mask.mode == 'L':
                mask = mask.convert('L')
            
            if resize_mode is None:
                pass
            elif resize_mode == "img_to_mask":
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(mask.height, mask.width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            elif resize_mode == "mask_to_img":
                mask = mask.resize((img_tensor.shape[2], img_tensor.shape[1]), PIL.Image.BILINEAR)
            else:
                raise NotImplementedError
            
            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).repeat(3, 1, 1)
            if inv_mask:
                mask_tensor = 1 - mask_tensor
            # make masked pixels lighter
            img_tensor = img_tensor * 0.3 + 0.7 * mask_tensor
        
        # Apply auto padding if needed
        if auto_pad_if_not_same_size and i > 0 and (img_tensor.shape[1] != images[0].shape[1] or img_tensor.shape[2] != images[0].shape[2]):
            # Resize to match the first image dimensions
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(images[0].shape[1], images[0].shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        images.append(img_tensor)
    
    if grid_layout == "row":
        grid_tensor = vutils.make_grid(images, nrow=len(images), )
    elif grid_layout == "column":
        grid_tensor = vutils.make_grid(images, nrow=1, )
    elif grid_layout == "square":
        grid_tensor = vutils.make_grid(images, nrow=int(np.sqrt(len(images))), )
    else:
        raise NotImplementedError
    
    grid = grid_tensor.numpy().transpose(1, 2, 0)
    grid = PIL.Image.fromarray((grid * 255).astype(np.uint8))
    
    if downsample > 1:
        original_size = grid.size
        new_size = (original_size[0] // downsample, original_size[1] // downsample)
        grid = grid.resize(new_size, PIL.Image.LANCZOS)
    
    if grid_path is not None:
        grid_path = Path(grid_path)
        grid_path.parent.mkdir(parents=False, exist_ok=True)
        grid.save(grid_path)
        if verbose>-1:
            print(f"saved {grid_path}")
    
    return grid

def img_paths_2_grid_A(
    paths, # paths of rgb img
    # any mask option makes masked pixels lighter per image
    mask_paths=None,
    path_img_2_path_mask=None, # callback to convert RGB image path to mask path
    # if provided, save grid
    grid_path=None,
    # other settings
    downsample=1, # downsample factor for the grid
    inv_mask:bool=False,
    resize_mode:str=None, # None | 'mask_to_img' | 'img_to_mask' (resize image to match mask shape)
    grid_layout:str="row", # 'row' | 'column' |'square'
    auto_pad_if_not_same_size=True,
):
    """
    Create a grid of images from paths, optionally with masks overlaid.
    """
    import PIL.Image
    
    # Load images from paths
    imgs = [PIL.Image.open(path).convert('RGB') for path in paths]
    
    # Load masks if provided
    masks = None
    if mask_paths is not None:
        masks = [PIL.Image.open(mask_path).convert('L') for mask_path in mask_paths]
    elif path_img_2_path_mask is not None:
        masks = [PIL.Image.open(path_img_2_path_mask(path)).convert('L') for path in paths]
    
    # Call the img_2_grid_A function
    return imgs_2_grid_A(
        imgs=imgs,
        masks=masks,
        grid_path=grid_path,
        downsample=downsample,
        inv_mask=inv_mask,
        resize_mode=resize_mode,
        grid_layout=grid_layout,
        auto_pad_if_not_same_size=auto_pad_if_not_same_size,
    )



def save_any_A(
    a,
    path=None, # only valid when !dont_save
    dont_save = False,
    # log
    print_info :bool = True,
    value_range: tuple = None,  # (min, max) tuple to specify value range, if None then auto determine
):
    """
    can auto determine or specify by param:
        data shape mode:
            ...,1/3/4,h,w ; ...,h,w,1/3/4 ;  
        value range:
            0-1 ; -1~1 ; 0-255

    after scaling to 0-255, save a grid containing two images:
        scaled image
        contrast-adjusted scaled image via linear transform so min=0 and max=255
    """
    a:np.ndarray = to__image_in_npArr(a)
    a = a.copy()
    if print_info:
        import torch;  from .torch_util import custom_repr_v3
        print(custom_repr_v3(torch.Tensor(a)))
    while(a.ndim>3):
        a=a[0]
    #-----------now a is chw | hwc --------------------------------------------------------
    if a.ndim > 2:
        if a.shape[-3] <= 4:
            if a.shape[-3] <= a.shape[-1] and a.shape[-3] <= a.shape[-2]:
                # assume the -3 axis is the channel dimension; convert chw -> hwc
                a = a.transpose(1, 2, 0)  # chw -> hwc
    else: # ndim==2
        a = np.expand_dims(a, axis=-1)  # hw -> hwc
    #-----------now a is hwc --------------------------------------------------------
    if value_range is None: # Auto determine
        mean = np.mean(a)
        std = np.std(a)
        min_ = np.min(a)
        max_ = np.max(a)
        if a.dtype == np.uint8 or a.dtype == np.int32 or a.dtype == np.int64:
            range_ = (0, 255)
        elif a.dtype == bool:
            range_ = (0, 1)
        elif max_ > 100:
            range_ = (0, 255)
        elif mean > 1:
            range_ = (0, 255)
        elif min_ <= -1 or mean < 0 : # treat as range -1 to 1
            range_ = (-1, 1)
        else: # treat as range 0 to 1
            range_ = (0, 1)
        print(f"Auto determined {range_=}")
    else:
        range_ = value_range
    range_min, range_max = range_
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255  # bool -> 0/255
    else:
        if range_min == 0 and range_max == 255:
            pass
        else:
            # Custom range, normalize to 0~255
            a = (a - range_min) / (range_max - range_min) * 255
    #-----------now a is hwc and scaled to 0~255 --------------------------------------------------------
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    #-----------now a is hwc, 0~255, and channels==3/4 --------------------------------------------------------
    
    if 1:  # create contrast-adjusted version by linearly mapping min to 0 and max to 255
        a_contrast = a.copy().astype(np.float32)
        current_min = np.min(a_contrast)
        current_max = np.max(a_contrast)
        if current_max > current_min:  # avoid division by zero
            a_contrast = (a_contrast - current_min) / (current_max - current_min) * 255
        a = np.clip(a, 0, 255).astype(np.uint8)
        a_contrast = np.clip(a_contrast, 0, 255).astype(np.uint8)
    if dont_save:
        path = None
    else:
        if path is None:
            save_dir = Path("/tmp/scy_auto_save")
            save_dir.mkdir(exist_ok=True)
            import time
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            ext = "jpg" if a.shape[-1] <= 3 else "png" # Use jpg by default if num channels <= 3
            path = save_dir / f"auto_{timestamp}.{ext}"
        else:
            path = Path(path)
            path.parent.mkdir(exist_ok=True)
        path = str(path)
    grid = imgs_2_grid_A( # create grid with 2 images: original scaled + contrast adjusted
        imgs=[a, a_contrast],
        grid_path=path,
        grid_layout="row",
        verbose = -1,
    )
    if not dont_save: print(f"{path}")
    return grid
