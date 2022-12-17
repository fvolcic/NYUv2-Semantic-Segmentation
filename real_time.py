import freenect
import cv2
import torch
import numpy as np
import utils.utils as utils

from torchvision import transforms
from models.basic_unet import Unet

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 512

MODEL = Unet(True) 
MODEL_PTH_PATH = "models_pth_files/cpu_depth_unet_epoch_400.pth"

to_tensor = transforms.ToTensor()
resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), transforms.InterpolationMode.NEAREST)

def load_model(model, model_path):
    """Load model from path"""
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_depth():
    """Get depth from kinect"""
    depth = freenect.sync_get_depth()[0] / 1e4
    depth = to_tensor(depth) 
    return depth

def get_frame():
    """Get frame from kinect"""
    frame = freenect.sync_get_video()[0]
    frame = to_tensor(frame)
    return frame

def process_kinect_frame(frame):
    """Process kinect frame"""
    frame = resize(frame)
    frame = frame.unsqueeze(0)
    return frame

def predict(model, frame):
    """Predict mask from frame"""
    with torch.no_grad():
        model_out = model(frame)
        pred_mask = pred_mask = utils.convert_to_segmentation(model_out)
        pred_mask = pred_mask.squeeze(0)
        pred_mask = pred_mask.numpy()
        return pred_mask

def label2rgb(label, num_labels):
    """Convert label to rgb"""
    label_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [0, 255, 255],
        4: [255, 0, 0],
        5: [255, 0, 255],
        6: [255, 255, 0],
        7: [255, 255, 255],
        8: [0, 0, 128],
        9: [0, 128, 0],
        10: [0, 128, 128],
        11: [128, 0, 0],
        12: [128, 0, 128],
        13: [128, 128, 0],
        14: [128, 128, 128]
    }
    return np.array(label_map[label])

def segmentation_to_rgb(segmentation):
    """Convert segmentation to rgb"""
    rgb = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            rgb[i, j] = label2rgb(segmentation[i, j], 14)
    return rgb

def colorize_depth(depth_map):
    """Colorize depth map"""
    #depth_map = (depth_map - depth_map.min()) / depth_map.max()
    depth_map = depth_map * 400 * 4.5
    depth_map = np.clip(depth_map, 0, 255)
    depth_map = depth_map.squeeze(0)
    depth_map = depth_map.astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)

    # apply gaussian blur
    depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)

    return depth_map

def concat_images(img1, img2):
    """Concatenate two images"""
    return np.concatenate((img1, img2), axis=1)

def main():
    """Main function"""
    model = load_model(MODEL, MODEL_PTH_PATH).to(device) 
    while True:
        frame = get_frame().to(device) 
        depth = get_depth().to(device)
        frame_processed = process_kinect_frame(frame)
        depth_processed = process_kinect_frame(depth)
        model_in = torch.cat((frame_processed, depth_processed), 1).float()
        segmentation = predict(model, model_in)
        segmentation = segmentation_to_rgb(segmentation)
        frame = resize(frame).numpy().transpose(1, 2, 0)
        depth_col = colorize_depth(resize(depth).numpy())
        cv2.imshow("depth_grey", depth_col)
        cv2.imshow("out", frame)
        cv2.imshow("segmentation", segmentation)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()