# -*- coding: utf-8 -*-
"""
MFnet based 3d-conv heatmaps 
tested for pytorch version 0.4
"""
import os
import cv2
import torch
import argparse
import numpy as np
from mfnet_3d import MFNET_3D
from scipy.ndimage import zoom

def center_crop(data, tw=224, th=224):
    h, w, c = data.shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
    return cropped_data

def load_images(frame_dir, selected_frames):
    images = np.zeros((16, 224, 224, 3))
    orig_imgs = np.zeros_like(images)
    for i, frame_name in enumerate(selected_frames):
        im_name = os.path.join(frame_dir, frame_name)
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        scaled_img = cv2.resize(next_image, (256, 256), interpolation=cv2.INTER_LINEAR) # resize to 256x256
        cropped_img = center_crop(scaled_img) # center crop 224x224
        final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        images[i] = final_img
        orig_imgs[i] = cropped_img
        
    torch_imgs = torch.from_numpy(images.transpose(3,0,1,2))
    torch_imgs = torch_imgs.float() / 255.0
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    for t, m, s in zip(torch_imgs, mean_3d, std_3d):
        t.sub_(m).div_(s)
    return np.expand_dims(orig_imgs, 0), torch_imgs.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description='mfnet-base-parser')
    parser.add_argument("num_classes", type=int)
    parser.add_argument("model_weights", type=str)
    parser.add_argument("frame_dir", type=str)
    parser.add_argument("label", type=int)
    parser.add_argument("--base_output_dir", type=str, default=r"visualisations")
    return parser.parse_args()

args = parse_args()

frame_names = os.listdir(args.frame_dir)
frame_indices = list(np.linspace(0, len(frame_names)-1, num=16, dtype=np.int))
selected_frames = [frame_names[i] for i in frame_indices]

RGB_vid, vid = load_images(args.frame_dir, selected_frames)

# load network structure, load weights, send to gpu, set to evaluation mode
model_ft = MFNET_3D(args.num_classes)
model_ft = torch.nn.DataParallel(model_ft).cuda()
checkpoint = torch.load(args.model_weights, map_location={'cuda:1':'cuda:0'})
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.eval()

# get predictions, last convolution output and the weights of the prediction layer
predictions, layerout = model_ft(torch.tensor(vid).cuda())
layerout = torch.tensor(layerout[0].numpy().transpose(1, 2, 3, 0))
pred_weights = model_ft.module.classifier.weight.data.detach().cpu().numpy().transpose()

pred = torch.argmax(predictions).item()

cam = np.zeros(dtype = np.float32, shape = layerout.shape[0:3])
for i, w in enumerate(pred_weights[:, args.label]):

    # Compute cam for every kernel
    cam += w * layerout[:, :, :, i]

# Resize CAM to frame level
cam = zoom(cam, (2, 32, 32)) # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

# normalize
cam -= np.min(cam)
cam /= np.max(cam) - np.min(cam)

# make dirs and filenames
example_name = os.path.basename(args.frame_dir)
heatmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "heatmap")
focusmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "focusmap")
for d in [heatmap_dir, focusmap_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

file = open(os.path.join(args.base_output_dir, example_name, str(args.label), "info.txt"),"a")
file.write("Visualizing for class {}\n".format(args.label))
file.write("Predicted class {}\n".format(pred))
file.close()

# produce heatmap and focusmap for every frame and activation map
for i in range(0, cam.shape[0]):
#   Create colourmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam[i]), cv2.COLORMAP_JET)
#   Create focus map
    focusmap = np.uint8(255*cam[i])
    focusmap = cv2.normalize(cam[i], dst=focusmap, alpha=20, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Create frame with heatmap
    heatframe = heatmap//2 + RGB_vid[0][i]//2
    cv2.imwrite(os.path.join(heatmap_dir,'{:03d}.png'.format(i)), heatframe)
    
#   Create frame with focus map in the alpha channel
    focusframe = RGB_vid[0][i]
    focusframe = cv2.cvtColor(np.uint8(focusframe), cv2.COLOR_BGR2BGRA)
    focusframe[:,:,3] = focusmap
    cv2.imwrite(os.path.join(focusmap_dir,'{:03d}.png'.format(i)), focusframe)
    