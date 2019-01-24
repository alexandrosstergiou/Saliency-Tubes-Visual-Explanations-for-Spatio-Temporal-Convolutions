from keras.models import *
from keras.callbacks import *
import keras.backend as K
from keras.optimizers import SGD
from keras import Model
import cv2
import os
import h5py
import sys
import numpy as np
import argparse
from scipy.ndimage import zoom
from frames import frames_extractor



def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

# For Keras 2.2 bug
K.set_learning_phase(1)

# Define the path of the model's JSON file
model_path = sys.argv[2]
# h5 weights file to load trained model
weight_path = sys.argv[3]
# Path to the video that is to be visualised
vid_path = sys.argv[4]
# Class label for heatmap to be projected based on.
label = int(sys.argv[5])

cam_dir = vid_path.split('.')[0]

# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Crete directory to save frames
if not os.path.exists(cam_dir):
    os.makedirs(cam_dir)

# Make array of stacked frames - By default frames are cropped to 224x224
vid = np.array(frames_extractor(vid_path, 24, 6))

# Get shape of the video
f, w, h, d = vid.shape

vid = np.expand_dims(vid, axis=0)


f = h5py.File(weight_path, 'r')

# List all layers
group = f['model_1']

# The following lines of code were written based on a Keras bug
# The more straightforward alternative is to simple call the
# load_weights() function

# Layer-wise iterations
for elem in group.keys():
    # Only transfer weights that exist in both JSON model and H5 model weights file
    if (elem in layer.name for layer in model.layers):

        # BIAS and WEIGHTS (used for Convolutions)
        b = group[elem].values()[0][()]
        w = group[elem].values()[1][()]

        weights = []
        # Cases for Convolution layers
        if (len(group[elem].values()) == 2):
            weights = [w,b]
        else:
            # Iterate for every dataset element
            for i in range(0,len(group[elem].values())):
                weights.append(group[elem].values()[i][()])

        # Set layer weights
        model.get_layer(elem).set_weights(weights)



# Create Keras model
model = Model(inputs = model.input, outputs = model.output)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer= SGD(lr=0.01, momentum=.9,nesterov=True),metrics=['accuracy'])

# Predict class
y_prob = model.predict(vid)


#Get the y input weights to the softmax (based on the number of filters in the last Conv layer).
class_weights = model.get_layer('predictions').get_weights()[0]
final_conv_layer = get_output_layer(model, 'res5c_branch2c')

# Get outputs for Conv layer
get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
[conv_outputs, predictions] = get_output([vid])
conv_outputs = conv_outputs[0, :, :, :, :]

#Initialise CAM array.
cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:3])

# Get weights only for the specific class
for i, w in enumerate(class_weights[:, label]):

    # Compute cam for every kernel
    cam += w * conv_outputs[:,:, :, i]


cam /= np.max(cam)

# Resize CAM to frame level
cam = zoom(cam, (8, 32, 32))

# Revert video to RGB
RGB_vid = vid * 255

# Print the order of frames (Imagemagic usability)
file = open("order.txt","w")
for i in range(0,cam.shape[0]):
    # Create colourmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam[i]), cv2.COLORMAP_JET)
    # Make regions zero if cases that activation intensity is less than 20%
    heatmap[np.where(cam[i] < 0.2)] = 0

    # Create frame with heatmap
    frame = heatmap*0.5 + RGB_vid[0][i]
    cv2.imwrite(os.path.join(cam_dir,str(i)+'.png'),frame)
    file.write(str(os.path.join(cam_dir, str(i)+'.png'))+'\n')
file.close()
