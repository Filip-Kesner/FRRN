#! /usr/bin/python

import lasagne
import cv2
import dltools
import theano
import theano.tensor as T
import sys
import numpy as np
import scipy

sys.setrecursionlimit(10000)

config = {
    "num_classes": 19,
    "sample_factor": 4,
    "model_filename": "models/frrn_a.npz",
    "base_channels": 48,
    "fr_channels": 32
}

config["model_filename"] = "models/frrn_a.npz"

input_var = T.ftensor4()

builder = dltools.architectures.FRRNABuilder(
	base_channels=config["base_channels"],
	lanes=config["fr_channels"],
	multiplier=2,
	num_classes=config["num_classes"]
)

network = builder.build(
	input_var=input_var,
	input_shape=(None, 3, 1024 // config["sample_factor"], 2048 // config["sample_factor"])
)

network.load_model(config["model_filename"])

test_predictions = lasagne.layers.get_output(network.output_layers, deterministic=True)[0]

val_fn = theano.function(
	inputs=[input_var],
	outputs=test_predictions
)


while True:
    x = scipy.ndimage.imread("./aachen_000001_000019_leftImg8bit.png", flatten=False, mode=None);
    x =  scipy.misc.imresize(x, 25, interp='bilinear', mode=None)
    x = np.array([np.rollaxis(x, 2, 0)])
    x = x.astype(np.float32)
    x = x / 255.0
    # Process the image
    network_output = val_fn(x)
    # Obtain a prediction
    predicted_labels = np.argmax(network_output[0], axis=0)

    prediction_visualization = dltools.utility.create_color_label_image(predicted_labels)
    ground_truth_visualization = prediction_visualization# dltools.utility.create_color_label_image(t[0])
    image = dltools.utility.tensor2opencv(x[0])

    cv2.imshow("Image", image)
    cv2.imshow("Ground Truth", ground_truth_visualization)
    cv2.imshow("Prediction", prediction_visualization)
    cv2.waitKey()
