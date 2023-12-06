import torch
import os
os.chdir("/home/jesperdn/repositories/SuperSynth")

import ext.graph.config as config
from ext.graph.models import TopoFitGraph


features = torch.randn((1,64,100,100,100))
vertices = 10*torch.randn((1,62,3)) + 50
print("vertices min", vertices.amin(1))
print("vertices max", vertices.amax(1))

in_channels = 64


model = TopoFitGraph(
    in_channels,
    prediction_res = 6,
    config=config.TopoFitModelParameters,
)

pred = model(features, vertices)
