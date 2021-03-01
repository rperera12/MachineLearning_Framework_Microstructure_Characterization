# Microstructure Characterization Framework for Pores/Particles & Grain Boundaries
If you find this repository useful and would like to find out more about the implementation, please refer to our manuscript  [Optimized and autonomous machine learning framework for characterizing pores, particles, grains and grain boundaries in microstructural images](https://arxiv.org/abs/2101.06474)

      @misc{perera2021optimized,

      title={Optimized and autonomous machine learning framework for characterizing pores, particles, grains and grain boundaries in microstructural images}, 
      
      author={Roberto Perera and Davide Guzzetti and Vinamra Agrawal},
      
      year={2021},
      
      eprint={2101.06474},
      
      archivePrefix={arXiv},
      
      primaryClass={eess.IV}
      } 

## Models:

### Classifier CNN
NOTE: Found in models.py 
First step to classify images of pores/particles VS grain boundaries. Please refer to the paper to see the type of microstructures characterized. If needed, transfer learning can be implemented to re-train the networks for new datasets. 

### ConvNet_hist
NOTE: Found in models.py
Used to obtain the x-axis for the grain boundary size distribution histograms

### ConvNet_freq
NOTE: Found in models.py
Used to obtain the y-axis for the grain boundary size distribution histograms

### ConvNet_freq
NOTE: Found in models.py
Used to obtain the y-axis for the grain boundary size distribution histograms


### unet
NOTE: Found in models.py
Used for RGB segmentations of grain boundaries


### YOLOv5:
NOTE: The YOLOv5 algorithm is not included in models.py due to copyright. 
      Please refer to the following link for a step-by-step tutorial by the authors: [YOLOv5 Step-by-Step Tutorial](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
      Also, you can find and copy the Google Colab Notebook used of YOLOv5 in the following link:  [YOLOv5 Notebook](https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ)
      The trained weights for the YOLOv5 used in the prediction of Pores/Particles are included in the "Pretrained_weights" directory.
      This file must be loaded in line !python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source ../test/images
      
      
      


```python
import cv2
import math
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import color_masking, plot_masks

image = np.array(Image.open('images/original.png'))

K = 13
result, channel = color_masking(image, K)
combined = plot_masks(192, 192, K, channel)


```

