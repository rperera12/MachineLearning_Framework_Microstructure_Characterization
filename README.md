# Microstructure Characterization Framework for Pores/Particles & Grain Boundaries
If you find this repository useful and would like to find out more about the implementation, please refer to our manuscript 

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

https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/

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
# Results 

### Original Segmented Image

![Original Segmented Image](/images/original.png)
Format: ![Original Segmented Image](url)

### Separated Channels/Classes/Colors using Clusters = 13

![1st Channel Segmented Image](/images/channel1.png) Format: ![Channel 1](url)  ![2nd Channel Segmented Image](/images/channel2.png) Format: ![Channel 2](url)

![3rd Channel Segmented Image](/images/channel3.png) Format: ![Channel 3](url)  ![4th Channel Segmented Image](/images/channel4.png) Format: ![Channel 4](url)

![5th Channel Segmented Image](/images/channel5.png) Format: ![Channel 5](url)  ![6th Channel Segmented Image](/images/channel6.png) Format: ![Channel 6](url)

![7th Channel Segmented Image](/images/channel7.png) Format: ![Channel 7](url)  ![8th Channel Segmented Image](/images/channel8.png) Format: ![Channel 8](url)

![9th Channel Segmented Image](/images/channel9.png) Format: ![Channel 9](url)  ![10th Channel Segmented Image](/images/channel10.png) Format: ![Channel 10](url)

![11th Channel Segmented Image](/images/channel11.png) Format: ![Channel 11](url) ![12th Channel Segmented Image](/images/channel12.png) Format: ![Channel 12](url)

![13th Channel Segmented Image](/images/channel13.png) Format: ![Channel 13](url)

### Finally, all the channels are combined together and displayed below, note & compare the original segmented image to the resultant image shown below

![Original Segmented Image](/images/original.png) Format: ![Original](url)  ![Combined Results Image](/images/combined_results.png)
Format: ![Combined Results](url)
