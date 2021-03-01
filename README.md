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


### YOLOv5:
The YOLOv5 algorithm is not included in models.py due to copyright. 

Please refer to the following link for a step-by-step tutorial by the authors: [YOLOv5 Step-by-Step Tutorial](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
      
Also, you can find and copy the Google Colab Notebook of YOLOv5 tutorial in the following link:  [YOLOv5 Notebook](https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ)
      
The trained weights for the YOLOv5 used in the prediction of Pores/Particles are included in the "Pretrained_weights" directory.

This file must be loaded in line: 

!python detect.py --weights runs/train/yolov5s_results/weights/yolo_particles.pt --img 416 --conf 0.4 --source ../test/images


### Simple-UNet
NOTE: Found in models.py (UNET)

Used for binary segmentations of particles or pores


### unet
NOTE: Found in models.py 

Used as first comaparion model for RGB segmentations of grain boundaries.

Citation:

      @misc{ronneberger2015unet,
            title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
            author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
            year={2015},
            eprint={1505.04597},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }

### ResNet-UNet
NOTE: Found in models.py (ResNetUNet)

Used as second comaparion model for RGB segmentations of grain boundaries.


### DENSE-UNet
NOTE: Found in models.py (UNet) 

Model used for RGB segmentations of grain boundaries.


### ConvNet_hist
NOTE: Found in models.py

Used to obtain the x-axis for the grain boundary size distribution histograms

### ConvNet_freq
NOTE: Found in models.py

Used to obtain the y-axis for the grain boundary size distribution histograms


## Datasets:
The datasets are not included in the repository due to copyrights, but their descriptions, sources, and corresponding citations will be explained.

### Particles Train Set:
#### Raw Input Images: 
The particles' raw input images used to train the Classifier CNN, Binary Segmentation Simple-UNet, and YOLOv5, were obtained from the paper [A large dataset of synthetic SEM images of powder materials and their ground truth 3D structures](https://www.sciencedirect.com/science/article/pii/S2352340916306382)


      @article{DECOST2016727,
            title = {A large dataset of synthetic SEM images of powder materials and their ground truth 3D structures},
            journal = {Data in Brief},
            volume = {9},
            pages = {727-731},
            year = {2016},
            issn = {2352-3409},
            doi = {https://doi.org/10.1016/j.dib.2016.10.011},
            url = {https://www.sciencedirect.com/science/article/pii/S2352340916306382},
            author = {Brian L. DeCost and Elizabeth A. Holm},
            keywords = {Computer vision, Computational materials science, Image databases, Metal powder, Microstructural analysis},
            abstract = {This data article presents a data set comprised of 2048 synthetic scanning electron microscope (SEM) images of powder materials and descriptions of the corresponding 3D structures that they represent. These images were created using open source rendering software, and the generating scripts are included with the data set. Eight particle size distributions are represented with 256 independent images from each. The particle size distributions are relatively similar to each other, so that the dataset offers a useful benchmark to assess the fidelity of image analysis techniques. The characteristics of the PSDs and the resulting images are described and analyzed in more detail in the research article “Characterizing powder materials using keypoint-based computer vision methods” (B.L. DeCost, E.A. Holm, 2016) [1]. These data are freely available in a Mendeley Data archive “A large dataset of synthetic SEM images of powder materials and their ground truth 3D structures” (B.L. DeCost, E.A. Holm, 2016) located at http://dx.doi.org/10.17632/tj4syyj9mr.1 [2] for any academic, educational, or research purposes.}
            }

The open source dataset can be found in the Mendeley Database link [Mendeley Data: Synthetic Particles](https://data.mendeley.com/datasets/tj4syyj9mr/1): 


#### Binary Segmentation Labels:
Using the Raw Input Images of Particles, the following research article, which uses the Watershed City-Block Distance method can be used in MATLAB to generate the corresponding binary segmentations

      @article{RABBANI2014164,
            title = {An automated simple algorithm for realistic pore network extraction from micro-tomography images},
            journal = {Journal of Petroleum Science and Engineering},
            volume = {123},
            pages = {164-171},
            year = {2014},
            note = {Neural network applications to reservoirs: Physics-based models and data models},
            issn = {0920-4105},
            doi = {https://doi.org/10.1016/j.petrol.2014.08.020},
            url = {https://www.sciencedirect.com/science/article/pii/S0920410514002691},
            author = {Arash Rabbani and Saeid Jamshidi and Saeed Salehi},
            keywords = {micro-tomography images, city-block distance function, watershed segmentation algorithm, pore network extraction},
            abstract = {Using 3-D scanned data to analyze and extract pore network plays a vital role in investigation of porous media׳s characteristics. In this paper, a new simple method is developed to detect pores and throats for analyzing the connectivity and permeability of the network. This automated method utilizes some of the common and well-known image processing functions which are widely accessible by researchers and this has led to an easy algorithm implementation. In this method, after polishing and quality control of images, using city-block distance function and watershed segmentation algorithm, pores and throats are detected and 3-D network is produced. This method can also be applied on 2-D images to extract some characteristics of the porous media such as pore and throat size distribution. The results of network extraction were verified by comparing the distribution of coordination number with a prevalent method in the literature.}
      }

      
      


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

