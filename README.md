# Machine Learning Framework for Microstructure Characterization of Pores/Particles & Grain Boundaries
Should you find this repository as a useful tool for research or application, please kindly cite the original article [Optimized and autonomous machine learning framework for characterizing pores, particles, grains and grain boundaries in microstructural images](https://arxiv.org/abs/2101.06474)

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

Used as first comparion model for RGB segmentations of grain boundaries.

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

Please, once you have gathered the datasets place them in their respective directories as follows:

Classifier Pores and Grains Images + Labels:        --->   /data/classifier_pores_grains/

Raw Particles' Input Images:        --->   /data/pore_particle/train/

Particles' Binary Segmentations:    --->   /data/pore_particle/seg_label/

Particles' Bounding Boxes:          --->   /data/pore_particle/box_label/

Raw Grains' Input Images:           --->   /data/grain/train/

Grains' RGB Segmentations:          --->   /data/grain/seg_label/

Grains' Size Radii and Frequency:                 --->   /data/grain/histogram_label/

### Particles Train Set:
#### Raw Input Images: 
The synthetic particles' raw input images used to train the Classifier CNN, Binary Segmentation Simple-UNet, and YOLOv5, were obtained from the following research article by Brian L. DeCost, and Elizabeth A.Holm [A large dataset of synthetic SEM images of powder materials and their ground truth 3D structures](https://www.sciencedirect.com/science/article/pii/S2352340916306382)


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

### Particles Label Set:
#### Binary Segmentation Labels:
Using the Raw Input Images of Particles, the following research article by Arash Rabbani, Saeid Jamshidi, Saeed Salehi, which uses the Watershed City-Block Distance Method (WCBD) can be used along with MATLAB to generate the corresponding binary segmentations: [An automated simple algorithm for realistic pore network extraction from micro-tomography images](https://www.sciencedirect.com/science/article/pii/S0920410514002691)

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

Check out the following Tutorials provided by the author to implement the WCBD algorithm and obtain the label set of binary segmentations: [Porous Material 101 via MATLAB by Arash Rabbani](https://www.youtube.com/playlist?list=PLaYes2m4FtR3DBM7TIb6oOZYI-tG4fHLd)      


#### Bounding Boxes Labels:
In order to obtain the bounding boxes used to train the YOLOv5 algorithm, the MATLAB code provided above for the WCBD method was adjusted to output and save the detected x and y coordinates of the binary segmentations along with their diameters/radii values. Check out the attached Tutorials above for more details on generating the bounding boxes

A very important point to make is that the online open source computer vision tool ["Roboflow.ai"](https://roboflow.com/), was used to generate the bounding boxes label set in the format required by YOLOv5. This dataset is not included due to copyrights. 


### Grain Boundaries Train Set:
#### Raw Input Images: 
The SLM 316L stainless steel grain boundaries' raw input images used to train the Classifier CNN, RGB Segmentation networks (U-Net, ResNet-Unet, and DENSE-UNet), and Regression CNNs, were obtained from the following research article by Xinwei Li and Habimana Jean Willy [Selective laser melting of stainless steel and alumina composite: Experimental and simulation studies on processing parameters, microstructure and mechanical properties](https://www.sciencedirect.com/science/article/pii/S0264127518301412)

NOTE: These high-resolution images were resized to 2048x2048, and cropping each section into separate 256x256 images, a total of 3776 training images was obtained

      @article{LI20181,
            title = {Selective laser melting of stainless steel and alumina composite: Experimental and simulation studies on processing parameters, microstructure and mechanical properties},
            journal = {Materials & Design},
            volume = {145},
            pages = {1-10},
            year = {2018},
            issn = {0264-1275},
            doi = {https://doi.org/10.1016/j.matdes.2018.02.050},
            url = {https://www.sciencedirect.com/science/article/pii/S0264127518301412},
            author = {Xinwei Li and Habimana Jean Willy and Shuai Chang and Wanheng Lu and Tun Seng Herng and Jun Ding},
            keywords = {Selective laser melting, Stainless steel, Alumina, Finite element modeling, Metal matrix composite, Microlattice},
            abstract = {Metal matrix composites (MMC) find their uses as high performance materials. The selective laser melting (SLM) of a 316L stainless steel and Al2O3 MMC is presented in this paper. Agglomerate Al2O3 particles had shown to be an adequate powder choice with uniform dispersions in the resultant prints. Relative density, phase, microstructure and mechanical properties of all 1-, 2-, 3-wt% doped products were carefully analyzed. Finite element modeling model was developed to study the associated multi-physics phenomena with high efficiency for process parameter optimization. It is found that the change in SLM temperature profile with Al2O3 addition is mainly due to the change in optical properties rather than thermal. Hence, both simulation and experimentation revealed that higher laser energy input is needed for optimized melting. In addition, cellular dendrites were found to coarsen with increasing Al2O3 addition due to the decreased cooling rate. With hard particle strengthening effects, all samples showed improved hardness with 3-wt% up to 298HV and 1-wt% samples showing much improved yielding and tensile stresses of 579 and 662MPa from 316L. Corresponding microlattice built this way demonstrated a 30 and 23% increase in specific strength and energy absorption from that of 316L too.}
      }
      
  
#### RGB Segmentation and Size Distribution Labels :
Using the Raw Input Images (see above), the following ASTM reference, which uses and justifies the Point-Sampled Intercept Length Method can be used along with MATLAB to generate the corresponding RGB segmentations and grain boundary size distribution:

ASTM Standard Point-Sampled Intercept Length Method:

      @book{ASTM2015Standard,
            title         = "{Standard test methods for determining average grain size
                             using semiautomatic and automatic image analysis}",
            journal       = "{ASTM Journal}",                       
            publisher     = "ASTM",
            author        = "American Society for Testing and Materials. Philadelphia",
            address       = "West Conshohocken, PA",
            year          = "2015",
            url           = "http://cds.cern.ch/record/1463023",
            note          = "Reapproved in 2015",
      }
   
   
   
Point-Sampled Intercept Length Method research articles by Pauli Lehto, Heikki Remes, Tapio Saukkonen, Teemu Sarikka, Hannu Hänninen and Jani Romanoff: 

      @article{Lehto2014Influence,
            title = "Influence of grain size distribution on the Hall–Petch relationship of welded structural steel",
            journal = "Materials Science and Engineering: A",
            volume = "592",
            pages = "28 - 39",
            year = "2014",
            issn = "0921-5093",
            doi = "https://doi.org/10.1016/j.msea.2013.10.094",
            url = "http://www.sciencedirect.com/science/article/pii/S0921509313012094",
            author = "Pauli Lehto and Heikki Remes and Tapio Saukkonen and Hannu Hänninen and Jani Romanoff",
            keywords = "Grain size, Hall–Petch relationship, Hardness, Strength, Steel, Welding"
      }
      
      
      @article{Lehto2016Characterization,
            title = "Characterisation of local grain size variation of welded structural steel",
            journal = "Welding in the World",
            volume = "60",
            pages = "673 - 688",
            year = "2016",
            issn = "1878-6669",
            doi = "10.1007/s40194-016-0318-8",
            url = "https://doi.org/10.1007/s40194-016-0318-8",
            author = "Pauli Lehto and Heikki Remes and Teemu Sarikka and Jani Romanoff"
      }

Check out the open source MATLAB Codes by Pauli Lehto, Heikki Remes, Tapio Saukkonen, Teemu Sarikka, Hannu Hänninen and Jani Romanoff [PSILM](https://wiki.aalto.fi/display/GSMUM/Characterization+of+local+grain+size+variation)


## Pretrained Models/Weights
The pretrained weights and models can be found at https://drive.google.com/drive/folders/1QebmbTBFOvzmRISkUmkYTzFiQEpCnbsD

Please, download the required pretrained weights and save them in the "/Pretrained_weights/" directory.

Classifier CNN   -  <pore_vs_grain_weights.pth>

Simple-UNet      -  <simple_unet_weights.pt>

YOLOv5           -  <yolo_particles_weights.pt>

U-Net            -  <unet_weights.pth>

ResNet_UNet      -  <ResNet_UNet_weights.pth>

DENSE_UNet       -  <DENSE_UNet_weights.pth>

Grain Frequency  -  <histogram_freq_weights.pth> 

Grain Radii      -  <histogram_rad_weights.pth> 



## Training
If retraining of the models or transfer learning is needed for your own dataset, you can use the training files along with their bash:

#### train_classifier.py -   Classifier CNN trainer
Command: bash run_scripts/train_classifier.sh

#### train_Binary.py     -   Binary Segmentation (Simple-UNet) trainer
Command: bash run_scripts/train_Binary.sh

#### train_RGB.py        -   RGB Segmentation (DENSE-UNet) trainer
Command: bash run_scripts/train_RGB.sh

#### train_hist.py       -   Histogram Predictor trainer
Command: bash run_scripts/train_his.sh

## Notes:
I am in the process of creating a series of open-source Youtube tutorial videos for a step-by-step description of the entire framework.

The current repository can be used with transfer learning to extend the framework's capabilities and accuracy for other material microstructures.

The DENSE-UNet network shows reduction in training time and GPU usage by half, compared to the standard U-Net and ResNet-UNet. 

If you have any questions, I'd be happy to offer additional help! please email me at rzp0063@auburn.edu 

#### Due to Copyright from the YOLOv5 algorithm, I am not able to provide a full tutorial for running the pore characterization process, but if you follow the notebook tutorial along with the original paper [Optimized and autonomous machine learning framework for characterizing pores, particles, grains and grain boundaries in microstructural images](https://arxiv.org/abs/2101.06474), the steps for the complete set up are explained .

### Please check out the additional provided notebooks for a quick tutorial on setting up and running the complete framework. 
