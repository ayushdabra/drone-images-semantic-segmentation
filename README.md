<p>
  <a href="https://colab.research.google.com/drive/1J-PQgIJWOCb7hoc4eiOCna1gUf0cnMCW">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

  <a href="https://mybinder.org/v2/gh/ayushdabra/drone-images-semantic-segmentation/HEAD">
    <img src="https://mybinder.org/badge_logo.svg" alt="launch binder"/>
  </a>
</p>

# Multiclass Semantic Segmentation of Aerial Drone Images Using Deep Learning

## Abstract

<p align="justify">
Semantic segmentation is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. In this project, I have performed semantic segmentation on <a href="http://dronedataset.icg.tugraz.at/">Semantic Drone Dataset</a> by using transfer learning on a VGG-16 backbone (trained on ImageNet) based UNet CNN model. In order to artificially increase the amount of data and avoid overfitting, I preferred using data augmentation on the training set. The model performed well, and achieved ~87% dice coefficient on the validation set.</p>

## Tech Stack

|<a href="https://www.python.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/python.png" /></p></a>|<a href="https://jupyter.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/jupyter.png" /></p></a>|<a href="https://ipython.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/IPython.png" /></p></a>|<a href="https://numpy.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/numpy.png" /></p></a>|<a href="https://pandas.pydata.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/pandas.png" /></p></a>|
|---|---|---|---|---|

|<a href="https://matplotlib.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/matplotlib.png" /></p></a>|<a href="https://opencv.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/opencv.png" /></p></a>|<a href="https://albumentations.ai/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/albumentations.png" /></p></a>|<a href="https://keras.io/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/keras.png" /></p></a>|<a href="https://www.tensorflow.org/"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/tensorflow.png" /></p></a>|<a href="https://github.com/philipperemy/keract"><p align="center"><img width = "auto" height= "auto" src="./tech_stack/keract.png" /></p></a>|
|---|---|---|---|---|---|

The Jupyter Notebook can be accessed from <a href="./semantic-drone-dataset-vgg16-unet.ipynb">here</a>.

## What is Semantic Segmentation?

<p align="justify">
Semantic segmentation is the task of classifying each and very pixel in an image into a class as shown in the image below. Here we can see that all persons are red, the road is purple, the vehicles are blue, street signs are yellow etc.</p>

<p align="center">
<img src="https://miro.medium.com/max/750/1*RZnBSB3QpkIwFUTRFaWDYg.gif" />
</p>

<p align="justify">
Semantic segmentation is different from instance segmentation which is that different objects of the same class will have different labels as in person1, person2 and hence different colours.</p>

<p align="center">
<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/03/Screenshot-from-2019-03-28-12-08-09.png" />
</p>

<!-- ## Applications

1.  **Medical Images**

    <p align="justify">Automated segmentation of body scans can help doctors to perform diagnostic tests. For example, models can be trained to segment tumor.</p>

    <p align="center"> <img src="https://research.nvidia.com/sites/default/files/publications/1111.png" /> </p>

2.  **Autonomous Vehicles**

    <p align="justify">Autonomous vehicles such as self-driving cars and drones can benefit from automated segmentation. For example, self-driving cars can detect drivable regions.</p>

    <p align="center"> 
    <img  width="600" height="323" src="https://divamgupta.com/assets/images/posts/imgseg/image10.png?style=centerme" /> 
    </p>

3.  **Satellite Image Analysis**

    <p align="justify">Aerial images can be used to segment different types of land. Automated land mapping can also be done.</p>

    <p align="center"> 
    <img  width="600" height="200" src="https://www.spiedigitallibrary.org/ContentImages/Journals/JARSC4/12/4/042804/FigureImages/JARS_12_4_042804_f003.png" /> 
    </p> -->

## Semantic Drone Dataset

<p align="justify">The <a href="http://dronedataset.icg.tugraz.at/">Semantic Drone Dataset</a> focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images and the test set is made up of 200 private images.</p>

<p align="center">
  <img width="600" height="423" src="https://www.tugraz.at/fileadmin/_migrated/pics/fyler3.png" />
</p>

<br>

### Semantic Annotation

The images are labeled densely using polygons and contain the following 24 classes:

<!-- | Name        | R   | G   | B   | Color                                                                                           |
| ----------- | --- | --- | --- | ----------------------------------------------------------------------------------------------- |
| unlabeled   | 0   | 0   | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,0,0)" /></svg>       |
| paved-area  | 128 | 64  | 128 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(128,64,128)" /></svg>  |
| dirt        | 130 | 76  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(130,76,0)" /></svg>    |
| grass       | 0   | 102 | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,102,0)" /></svg>     |
| gravel      | 112 | 103 | 87  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(112,103,87)" /></svg>  |
| water       | 28  | 42  | 168 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(28,42,168)" /></svg>   |
| rocks       | 48  | 41  | 30  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(48,41,30)" /></svg>    |
| pool        | 0   | 50  | 89  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,50,89)" /></svg>     |
| vegetation  | 107 | 142 | 35  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(107,142,35)" /></svg>  |
| roof        | 70  | 70  | 70  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(70,70,70)" /></svg>    |
| wall        | 102 | 102 | 156 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(102,102,156)" /></svg> |
| window      | 254 | 228 | 12  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(254,228,12)" /></svg>  |
| door        | 254 | 148 | 12  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(254,148,12)" /></svg>  |
| fence       | 190 | 153 | 153 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(190,153,153)" /></svg> |
| fence-pole  | 153 | 153 | 153 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(153,153,153)" /></svg> |
| person      | 255 | 22  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(255,22,0)" /></svg>    |
| dog         | 102 | 51  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(102,51,0)" /></svg>    |
| car         | 9   | 143 | 150 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(9,143,150)" /></svg>   |
| bicycle     | 119 | 11  | 32  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(119,11,32)" /></svg>   |
| tree        | 51  | 51  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(51,51,0)" /></svg>     |
| bald-tree   | 190 | 250 | 190 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(190,250,190)" /></svg> |
| ar-marker   | 112 | 150 | 146 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(112,150,146)" /></svg> |
| obstacle    | 2   | 135 | 115 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(2,135,115)" /></svg>   |
| conflicting | 255 | 0   | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(255,0,0)" /></svg>     | -->

| Name        | R   | G   | B   | Color                                                                                        |
| ----------- | --- | --- | --- | -------------------------------------------------------------------------------------------- |
| unlabeled   | 0   | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/unlabeled.png" /></p>   |
| paved-area  | 128 | 64  | 128 | <p align="center"><img width = "30" height= "20" src="./label_colors/paved-area.png" /></p>  |
| dirt        | 130 | 76  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dirt.png" /></p>        |
| grass       | 0   | 102 | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/grass.png" /></p>       |
| gravel      | 112 | 103 | 87  | <p align="center"><img width = "30" height= "20" src="./label_colors/gravel.png" /></p>      |
| water       | 28  | 42  | 168 | <p align="center"><img width = "30" height= "20" src="./label_colors/water.png" /></p>       |
| rocks       | 48  | 41  | 30  | <p align="center"><img width = "30" height= "20" src="./label_colors/rocks.png" /></p>       |
| pool        | 0   | 50  | 89  | <p align="center"><img width = "30" height= "20" src="./label_colors/pool.png" /></p>        |
| vegetation  | 107 | 142 | 35  | <p align="center"><img width = "30" height= "20" src="./label_colors/vegetation.png" /></p>  |
| roof        | 70  | 70  | 70  | <p align="center"><img width = "30" height= "20" src="./label_colors/roof.png" /></p>        |
| wall        | 102 | 102 | 156 | <p align="center"><img width = "30" height= "20" src="./label_colors/wall.png" /></p>        |
| window      | 254 | 228 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/window.png" /></p>      |
| door        | 254 | 148 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/door.png" /></p>        |
| fence       | 190 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence.png" /></p>       |
| fence-pole  | 153 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence-pole.png" /></p>  |
| person      | 255 | 22  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/person.png" /></p>      |
| dog         | 102 | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dog.png" /></p>         |
| car         | 9   | 143 | 150 | <p align="center"><img width = "30" height= "20" src="./label_colors/car.png" /></p>         |
| bicycle     | 119 | 11  | 32  | <p align="center"><img width = "30" height= "20" src="./label_colors/bicycle.png" /></p>     |
| tree        | 51  | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/tree.png" /></p>        |
| bald-tree   | 190 | 250 | 190 | <p align="center"><img width = "30" height= "20" src="./label_colors/bald-tree.png" /></p>   |
| ar-marker   | 112 | 150 | 146 | <p align="center"><img width = "30" height= "20" src="./label_colors/ar-marker.png" /></p>   |
| obstacle    | 2   | 135 | 115 | <p align="center"><img width = "30" height= "20" src="./label_colors/obstacle.png" /></p>    |
| conflicting | 255 | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/conflicting.png" /></p> |

### Sample Images

<p align="center"><img width = "95%" height= "auto" src="./sample_images/image_002.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./sample_images/image_001.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./sample_images/image_004.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./sample_images/image_003.jpg" /></p>

## Technical Approach

### Data Augmentation using Albumentations Library

<p align="justify"><a href="https://albumentations.ai/">Albumentations</a> is a Python library for fast and flexible image augmentations. Albumentations efficiently implements a rich variety of image transform operations that are optimized for performance, and does so while providing a concise, yet powerful image augmentation interface for different computer vision tasks, including object classification, segmentation, and detection.</p>

<p align="justify">There are only 400 images in the dataset, out of which I have used 320 images (80%) for training set and remaining 80 images (20%) for validation set. It is a relatively small amount of data, in order to artificially increase the amount of data and avoid overfitting, I preferred using data augmentation. By doing so I have increased the training data upto 5 times. So, the total number of images in the training set is 1600, and 80 images in the validation set, after data augmentation.</p>

Data augmentation is achieved through the following techniques:

- Random Cropping
- Horizontal Flipping
- Vertical Flipping
- Rotation
- Random Brightness & Contrast
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Grid Distortion
- Optical Distortion

Here are some sample augmented images and masks of the dataset:

<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_image_376.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_mask_376.png" /></p>

<br>

<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_image_277.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_mask_277.png" /></p>

<br>

<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_image_118.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_mask_118.png" /></p>

<br>

<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_image_092.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./augmented_images/aug_mask_092.png" /></p>

### VGG-16 Encoder based UNet Model

<p align="justify">The <a href="https://arxiv.org/abs/1505.04597">UNet</a> was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus, it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.</p>

In the original paper, the UNet is described as follows:

<p align="center">
<img width = "800" height= "auto" src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" />
</p>

<p align="center"><i>U-Net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.</i></p>

#### Custom VGG16-UNet Architecture

- VGG16 model pre-trained on the ImageNet dataset has been used as an Encoder network.

- A Decoder network has been extended from the last layer of the pre-trained model, and it is concatenated to the consecutive convolution blocks.

<p align="center">
<img width = "90%" height= "auto" src="./vgg16_unet.png" />
</p>
<p align="center"><i>VGG16 Encoder based UNet CNN Architecture</i></p>

A detailed layout of the model is available [here](./vgg16_unet_plot.png).

### Hyper-Parameters

1. Batch Size = 8
2. Steps per Epoch = 200.0
3. Validation Steps = 10.0
4. Input Shape = (512, 512, 3)
5. Initial Learning Rate = 0.0001 (with Exponential Decay LearningRateScheduler callback)
6. Number of Epochs = 20 (with ModelCheckpoint & EarlyStopping callback)

## Results

### Training Results

|   Model    |                   Epochs                   | Train Dice Coefficient | Train Loss | Val Dice Coefficient | Val Loss |    Max. (Initial) LR    |         Min. LR         | Total Training Time |
| :--------: | :----------------------------------------: | :--------------------: | :--------: | :------------------: | :------: | :---------------------: | :---------------------: | :-----------------: |
| VGG16-UNet | 20 (best weights at 18<sup>th</sup> epoch) |         0.8781         |   0.2599   |        0.8702        | 0.29959  | 1.000 × 10<sup>-4</sup> | 1.122 × 10<sup>-5</sup> | 23569 s (06:32:49)  |

<p align="center"><img width = "auto" height= "auto" src="./model_metrics_plot.png" /></p>

The <a href="https://github.com/ayushdabra/drone-images-semantic-segmentation/blob/main/model_training_csv.log">`model_training_csv.log`</a> file contain epoch wise training details of the model.

### Visual Results

Predictions on Validation Set Images:

<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_21.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_23.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_49.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_24.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_58.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_28.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_55.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_60.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_69.jpg" /></p>
<p align="center"><img width = "auto" height= "auto" src="./predictions/compressed/prediction_73.jpg" /></p>

All predictions on the validation set are available in the <a href="https://github.com/ayushdabra/drone-images-semantic-segmentation/tree/main/predictions">`predictions`</a> directory.

## Activations (Outputs) Visualization

Activations/Outputs of some layers of the model-

| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/1_block1_conv1.png" /><b>block1_conv1</b></p>    | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/4_block4_conv1.png" /><b>block4_conv1</b></p>              | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/6_conv2d_transpose.png" /><b>conv2d_transpose</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/7_concatenate.png" /><b>concatenate</b></p>                |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/8_conv2d.png" /><b>conv2d</b></p>                | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/10_conv2d_transpose_1.png" /><b>conv2d_transpose_1</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/13_conv2d_3.png" /><b>conv2d_3</b></p>                | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/14_conv2d_transpose_2.png" /><b>conv2d_transpose_2</b></p> |
| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/15_concatenate_2.png" /><b>concatenate_2</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/17_conv2d_5.png" /><b>conv2d_5</b></p>                     | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/21_conv2d_7.png" /><b>conv2d_7</b></p>                | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/22_conv2d_8.png" /><b>conv2d_8</b></p>                     |

Some more activation maps are available in the <a href="https://github.com/ayushdabra/drone-images-semantic-segmentation/tree/main/activations">`activations`</a> directory.

## References

1. Semantic Drone Dataset- http://dronedataset.icg.tugraz.at/
2. Karen Simonyan and Andrew Zisserman, "**Very Deep Convolutional Networks for Large-Scale Image Recognition**", arXiv:1409.1556, 2014. [\[PDF\]](https://arxiv.org/pdf/1409.1556v6.pdf)
3. Olaf Ronneberger, Philipp Fischer and Thomas Brox, "**U-Net: Convolutional Networks for Biomedical Image Segmentation**", arXiv:1505. 04597, 2015. [\[PDF\]](https://arxiv.org/pdf/1505.04597.pdf)
4. Towards Data Science- [Understanding Semantic Segmentation with UNET](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47), by Harshall Lamba
5. Keract by Philippe Rémy [\(@github/philipperemy\)](https://github.com/philipperemy/keract) used under the IT License Copyright (c) 2019.
