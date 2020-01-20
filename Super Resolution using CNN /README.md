# Super Resolution using CNNs
In this project, reconstructed higher-resolution images from observed lower-resolution images using Convolutional Neural Networks.

# Data Processing
The input image is used to create blur and sharp training data by upsampling and downsampling it upto 7 images.

![1](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/DataProcessing.png)
The above dimensions mentioned in Table are with respect to RGB images.
The training set consists of shuffled data with input as blurred images and output/ground truth as the difference of sharp and blurred images.

# Model Architecture

Network Architecture for Super Resolution using Convolutional Neural Networks
![2](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/ModelArch.png)

Input Layer:
The input layer consists of image with dimensions (Image_height, Image_width, number of Channels)
The input image is convolved with 64kernels with dimensions (3x3 x number of Channels)

Hidden Layers:
The model consists of 8 hidden layers where each layer is a Convolution+ReLU Layer

Output Layer:
The output layer consists of sharp image whose dimensions are the same as that of the input image (Image_height, Image_width, number of Channels)

The number of channels will vary with the type of input image (RGB = 3, Grayscale = 1)

# Convolution output at 250th epoch
![3](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/ModelArch.png)

# Original Image
![4](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/Original.png)

# Blur Image
![5](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/blurImage.png)

# Convolution output at 250th epoch
![6](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/epoch250.png)

# Sharp Image
![6](https://github.com/Jaisinghani/Machine-Learning/blob/master/Super%20Resolution%20using%20CNN%20/images/sharpImage.png)

# Packages used
<ul>
<li> Numpy</li>
<li> sklearn</li>
</ul>


