# RGB-D Semantic Segmentation with UNet and ENet

## Authors

[@fvolcic](www.github.com/fvolcic)

[@dlugoszj](www.github.com/dlugosz)

[@broderio](https://github.com/broderio)

[@dhilpickle21](https://github.com/dhilpickle21)

## Project Background
Image segmentation is the process by which the pixels of an image are partitioned into multiple segments based on shared characteristics that can include color, texture, and intensity. The goal of semantic image segmentation is to predict a class for every pixel in an image. Semantic image segmentation has important applications in fields such as medical imaging and autonomous vehicles.

This project aimed to investigate the use of depth information in semantic image segmentation. We evaluate the performance of two network architectures, Efficient Network (ENet) and a UNet with skip connections, on the task of semantic image segmentation using both RGB and RGB-D images. Our experiments were conducted using the NYUv2 dataset. A combination of Dice Loss and Cross Entropy Loss were used to stabilize the losses. Our results demonstrate the benefits of incorporating depth information in improving the accuracy of semantic image segmentation, showing that Depth gave an improvement of up to 17% mean IoU (intersection over union) on the NYUv2 test set. We acheived our best results when allowing each network to train for 400 epochs. 

The reference paper for this repository is linked [here](https://github.com/fvolcic/NYUv2-Semantic-Segmentation/blob/main/report.pdf).

## Model Example

Below showcases our model running on the NYUv2 dataset. The first figure showcases our models output, while the second showcases the input and the ground truth mask. 

Model outputs

<img src="https://user-images.githubusercontent.com/59806465/207989873-6b0ea379-3948-41c8-916c-cb4f1175e46a.png" alt="drawing" width="400"/>

Expected outputs

<img src="https://user-images.githubusercontent.com/59806465/207989889-04eefe63-d989-4518-834c-ef59e3a4aaab.png" alt="drawing" width="400"/>

## Model Results

Our results indicate that depth has the ability to significantly improve semantic segmentation results. While ENet only saw a marginal improvement, but our UNet model saw an improvement of nearly 17% in mean IoU, giving an mIoU of about 48%. Our results are tabulated below. 

<img src="https://user-images.githubusercontent.com/59806465/208209542-35250bb6-b105-421e-a338-8c8759100b9d.png" alt="drawing" width="400"/>

## Real Time Model View 

In addition to proving a number of model testing utilities, this repo also provides the ability to view your models working in real time. The python script, named real_time.py, is uses freenect to interface with an xbox kinect V1. Install freenect to your system, then run the program and watch the magic happen!

Example of the model running in real time: 

<img src="https://user-images.githubusercontent.com/59806465/208266247-ae83ea7d-8d33-43f9-823a-6f712f58f0cb.png" alt="drawing" width="800"/>

## Installation

To setup the environment, run 
 
```bash
pip3 install -r requirements.txt
```
## Usage

Once installed, you can train the networks using any of the train files. You can visualize the results with the associated ipynb files. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
