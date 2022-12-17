# RGB-D Semantic Segmentation with UNet and ENet

## Authors

[@fvolcic](www.github.com/fvolcic)

[@dlugoszj](www.github.com/dlugosz)

[@broderio](https://github.com/broderio)

[@dhilpickle21](https://github.com/dhilpickle21)

## Project Background
The goal of the project was to figure out the difference in the accuracy of semantic segmentation when using RGB data and RGB-D data. We used an implementation of ENet and UNet to test the differences. We trained the networks based on a combination of Dice Loss and Cross Entropy Loss. From here we calculated the accuracy of the networks by using the mean IoU (Intersection over Union) and mean pixel accuracy. 

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

## Installation

To setup the environment, run 
 
```bash
pip3 install -r requirements.txt
```
## Usage

Once installed, you can train the networks using any of the train files. You can visualize the results with the associated ipynb files. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
