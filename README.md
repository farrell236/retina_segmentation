# Retina Vessel Segmentation

TODO

<div align="center">

|          Fundus Image          |        Predicted Vessel Segmentation        | 
|:------------------------------:|:-------------------------------------------:|
| ![](assets/ret-hem250-304.jpg) | ![](assets/ret-hem250-304_segmentation.jpg) |

</div>

Test image obtained from: https://www.opsweb.org/page/fundusimaging

### Segmentation Pipeline

TODO

### Results

DeepLabV3+ Model trained on DRIVE dataset only (with 5-Fold Cross Validation).

|                   |     Dice      |  Sensitivity  |  Specificity  |      AUC      |
|:------------------|:-------------:|:-------------:|:-------------:|:-------------:|
| DRIVE (1stHO)     | 0.757 (0.001) | 0.795 (0.005) | 0.974 (0.000) | 0.963 (0.000) | 
| DRIVE (2ndHO)     | 0.766 (0.001) | 0.814 (0.007) | 0.974 (0.001) | 0.968 (0.001) | 
| CHASE_DB1 (1stHO) | 0.699 (0.024) | 0.774 (0.051) | 0.972 (0.004) | 0.955 (0.010) | 
| CHASE_DB1 (2ndHO) | 0.687 (0.021) | 0.756 (0.080) | 0.971 (0.007) | 0.952 (0.013) | 
| STARE (ah)        | 0.726 (0.004) | 0.724 (0.021) | 0.977 (0.002) | 0.940 (0.004) |  
| STARE (vk)        | 0.701 (0.007) | 0.662 (0.025) | 0.982 (0.002) | 0.919 (0.008) | 

`()` denotes Confidence Interval at 95%

### Run Model
```
usage: run_segmentation.py [-h] [-i INPUT_FN] [-o OUTPUT_FN] [-m MODEL_FN] [-v]

Retina Vessel Segmentation

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FN, --input_fn INPUT_FN
                        Input Retina Image
  -o OUTPUT_FN, --output_fn OUTPUT_FN
                        Output Segmentation
  -m MODEL_FN, --model_fn MODEL_FN
                        Trained Model
  -v, --verbose         Verbose Output
```

### Requirements

```
Python 3.7.11
```

### Packages:

```
matplotlib==3.5.2
numpy==1.21.6
opencv-contrib-python-headless==4.6.0.66
opencv-python-headless==4.6.0.66
pandas==0.25.2
tensorflow==2.9.1
tensorflow-addons==0.17.1
tqdm==4.36.1
```

### References

```
TODO
```
