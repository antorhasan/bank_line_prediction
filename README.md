# River Bank Line Prediction from Historical Multichannel Landsat Satellite Images

## Objectives :
Very few studies/researches have carried out previously on bank erosion prediction solely based on remote sensing data. The techniques of deep learning have proven to be effective in a vast domain of image related problems. The use of such methods along with the availability of historical satellite data, hold potential in solving prediction tasks related to water using remote sensing data. 

For this project Jamuna River in Bangladesh was chosen to be the study area. As far as we have explored there are no existing ground truth bank line data given a Landsat image. And so the first objective of this project is to make a comprehensive dataset for solving river bank line prediction problem for our study area. 

The second task is to develop a baseline model architecture using the dataset and perform subsequent experiments to improve the performance. This task involves exploring a wide variety of both preprocessed input output data and domain specific network architectures.

The final task is to primarily notice the effect of multichannel Landsat images as opposed to only using the RGB channels of the data and by extension attempt to incorporate numerical modelling results to notice their effect on bank line prediction accuracy.

## Methods :

**Dataset Preparation :**
GIS tools are used to digitize the bank lines from landsat data. We consulted with GIS specialists to ensure that the digitized lines are accurate for even the conflicting regions of interest where bank lines are not easily identifiable. The dataset contains thirty two 30m by 30m landsat raster image of 1403 by 2638 resolution with blue, green, red, near infrared, shortwave infrared 1 and shortwave infrared 2 channels. The images represent median valued images of January month for the last thirty two years. 

**Baseline Model :**
Initially, automatically detected edges from image processing were used as ground truth labels as data and an u-net like architecture as the network. A 256 by 256 image patch was used as input and next year bankline binary masks were treated as ground truth data. This approach produced highly noisy output and was abandoned.

A many to one LSTM with last 28 years’ bankline coordinates as input and the bankline coordinate, one year into the future as output showed promising results. At the time of this writing this many to one LSTM model is considered as the baseline model. 

Currently, an approach with RGB images as input and bankline coordinates as output is being looked into along with many to many Conv-LSTM networks.

**Results from the validation set of baseline model :**

The actual banklines are idetified by white and the predicted banklines are identified by red.

![alt text](https://github.com/antorhasan/bank_line_prediction/blob/master/pngs/label0.png) ![alt text](https://github.com/antorhasan/bank_line_prediction/blob/master/pngs/label1.png) 

