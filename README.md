# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./histogram.png "Histogram"
[image2]: ./gray.jpg "Grayscaling"
[image3]: ./normalized.jpg "normalized"
[imageO]: ./original.jpg "original"
[image4]: ./data/noentry.jpg "Traffic Sign 1"
[image5]: ./data/red100.jpg "Traffic Sign 2"
[image6]: ./data/stop.png "Traffic Sign 3"
[image7]: ./data/uparrow.png "Traffic Sign 4"
[image8]: ./data/yield.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is my writeup, and my project notebook is [here](https://github.com/nigelredding/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In the second cell of my notebook, I printed out the following information about our data (training, testing and validation data):

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a histogram showing the distribution of the traffic sign types in the training data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I preprocessed the data in two steps. Our original image is here

![alt text][imageO]

Next, we convert this to grayscale. 


![alt text][image2]

Finally, we normalize the image. Unfortunately, the image appears almost entirely black. But nevertheless, this poses no problem for the neural network.

![alt text][image3]

The reason I chose to use grayscale to preprocess the images was that I did not feel that the color was relevant to the classification of the traffic signs.

I chose to normalize the data as I found that my model performed better with normalized data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale   							|
| Convolution           | Input 32x32x1, Output 28x28x32                |
| RELU              	|                                               |
| Max Pooling			| Input 28x28x32, Output 14x14x32				|
| Convolution       	| Input 14x14x32, Output 10x10x64               |
| RELU                  |      									        |
| Max Pooling           | Input = 10x10x64, Output = 5x5x64.        	|
| Flatten				| Input = 5x5x64, Output = 1600.        		|
| Dropout				|	.											|
| Dense 			    |	Input = 1600, Output = 120					|
| RELU					|												|
| Dropout				|												|
| Dense 			    |	Input = 120, output = 84					|
| RELU					|												|
| Dropout				|												|
| Dense					|	Input = 84, Output = 43						|


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I minimized the cross entropy using the Adam optimizer. I ran for 25 epochs, used a batch size of 128 and a learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.980 
* test set accuracy of 0.906

The first approach I took was simply to copy the LeNet architecture. I found that this never performed any better than 91% on the validation set. I chose to add a dropout later after the flattening operation, and found that slightly improved performance. After adding two "Dense -> RELU -> Dropout" layers, I found that I consistently got a performance of at least 93% on the validation set.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image was the hardest for my classifier to classify as it was similar to the other speed limit signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry    			| No entry 										|
| Yield					| Yield											|
| 100 km/h	      		| Vehicles over 3.5 metric tons prohibited.		|
| Ahead only			| Ahead only      							    |


Our model predicted 4 out of the 5 signs correctly, giving an accuracy of 80%.

Our accuracy on the test set was 93.5%. However, given the small sample size here (n=5 examples), we should not be too surprised by the relatively low accuracy (80%). 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the final cell of the Ipython notebook.

The predictions for each sign from the web, along with the given softmax probabilities is given below:
Our softmax probabilities are given by: 

First sign (Stop sign):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  0.9999884366989136   | Stop sign   									| 
|5.4538154472538736e-06 | Speed limit (30km/h) 							|
|2.8945223675691523e-06	| Turn left ahead								|
|2.3776767648087116e-06 | Yield					                		|
|4.998565827918355e-07  | Speed limit (50km/h)      			    	|

Speed limit (100km/h):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  0.7623458504676819   | Speed limit (80km/h)  	                    | 
|  0.1487751007080078   | Vehicles over 3.5 metric tons prohibited 		|
|  0.05825525149703026	| No passing for vehicles over 3.5 metric tons	|
|  0.009625362232327461	| Speed limit (100km/h)					 		|
|  0.00924014113843441  | Speed limit (60km/h)      				   	|

No Entry:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.0                  | No entry  	                                | 
|1.3217007349686227e-11 | Speed limit (20km/h) 					        |
|8.233778242550116e-13  | Go straight or left							|
|2.4248215468956197e-14 | Stop              					 		|
|2.4032408414654746e-14 | Speed limit (120km/h)      				    |

Ahead only:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.0                  | Ahead only  	                                | 
|1.017736390469521e-17  | Go straight or right 						    |
|1.50526334141005e-19   | No passing        							|
|4.2008095989867017e-20 | Go straight or left             				|
|1.505641582669837e-20  | Yield                     		        	|

Yield:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.0                  | Yield     	                                | 
|1.5241646877318883e-25 | Ahead only 			        			    |
|2.0782129169411435e-26 | No vehicles  			        				|
|1.3011873904887757e-26 | No passing             						|
|1.2123431834945762e-29 | Speed limit (60km/h)      	            	|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


