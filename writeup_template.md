#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[image1]: ./german_traffic_signs/sign_1.jpg "Traffic Sign 1"
[image2]: ./german_traffic_signs/sign_2.jpg "Traffic Sign 2"
[image3]: ./german_traffic_signs/sign_3.jpg "Traffic Sign 3"
[image4]: ./german_traffic_signs/sign_4.jpg "Traffic Sign 4"
[image5]: ./german_traffic_signs/sign_5.jpg "Traffic Sign 5"
[image6]: ./german_traffic_signs/sign_6.jpg "Traffic Sign 6"


---
###Writeup / README

[Project code](https://github.com/cecukemon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[HTML version of notebook](https://github.com/cecukemon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

###Data Set Summary & Exploration

Summary of data set in second code cell. I used the Python len() function to find the length of the data sets, the numpy shape() function to find the format of the image (32x32 pixels, RGB, no alpha channel) and the numpy unique() function to find out how many different labels are in the data set.

Result:
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Visualization in third code cell.

###Design and Test a Model Architecture

For preprocessing, I shuffled the training set.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data set provided was already separated in a training, testing and validation set.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


The code for my final model is located in the fifth cell of the notebook. The model consisted of the following layers:

Input: 32x32 RGB image

- Layer 1:
-- Convolution, Input = 32x32x3. Output = 28x28x6.
-- RELU
-- Pooling Input = 28x28x6. Output = 14x14x6.
- Layer 2:
-- Convolution, Input 14x14x6, Output = 10x10x16.
-- RELU
-- Pooling, Input = 10x10x16. Output = 5x5x16.
- Layer 3:
-- Fully Connected, Input = 400. Output = 120.
-- RELU
- Layer 4:
-- Fully Connected. Input = 120. Output = 84.
-- RELU
- Layer 5:
-- Fully Connected. Input = 84. Output = 43 (number of labels)


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth and seventh cell of the notebook.

I used an exponentially decaying training rate, starting at 0.001, using the Tensorflow tf.train.exponential_decay function. The sigma value for the LeNet network was sigma = 0.0765, and I had the training run for up to 40 epochs, stopping the training as soon as the network reached 0.93 accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

Validation set accuracy 93%
Test set accuracy on my own images 16%

I used the architecture presented in the course which is used to classify the MNIST data set. I believed it would be relevant for the traffic sign application because the problem is similar - recognizing clear and well-defined shapes on 2D images. I found it difficult to get the model's accuracy over 93%, so I'm not sure if it's really a good fit for this prpoblem.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]

The third image might be difficult to classify because it's an older version of the traffic sign (newer ones don't have the km/h text). The fourth image might be difficult to classify because it's taken from a tilted perspective and the sign is distorted. The fifth image is a sign that isn't in the training set at all.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the sixteenth cell of the Ipython notebook.
Results:

| Image			        |     Best prediction	        					|  Second best prediction |
|:---------------------:|:------------------------------:| :------------------------------:|
| No vehicles      		| Keep right   									|  Priority road |
| No passing     			| Traffic signals 										| General caution |
| 50 km/h					| Speed limit (30km/h)											|Speed limit (120km/h) |
| Stop      		| No passing for vehicles ...					 				| Yield|
| Cow			| No passing for vehicles ...      							| No passing|
| Priority road | Priority road | No passing for vehicles ... |


The model guessed one out of six signs correctly, giving an accuracy of 16%. This compares unfavorably with the training set. The image quality in the training set seems to be very different from the images I used here, they are much darker. Maybe this is a reason.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

First image:

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:| 


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
