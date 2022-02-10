# python-opencv-img-preprocessing

# Abstract
This paper presents a classification of the retinal image set into available and not available to obtain only available images using the python environment. Classification processing begins with a large category of classification. The classification of the large category is decided fundus images using feature extraction. Features used include the retinal vascular and optic disc. After selecting the fundus image we classify the not available images. The retinal image-set contains normal images and not available images that reflected or blackout. we use Python Opencv that is an image processing library. In blackout images classifying, we have weighted the pixel luminance so we can figure out a partial blackout. On the other hand, in reflected images classifying, the area where light reflection occurs is divided into 4parts and classified by comparing with adjacent pixels.

# Introduction
There are many types of fundus diseases, including diabetic retinopathy, glaucoma, and age-related macular degeneration. These diseases can weaken or even cause loss of sight in many people. We usually use machine learning to diagnose these fundus diseases. we need a lot of learning data to use machine learning. The learning data can be to includes data that does not need to be entered. It can be adversely affect learning model creation if the learning data include not available images. So we have to classify available or not available images. we are classified through image processing because these cannot be classified manually. Through the feature extraction, the images classified as the fundus image is filtered out of the blackout image and the not available image that is the light reflection.

# Method
1. Histogram analysis for blackout image classification.
2. Masking for reflected image classification.

# Result
-table img-
Table shows the results for 144 normal images, 77 not available images, and 211 total images. In the case of TN, all three images are classified by the reflection image classifier, not the blackout classifier. We checked and found that there was a very fine light reflection, but not enough to be classified as a not available image. On the other hand, in the case of FP, three images were classified in the reflection classifier and the other in the blackout classifier. Misclassified images in the reflection classifier don’t have a dramatic change of luminance. These have a very slow change of luminance. Finally, This table showed an accuracy of 96.6%, a precision of 97.0%, and a recall of 97.7%.
  
# Conclusions
We will get better quality data when gathering training datasets for machine learning training. We can expect a better effect on the training model generation if we get good quality data. Of course, there are still kinds of images that need to be classified. For instance, an image that is out of focus, an uncleared feature, and so on. However, blacked out or reflected images can be classified with high probability
