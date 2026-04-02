# Deception detection model using Deep Learning!!!
This project have 3 Layers -
 1.Deep Learning Model 
 2.Application (Frontend Layer) 
 3.Server Layer

The Goal of this Project is to eliminate the need of physical Lie Detector appliance (The Polygraph) and to use capabilities of Deep Learning for the same. 

Model consist of around 50M parameters, trained on around 500 Real life judiciary videos available on the official sites.

Hyperparameters,
 1.Batch Size - 8 (As low resource availability of GPU) 
 2.Learning Rate - 10^-4 to 10^-5 (Kept low intially)
 3.Augumentation - The Videos were rotated, flipped , jiggled , colored !
 4.Epochs - Around 50 epochs as used some base models and of low resources availability!!!

The videos were around 30sec to 1min long passed to model with frame rate of 30FPS.
Along with the videos the model also take , Voice (To gather intonation and frequency disturbance from the user) .

Initally the model also used annotation file but model was showing overfitting . So that was not used and trained the model again without annotation explicitly.

Along with Videos and Voice , The text that has spoken also passed to models , So the pattern of words that could be used in lie can be determined . 

So having all these being passed , model instantly predicts if a person is telling the truth or not.

Right now the model is slightly biased towards lie but gradually with more training set , it will be reduced .

For the Proof of concept here are some references published globally --


And Practically one can try out the model through the site -- 
