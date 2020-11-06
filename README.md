# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about calls made to possible customers trying to get them subscribed. Each call also has a Campaing ID where we can identify from which of the 39 campaings the possible customer came. Our goal with this data is trying to messure the probabilities if a customer is going to subscribe or not based on their information.

The best model was a VotingEnsemble with an accuracy of 0.918 created by AutoML Pipeline.

## Scikit-learn Pipeline
The creation of this pipeline started with the data, the CSV data is downloaded from a Azure Blob Storage URL and then cleaned and splitted between test (30%) and train (70%), then we create a LogisticRegression Model with the following two hyperparameters "Inverse of Regulation Strength" between 0.1 and 1, and "Maximum Number Of Interactions" of 20,40 or 80 choosed randomly. Logistic Regression Model are used for binary classification, and for this problem is a good candidate to be used as our model, our hyperparameters "Inverse of Regulation Strength" and "Maximum Number of Interactions" are set to avoid overfitting.

All this HyperDrive configuration run inside an experiment, that is going to stop earlier if the slack factor fall 0.1 after 5 interactions (based on the "delay_evaluation=5" property). Early Stopping Policy is important because we can prevent overfitting and save compute resources.
![Scikit-learn](https://github.com/MariojosePalma/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/raw/master/HyperDrive.png)
## AutoML
AutoML is a very simple but powerfull tool that help to find the right model just passing few parameters. In this case it was the previous Dataset, the task to perform ("classification"), "accuracy" is choosen as primary metric and "y" as our label column. After write this config and submit the experiement, the AutoML tool start looking for the best model based on config we wrote previously.
![AutoML](https://github.com/MariojosePalma/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/raw/master/AutoML.png)

## Pipeline comparison
AutoML had a model with higher accuracy than Scikit-learn but the difference was very small (less than 0.5%), base on all the work/time and possible human error to put together an Scikit-learn Pipeline vs AutoML Pipeline, AutoML is the most efficient option to go.   

## Future work
To improve accuracy of this experiments we can take a look to the metrics and check which of the features are more important, remove the less important and collect other pieces of data that may improve the accuracy of the models. This may improve the model because we will be reducing the noise of the no important features and the weights of the features will be more balanced. 
