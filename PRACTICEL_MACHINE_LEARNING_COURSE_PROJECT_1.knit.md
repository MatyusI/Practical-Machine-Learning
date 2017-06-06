---
title: "PRACTICAL MACHINE LEARNING COURSE PROJECT"
author: "MatyusI"
date: '2017 j√∫nius 6 '
output:
  pdf_document: default
  html_document: default
---



## R Markdown

Practical Machine Learning - Course Project

Introduction

For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.
Below is the code I used when creating the model, estimating the out-of-sample error, and making predictions. I also include a description of each step of the process.

Data Preparation

I load the caret package, and read in the training and testing data:

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```

## Loading required package: lattice
## Loading required package: ggplot2


```r
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
library(caret)
```

Because I want to be able to estimate the out-of-sample error, I randomly split the full training data (train) into a smaller training set (train1) and a validation set (train2):


```r
set.seed(10)
inTrain <- createDataPartition(y=train$classe, p=0.7, list=F)
train1 <- train[inTrain, ]
train2 <- train[-inTrain, ]
```

I am now going to reduce the number of features by removing variables with nearly zero variance, variables that are almost always NA, and variables that donít make intuitive sense for prediction. Note that I decide which ones to remove by analyzing train1, and perform the identical removals on train2:

# remove variables with nearly zero variance

```r
rnvz <- nearZeroVar(train1)
train1 <- train1[, -rnvz]
train2 <- train2[, -rnvz]
```


# remove variables that are almost always NA

```r
mNA <- sapply(train1, function(x) mean(is.na(x))) > 0.95
train1 <- train1[, mNA==F]
train2 <- train2[, mNA==F]
```

# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables


```r
train1 <- train1[, -(1:5)]
train2 <- train2[, -(1:5)]
```

Model Building

I decided to start with a Random Forest model, to see if it would have acceptable performance. I fit the model on train1, and instruct the ìtrainî function to use 3-fold cross-validation to select optimal tuning parameters for the model.

# instruct train to use 3-fold CV to select optimal tuning parameters


```r
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
```

# fit model on train1

```r
fit <- train(classe ~ ., data=train1, method="rf", trControl=fitControl)
```


# print final model to see tuning parameters it chose


```r
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.26%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    7 2647    3    1    0 0.0041384500
## C    0    7 2389    0    0 0.0029215359
## D    0    0    9 2242    1 0.0044404973
## E    0    0    0    7 2518 0.0027722772
```



I see that it decided to use 500 trees and try 27 variables at each split.

Model Evaluation and Selection

Now, I use the fitted model to predict the label (ìclasseî) in train2, and show the confusion matrix to compare the predicted versus the actual labels:

# use model to predict classe in validation set (train2)

```r
predicts <- predict(fit, newdata=train2)
```

# show confusion matrix to get estimate of out-of-sample error


```r
confusionMatrix(train2$classe, predicts)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    3 1134    1    1    0
##          C    0    3 1023    0    0
##          D    0    0    2  962    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9967, 0.9991)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9974   0.9971   0.9979   1.0000
## Specificity            1.0000   0.9989   0.9994   0.9996   0.9998
## Pos Pred Value         1.0000   0.9956   0.9971   0.9979   0.9991
## Neg Pred Value         0.9993   0.9994   0.9994   0.9996   1.0000
## Prevalence             0.2850   0.1932   0.1743   0.1638   0.1837
## Detection Rate         0.2845   0.1927   0.1738   0.1635   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9991   0.9982   0.9982   0.9988   0.9999
```

The accuracy is 99.8%, thus my predicted accuracy for the out-of-sample error is 0.2%.

This is an excellent result, so rather than trying additional algorithms, I will use Random Forests to predict on the test set.

Re-training the Selected Model

Before predicting on the test set, it is important to train the model on the full training set (train), rather than using a model trained on a reduced training set (train1), in order to produce the most accurate predictions. Therefore, I now repeat everything I did above on train and test:

# remove variables with nearly zero variance


```r
rnvz <- nearZeroVar(train)
train <- train[, -rnvz]
test <- test[, -rnvz]
```

# remove variables that are almost always NA


```r
mNA <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, mNA==F]
test <- test[, mNA==F]
```

# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables


```r
train <- train[, -(1:5)]
test <- test[, -(1:5)]
```

# re-fit model using full training set (train)

```r
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=train, method="rf", trControl=fitControl)
```

Making Test Set Predictions

Now, I use the model fit on train to predict the label for the observations in test, and write those predictions to individual files:

# predict on test set

```r
predicts <- predict(fit, newdata=test)
```

# convert predictions to character vector

```r
predicts <- as.character(predicts)
```

# create function to write predictions to files


```r
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}
```

# create prediction files to submit

```r
pml_write_files(predicts)
```

