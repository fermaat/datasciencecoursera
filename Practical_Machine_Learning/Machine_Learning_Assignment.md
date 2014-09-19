# Machine_Learning_Assignment

Synopsis
-----------
   
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible
to collect a large amount of data about personal activity relatively inexpensively.
These type of devices are part of the quantified self movement. In this project,
the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6
participants asked to perform barbell lifts correctly and incorrectly in 5 different ways,
in order to predict the manner in which they did the exercise


Data
------------
   
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.
I would like to thank these people for allowing their data to be used for this kind of assignment.

First of all, we read the data from the directory where we previously downoaded it:


```r
setwd("C:/Users/Fer/Desktop/Coursera")
wholetrain <- read.csv("./data/pml-training.csv")
validation <- read.csv("./data/pml-testing.csv")
```

And we can, firstly, create a partition for having test and train sets:
(setting the seed to a given number)


```r
library (ggplot2)
library (lattice)
library (caret)
set.seed(31415)
trainpartition <- createDataPartition(wholetrain$classe, p = 0.6, list = FALSE)
train <- wholetrain[trainpartition, ]
testset <- wholetrain[-trainpartition, ]
```

Please allow me to denote "validation" the test document downloaded and just
"testset" the divided one, as I understand the validation more like the results
to be submitted.

According to the descriptions, The variable to be predicted is "classe".
Now we can perform a bit of exploratory data analysis by firstly removing near
zero variance variables from the train set, as they will mean no real difference


```r
nearzeronames <- nearZeroVar(train)
train <- train[, -nearzeronames]
testset <- testset [, -nearzeronames]
validation <- validation [, -nearzeronames]
```

Now, if we take a look at the summary for the remaining data (I will hide it as
there are still a lot of variables there)


```r
summary (train)
```

we can see there are a lot of NA values. These variables will add nothing to the
model but increasing the computation time. So, we will consider null a variable
with more than 80% NA values. We will remove them. first six variables (names, 
dates, etc will also be removed)


```r
nasremoval <- sapply (train, function (x){sum (is.na(x))})
usablecol <- names(nasremoval[nasremoval < 0.8 * length(train$classe)])
usabletrain <- train [, names(train) %in% usablecol [7:length(usablecol)]]
usabletest <- testset [, names(testset) %in% usablecol [7:length(usablecol)]]
```

The same strategy is applied to the validation data, but adding the "classe" 
variable (full of NAs).


```r
usablevalidation <- validation [, names (validation) %in% usablecol[7:length(usablecol)]]
usablevalidation$classe <- NA
```


Modeling
------------
   
Now we can get our hands dirty and get into the model itself. We have picked the
RandomForest algorithm in the Breiman and Cutler implementation, and the main
reasons are mainly avoiding overfitting and easing computation time (different
implementations have been tested with a much slower result). We will then, predict
the "classe" variable



```r
library (randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fit <- randomForest(classe ~ ., data = usabletrain)
trainingprediction <- predict(fit, usabletrain)
print(confusionMatrix(trainingprediction, usabletrain$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

Testing
---------
   
We can see the accuracy of our model is really good (100%), but we need to perform a
cross validation on the test set in order to have a real reference:


```r
testprediction <- predict(fit, usabletest)
print(confusionMatrix(testprediction, usabletest$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    6    0    0    0
##          B    1 1510   21    0    0
##          C    0    2 1344   16    0
##          D    0    0    3 1269    6
##          E    1    0    0    1 1436
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.995    0.982    0.987    0.996
## Specificity             0.999    0.997    0.997    0.999    1.000
## Pos Pred Value          0.997    0.986    0.987    0.993    0.999
## Neg Pred Value          1.000    0.999    0.996    0.997    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.171    0.162    0.183
## Detection Prevalence    0.285    0.195    0.174    0.163    0.183
## Balanced Accuracy       0.999    0.996    0.990    0.993    0.998
```


Accuracy in cross validation test resulted on a 99.27%, and so, the out of sample
error is 0.73%. No adjustments will be performed as accuracy is really high.

Validation
-----------

Finally, we will perform the validation prediction:
   

```r
validationprediction <- predict(fit, usablevalidation)
answers <- as.vector(validationprediction)
validationprediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


