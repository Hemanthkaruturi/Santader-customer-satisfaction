#Santandar Customer Statisfaction

#importing packages
if(!require(data.table) | !require(mice) | !require(caTools) | !require(kernlab) | !require(class) | !require(e1071)
   | !require(rpart) | !require(randomForest) | !require(xgboost) | !require(gbm) | !require(caret) | !require(Matrix)
   | !require(MatrixModels) | !require(VIM)) {
  install.packages(c('rpart','randomForest', 'xgboost', 'gbm', 'caret','data.table','mice','caTools',
                     'Kernlab','class','e1071'))
}

#Importing Data
library(data.table)
train <- fread(file.choose(), integer64 = 'numeric', data.table = F)
test <- fread(file.choose(), integer64 = 'numeric', data.table = F)

#Removing ID
test.id <- train$ID
train$ID <- NULL
test$ID <- NULL

#Removing Target Variable from train data
train.y <- train$TARGET
train$TARGET <- NULL

#Replacing the values -999999 with NA in var3
train[train$var3 == -999999,"var3"] <- NA
test[test$var3 == -999999, "var3"] <- NA

train[train$var36 == 99, "var36"] <- NA
test[test$var36 == 99,"var36"] <- NA

#Replacing 117310.979016494 in var38 with NA
train[train$var38 == 117310.979016494, "var38"] <- NA
test[test$var38 == 117310.979016494, "var38"] <- NA

#Visualising Missing Values
library(VIM)
mice_plot <- aggr(train, col=c('navyblue','yellow'),
                    numbers=TRUE, sortVars=TRUE,
                    labels=names(train), cex.axis=.7,
                    gap=3, ylab=c("Missing data","Pattern"))

#checking Missing values
sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)

hist(log1p(train$var38), 100)

#Removing the constants features.
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

# Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)    #setdiff (remove the names in 'toRemove' list from train set)
train <- train[, feature.names]
test <- test[, feature.names]

rm(features_pair,f,f1,f2,feature.names,pair,toRemove,count0,count3mod)


train$TARGET <- train.y    #Including target variable to the training set

# Make some variables categorical

train$var3 <- as.factor(train$var3)
test$var3 <- as.factor(test$var3)

train$var36 <- as.factor(train$var36)
test$var36 <- as.factor(test$var36)

# Imputing Missing and feature scaling using Vtreat package
library(mice)
imputed_data <- mice(train[,c(2,135,247)], m=5, maxit = 1, method = 'rf', seed = 500)
train[,c(2,135,247)] <- complete(imputed_data, 1)
################################################# End of Data Preparation ##################################################


#Preparing Validation set
library(caTools)
set.seed(123)
split <- sample.split(train$TARGET, SplitRatio = 0.80)
trainset <- subset(train, split == TRUE)
testset <- subset(train, split == FALSE)

##############################################################
#Applying kernel PCA for dimensinality reduction
library(kernlab)
kpca = kpca(~., data = trainset, kernel = 'rbfdot', features = 200)
training_set_pca = as.data.frame(predict(kpca, trainset))
training_set_pca$TARGET = trainset$TARGET
test_set_pca = as.data.frame(predict(kpca, testset))
test_set_pca$TARGET = testset$TARGET
############# Something wrong in PCA #########################


#Preparing Model Starts here

#Fitting data to xgboost
library(xgboost)
library(Matrix)
require(MatrixModels)
train.y <- trainset[,'TARGET']
test.y <- testset[,'TARGET']

# converting dataser into a Matrix
dtrain <- sparse.model.matrix(TARGET ~ .-1, data = trainset)
dtest <- sparse.model.matrix(TARGET ~ .-1, data = testset)

xg_classifier <- xgboost(data = dtrain, label = train.y, nrounds = 10)

#Fitting Data to random forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = trainset[,-248], y = trainset$TARGET, ntree = 500)

######################  These Methods are not working #######################################

#Fitting training data to KNN
library(class)
knn_classifier <- knn(train = trainset[,-278], test = testset[,-278], cl = trainset$TARGET, k=5, prob = TRUE)

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = TARGET ~ .,
                     data = trainset,
                     type = 'C-classification',
                     kernel = 'sigmoid')

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(x = trainset[,-278],
                           y = trainset$TARGET)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = TARGET ~ ., data = trainset)

################### End of incorrect methods ################################

#Predicting the results
svm_pred <- predict(svm_classifier, newdata = testset[,-278])              # (subscript) logical subscript too long
nb_pred <-  predict(nb_classifier, newdata = testset[,-278])                 #subscript out of bounds
dt_pred <-  predict(dt_classifier, newdata = testset[,-278], type='class')   #Error

rf_pred <- predict(rf_classifier, newdata = testset)
xg_pred <-  predict(xg_classifier, newdata = as.matrix(testset[,-278]))
xg_pred <- (xg_pred >= 0.5)


#Confusion Matrix
cm_rf <- table(testset$TARGET, rf_pred)
cm_xg <- table(testset$TARGET, xg_pred)

