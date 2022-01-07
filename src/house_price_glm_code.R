################ imports used ################

library(caret)
library(data.table)
library(glmnet)
library(plotmo)

################ read in data ################

train <- fread("./project/volume/data/raw/Stat_380_train.csv")
test <- fread("./project/volume/data/raw/Stat_380_test.csv")
exampleSubmission <- fread("./project/volume/data/raw/Stat_380_sample_submission.csv")

################ reformat test data ################

test <- as.data.table(test)

# add a sales price column to test
test$SalePrice <- 0

# subset out id columns, and store response vars
drops <- c('Id')

train <- train[, !drops, with = F]
test <- test[, !drops, with = F]

train_y <- train$SalePrice
test_y <- test$SalePrice

################ set up dummies ################

dummies <- dummyVars(SalePrice ~ ., data = train)

train <- predict(dummies, newdata = train)
test <- predict(dummies, newdata = test)

# Handle NA values in train
train[is.na(train)] <- 0
test[is.na(test)] <- 0

################ cross validation step  ################

train <- as.matrix(train)
test <- as.matrix(test)

gl_model <- cv.glmnet(train, train_y)
bestlam <- gl_model$lambda.min

################ fit logistic model ################

gl_model <- glmnet(train, train_y, alhpa = 1, family = "gaussian")
plot_glmnet(gl_model)

################ save model and dummies ################

saveRDS(gl_model, "./project/volume/models/glm_model.model")
saveRDS(dummies, "./project/volume/models/glm_dummies.dummies")

### Use and fit full model to test data
pred <- predict(gl_model, s = bestlam, newx = test)
predict(gl_model, s = bestlam, newx = test, type = "coefficients")

################ save a submission ################

exampleSubmission$SalePrice <- pred
fwrite(exampleSubmission, "./project/volume/data/processed/glm_submission.csv")
