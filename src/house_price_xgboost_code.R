################ imports used ################

library(caret)
library(data.table)
library(xgboost)

################ read in data ################

setwd('~/Desktop/Fall2021/STAT380/house-prices-fall-2021')
### READ IN DATA
train <- fread("./project/volume/data/raw/Stat_380_train.csv")
test <- fread("./project/volume/data/raw/Stat_380_test.csv")
exampleSubmission <- fread("./project/volume/data/raw/Stat_380_sample_submission.csv")

################ reformat test data ################

test <- as.data.table(test)

# add a sales price column to test
test$SalePrice <- 0

### subset out id columns, and store response vars
drops <- c('Id')

train <- train[, !drops, with = F]
test <- test[, !drops, with = F]

y.train <- train$SalePrice
y.test <- test$SalePrice

################ set up dummies ################

dummies <- dummyVars(SalePrice ~ ., data = train)

x.train <- predict(dummies, newdata = train)
x.test <- predict(dummies, newdata = test)

# set up matrices for xgboost model

dtrain <- xgb.DMatrix(x.train, label = y.train, missing = NA)
dtest <- xgb.DMatrix(x.test, missing = NA)

hyper_perm_tune <- NULL

################ cross validation step  ################

param <- list( objective = "reg:linear",
               gamma = .02,
               booster = "gbtree",
               eval_metric = "rmse",
               eta = 0.01,
               max_depth = 5,
               min_child_weight = 1,
               subsample = .5,
               colsample_bytree = 1,
               tree_method = 'hist'
)


XGBm <- xgb.cv( params = param, nfold = 5, nrounds = 10000, missing = NA, data = dtrain, print_every_n = 1, 
                early_stopping_rounds = 25)

best_ntrees <- unclass(XGBm)$best_iteration

new_row <- data.table(t(param))     

new_row$best_ntrees <- best_ntrees

test_error <- unclass(XGBm)$evaluation_log[best_ntrees,]$test_rmse_mean
new_row$test_error <- test_error

hyper_perm_tune <- rbind(new_row, hyper_perm_tune)
hyper_perm_tune

################ fit xgboost model ################

watchlist <- list(train = dtrain)

XGBm <- xgb.train( params = param, nrounds = best_ntrees, missing = NA, data = dtrain, watchlist = watchlist, print_every_n = 1)

pred <- predict(XGBm, newdata = dtest)

################ save model and dummies ################

saveRDS(XGBm, "./project/volume/models/xgb_model.model")
saveRDS(dummies, "./project/volume/models/xgb_dummies.dummies")

################ save a submission ################

exampleSubmission$SalePrice <- pred
fwrite(exampleSubmission, "./project/volume/data/processed/xgb_submission.csv")
