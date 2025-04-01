# import the CSV file
your.cab<- read.csv("~/Desktop/IIT/Predictive Analytics/Data files/Yourcab-1.csv",stringsAsFactors = FALSE)

# List few records from dataset
head(your.cab) 

# summarize the numeric features
summary(your.cab)

# examine the structure of the yourcab data frame
str(your.cab)

# Convert to factor
your.cab$package_id <- factor(yourcab$package_id)
your.cab$travel_type_id <- factor(yourcab$travel_type_id)
your.cab$online_booking <- factor(yourcab$online_booking)
your.cab$mobile_site_booking <- factor(yourcab$mobile_site_booking)
your.cab$Car_Cancellation <- factor(yourcab$Car_Cancellation)

# Recheck
str(your.cab)

# Select relevant columns, including the target variable `car_cancellation`
your_cab_subset <- your.cab %>%
  select( travel_type_id, online_booking, mobile_site_booking, Car_Cancellation)
summary(your_cab_subset)

## create training and holdout sets
set.seed(1947)  # for reproducibility

# Split
idx <- createDataPartition(yourcab_subset$Car_Cancellation, p=0.8, list=FALSE) 
train.df <- your_cab_subset[idx,]
test.df <- your_cab_subset[-idx, ]


### run naive bayes using klaR package
# Use only origin and destination as predictors
your.cab.nb <- NaiveBayes(Car_Cancellation ~ mobile_site_booking + online_booking, data = train.df)
# Output model
your.cab.nb
# predict probabilities
your.cab.pred <- predict(your.cab.nb, newdata=test.df)
# Compute Confusion Matrices aka CrossTabs
confusionMatrix(predict(your.cab.nb, newdata=test.df)$class, factor(test.df$Car_Cancellation))

your.cab.nb <- NaiveBayes(Car_Cancellation ~ mobile_site_booking + online_booking + travel_type_id, data = train.df)
# Output model
your.cab.nb
# predict probabilities
your.cab.pred <- predict(your.cab.nb, newdata=test.df)
# Compute Confusion Matrices aka CrossTabs
confusionMatrix(predict(your.cab.nb, newdata=test.df)$class, factor(test.df$Car_Cancellation))


#Decision tree
your_cab_model <- C5.0(train.df[-21],train.df$Car_Cancellation)


# display simple facts about the tree
your_cab_model

# display detailed information about the tree
summary(your_cab_model)

plot(your_cab_model, type="s", main="Decision Tree")
## Evaluating model performance ----
# create a factor vector of predictions on test data
your_cab_pred <- predict(your_cab_model, test.df)

# cross tabulation of predicted versus actual classes
confusionMatrix(test.df$Car_Cancellation, your_cab_pred)

## Improving model performance ----
## Boosting the accuracy of decision trees
your_cab_boost10 <- C5.0(train.df[-21], train.df$Car_Cancellation, trials = 10)
your_cab_boost10

# See all the individual trees
summary(your_cab_boost10)

# Test boosted trees on test dataset
your_cab_boost_pred10 <- predict(your_cab_boost10, test.df)
confusionMatrix(test.df$Car_Cancellation, your_cab_boost_pred10)

#ANN
your.cab.df <- your_cab_subset %>%
  dummy_cols(select_columns=c("travel_type_id", "online_booking","mobile_site_booking"),
             remove_selected_columns=TRUE, remove_first_dummy=F) %>%
  dummy_cols(select_columns=c("Car_Cancellation"),remove_selected_columns=F)
head(your.cab.df)

set.seed(1947)
# partition the data
idx <- createDataPartition(your.cab.df$Car_Cancellation, p=0.8, list=FALSE)

train.data <- your.cab.df[idx, ]
test.data <- your.cab.df[-idx, ]

# Train the NN. Defaults to 1 hidden layer with 1 hidden node
yourcab_nn <- neuralnet(Car_Cancellation_0+ Car_Cancellation_1~travel_type_id_1+ travel_type_id_2+ travel_type_id_3+ online_booking_0+ online_booking_1+mobile_site_booking_0+ mobile_site_booking_1,  
                     data=train.data,hidden=3)
# Plot NN
plot(yourcab_nn)

# Predict, and compute CM
pred.test <- predict(yourcab_nn, test.data)
class.test <- apply(pred.test, 1, which.max)-1
confusionMatrix(factor(class.test), factor(test.data$Car_Cancellation))

# Train the NN. Defaults to 1 hidden layer with 1 hidden node
yourcab_nn <- neuralnet(Car_Cancellation_0+ Car_Cancellation_1~travel_type_id_1+ travel_type_id_2+ travel_type_id_3+ online_booking_0+ online_booking_1+mobile_site_booking_0+ mobile_site_booking_1,  
                     data=train.data,hidden=c(5, 3), act.fct = tanh)
# Plot NN
plot(yourcab_nn)

# Predict, and compute CM
pred.test <- predict(yourcab_nn, test.data)
class.test <- apply(pred.test, 1, which.max)-1
confusionMatrix(factor(class.test), factor(test.data$Car_Cancellation))

#SVM
# divide into training and test data
split <- createDataPartition(your_cab_subset$Car_Cancellation, p=0.8,list=F)
yourcab_train <- yourcab_subset[idx, ]
yourcab_test  <- yourcab_subset[-idx, ]

# begin by training a simple SVM with linear kernel
yourcab_classifier <- ksvm(Car_Cancellation ~ ., data = yourcab_train, kernel = "vanilladot")

# look at basic information about the model
yourcab_classifier

# predictions on testing dataset
yourcab_predictions <- predict(yourcab_classifier, yourcab_test)
head(yourcab_predictions)
confusionMatrix(yourcab_predictions, yourcab_test$Car_Cancellation)

## Improving model performance ----
# change to a RBF kernel
yourcab_classifier_rbf <- ksvm(Car_Cancellation ~ ., data = yourcab_train, kernel = "rbfdot")
confusionMatrix(yourcab_predictions_rbf, yourcab_test$Car_Cancellation)

# test various values of the cost parameter
cost_values <- c(5, seq(from = 10, to = 15, by = 5))

accuracy_values <- sapply(cost_values, function(x) {
  m <- ksvm(Car_Cancellation ~ ., data = yourcab_train, kernel = "rbfdot", C = x) 
  pred <- predict(m, yourcab_test)
  cm <-  confusionMatrix(pred, yourcab_test$Car_Cancellation)
  return (cm$overall["Accuracy"])
})

plot(cost_values, accuracy_values, type = "b")
