library(tidyverse) #importing needed packages
library(tidymodels)
library(vroom)
library(poissonreg)

bike <- vroom("train.csv") #importing the data via vroom
bike_test <-  vroom("test.csv") #importing test data


#cleaning section: changing the single "4" weather day to "3"
bike <- bike %>% 
         mutate(weather = ifelse(weather >= 4, 3, weather)) %>%
         select(-casual, -registered)

bike_test <- bike_test %>%
  mutate(weather = ifelse(weather >= 4, 3, weather))


#feature engineering section via recipe:  
my_recipe <- recipe(count ~ ., data=bike) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime)

#updating table to reflect above changes, will be saved with original name "bike"
prepped_recipe <- prep(my_recipe) 
bake(prepped_recipe, new_data=bike)


#creating a model
my_mod <- linear_reg() %>% 
  set_engine("lm") 

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike)

bike_predictions <- predict(bike_workflow,
                            new_data=bike_test) 


bike_predictions[bike_predictions<0] <- 0 #making all negative columns 0

bike_predictions <- rename(bike_predictions, count = .pred)


bike_predictions$datetime <- bike_test$datetime
bike_predictions$datetime <-format(bike_predictions$datetime)

vroom_write(bike_predictions, "bike_predictions.csv", delim=",")



#Poisson Model

pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pois_mod) %>%
fit(data = bike) # Fit the workflow9

bike_predictions_pois <- predict(bike_pois_workflow,
                            new_data=bike_test) # Use fit to predict12


bike_predictions_pois <- rename(bike_predictions_pois, count = .pred)


bike_predictions_pois$datetime <- bike_test$datetime
bike_predictions_pois$datetime <-format(bike_predictions_pois$datetime)

vroom_write(bike_predictions_pois, "bike_predictions_pois.csv", delim=",")



#Penalized Regression 


my_recipe_2 <- recipe(count ~ ., data=bike) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=18

prepped_recipe <- prep(my_recipe_2) 
bake(prepped_recipe, new_data=bike)



logTrainSet <- bike %>%
  mutate(count=log(count))
## Define the model
lin_model <- linear_reg() %>%
  set_engine("lm")


## Penalized regression model10
preg_model <- linear_reg(penalty=0.01, mixture=0.01) %>% #Set model and tuning11
  set_engine("glmnet") # Function to fit in R12
preg_wf <- workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(preg_model) %>%
  fit(data=logTrainSet)
preg_predictions <- predict(preg_wf, new_data=bike_test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=preg_predictions, file="./LogLinearPreds.csv", delim=",")



###########################
###Tuning Models Example
###########################

logTrainSet <- bike %>%
  mutate(count=log(count))

my_recipe_tree <- recipe(count ~ ., data=bike) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors(0, 18)) # Make mean 0, sd=18

prepped_recipe <- prep(my_recipe_tree) 
bake(prepped_recipe, new_data=bike)


preg_model_2 <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% 
                         set_engine("glmnet") 


preg_wf_2 <- workflow() %>%
add_recipe(my_recipe_tree) %>%
add_model(preg_model_2) 
  

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) 

folds <- vfold_cv(logTrainSet, v = 10, repeats=1)


CV_results <- preg_wf_2 %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>%
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()


bestTune <- CV_results %>%
  select_best("rmse")


final_wf <- preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=logTrainSet)


predictions_5 <- final_wf %>%
predict(new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=predictions_5, file="./LogLinearPreds_Penalized.csv", delim=",")




#####################
###Regression Tree###
#####################

logTrainSet_tree <- bike %>%
  mutate(count=log(count))

library(rpart)
library(tidymodels)

my_mod_tree <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

my_recipe_tree <- recipe(count ~ ., data=bike) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=18


## Create a workflow with model & recipe


preg_wf_tree <- workflow() %>%
  add_recipe(my_recipe_tree) %>%
  add_model(my_mod_tree)

## Set up grid of tuning values
tuning_grid_tree <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5) 

## Set up K-fold CV
  folds <- vfold_cv(logTrainSet_tree, v =5, repeats=1)

## Find best tuning parameters
CV_results_tree <- preg_wf_tree %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_tree,
            metrics=metric_set(rmse, mae, rsq))


bestTune_tree <- CV_results_tree %>%
  select_best("rmse")

## Finalize workflow and predict

final_wf_tree <- preg_wf_tree %>%
  finalize_workflow(bestTune_tree) %>%
  fit(data=logTrainSet)

predictions_tree <- final_wf_tree %>%
  predict(new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=predictions_tree, file="./LogLinearPreds_Tree.csv", delim=",")



###Random Forests ###


library(ranger)
library(tidymodels)

my_mod_forest <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
logTrainSet_tree <- bike %>%
  mutate(count=log(count))




my_recipe_forest <- recipe(count ~ ., data=bike) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=18

preg_wf_forest <- workflow() %>%
  add_recipe(my_recipe_forest) %>%
  add_model(my_mod_forest)


## Set up grid of tuning values
tuning_grid_forest <- grid_regular(mtry(range=c(1, 10)),
                                 min_n(),
                                 levels = 5) 

## Set up K-fold CV
folds <- vfold_cv(logTrainSet_tree, v =5, repeats=1)

## Find best tuning parameters
CV_results_forest <- preg_wf_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_forest,
            metrics=metric_set(rmse, mae, rsq))


bestTune_forest <- CV_results_forest %>%
  select_best("rmse")


## Finalize workflow and predict


final_wf_forest <- preg_wf_forest %>%
  finalize_workflow(bestTune_forest) %>%
  fit(data=logTrainSet_tree)

predictions_forest <- final_wf_forest %>%
  predict(new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=predictions_forest, file="./LogLinearPreds_Forest.csv", delim=",")



###Stacked Models###
library(stacks)

my_recipe_stacked <- recipe(count ~ ., data=bike) %>%
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=18


prepped_recipe_stacked <- prep(my_recipe_stacked) 
bake(prepped_recipe_stacked, new_data=bike)
bake(prepped_recipe_stacked, new_data=bike_test)


logbike <- bike %>%
  mutate(count=log(count))

folds <- vfold_cv(logbike, v=10, repeats=1)

untunedmodel <- control_stack_grid()
tunedmodel <-  control_stack_resamples()


lin_model_stacked <- linear_reg() %>%
  set_engine("lm")

linregwf_stacked <-  workflow () %>%
  add_recipe(my_recipe_stacked) %>%
  add_model(lin_model_stacked) 

lin_reg_model_stacked <- fit_resamples(linregwf_stacked, 
                                       resamples = folds, 
                                       metrics = metric_set(rmse, mae, rsq), 
                                       control=tunedmodel)

preg_model_stacked <- linear_reg(penalty=tune(),
                           mixture=tune()) %>% 
  set_engine("glmnet") 


preg_wf_stacked <-workflow() %>%
  add_recipe(my_recipe_stacked) %>%
  add_model(preg_model_stacked) 

preg_tuning_grid_stacked <- grid_regular(penalty(), 
                                         mixture(),
                                         levels= 10)


preg_models_stacked <- preg_wf_stacked %>%
  tune_grid(resamples=folds, 
            grid=preg_tuning_grid_stacked,
            metrics=metric_set(rmse, mae, rsq),
            control=untunedmodel)



my_stack <- stacks() %>%
  add_candidates(preg_models_stacked) %>%
  add_candidates(lin_reg_model_stacked)


stackData <- as_tibble(my_stack)


fitted_bike_stack <-  my_stack %>%
  blend_predictions() %>%
  fit_members()

stack_predictions <- predict(fitted_bike_stack, new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=stack_predictions, file="./LogLinearPreds_Stacked.csv", delim=",")



 

 



