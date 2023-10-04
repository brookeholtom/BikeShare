#Packages
library(tidyverse) 
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)


#Importing Data In
bike <- vroom("train.csv") #importing the data via vroom
bike_test <-  vroom("test.csv") #importing test data


#Cleaning Data
bike <- bike %>% 
  mutate(weather = ifelse(weather >= 4, 3, weather)) %>%
  select(-casual, -registered)

bike_test <- bike_test %>%
  mutate(weather = ifelse(weather >= 4, 3, weather))


#Recipe, Prep, and Bake
my_recipe <- recipe(count ~ ., data=bike) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(datetime)

prepped_recipe <- prep(my_recipe) 
bake(prepped_recipe, new_data=bike)
baked_recipe <- bake(my_recipe, new_data = bike_test)

#Linear Model
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


#Penalized Regression 

my_recipe_2 <- recipe(count~., data = bike) %>%
  step_num2factor(season, levels = c("Spring", "Summer", "Fall", "Winter")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_num2factor(weather, levels = c("Clear", "Mist", "Rain")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe_2 <- prep(my_recipe_2) 
bake(prepped_recipe_2, new_data=bike)

logTrainSet <- bike %>%
  mutate(count=log(count))

lin_model <- linear_reg() %>%
  set_engine("lm")

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


#Tuning Models

preg_model_2 <- linear_reg(penalty=tune(),
                           mixture=tune()) %>% 
  set_engine("glmnet") 


preg_wf_2 <- workflow() %>%
  add_recipe(my_recipe_2) %>%
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

final_wf <- preg_wf_2 %>%
  finalize_workflow(bestTune) %>%
  fit(data=logTrainSet)

predictions_tuning <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=predictions_tuning, file="./LogLinearPreds_Penalized.csv", delim=",")

#Regression Tree
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

preg_wf_tree <- workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = (5))

folds <- vfold_cv(logTrainSet, v = 5, repeats = 1)

CV_results_tree <- preg_wf_tree %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

bestTune <- CV_results_tree %>%
  select_best("rmse")

final_wf_tree <- preg_wf_tree %>%
  finalize_workflow(bestTune) %>%
  fit(data = logTrainSet)

predictions_tree <- final_wf_tree %>%
  predict(new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=predictions_tree, file="./LogLinearPreds_Tree.csv", delim=",")


#Random Forests 

my_mod_forest <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

preg_wf_forest <- workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(my_mod_forest)


## Set up grid of tuning values
tuning_grid_forest <- grid_regular(mtry(range=c(1, 10)),
                                   min_n(),
                                   levels = 5) 

## Set up K-fold CV
folds <- vfold_cv(logTrainSet, v =5, repeats=1)

## Find best tuning parameters
CV_results_forest <- preg_wf_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_forest,
            metrics=metric_set(rmse, mae, rsq))


bestTune_forest <- CV_results_forest %>%
  select_best("rmse")

final_wf_forest <- preg_wf_forest %>%
  finalize_workflow(bestTune_forest) %>%
  fit(data=logTrainSet)

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

prepped_recipe_stacked <- prep(my_recipe_2) 
bake(prepped_recipe_stacked, new_data=bike)
bake(prepped_recipe_stacked, new_data=bike_test)


folds <- vfold_cv(logTrainSet, v=10, repeats=1)

untunedmodel <- control_stack_grid()
tunedmodel <-  control_stack_resamples()


lin_model_stacked <- linear_reg() %>%
  set_engine("lm")

linregwf_stacked <-  workflow () %>%
  add_recipe(my_recipe) %>%
  add_model(lin_model_stacked) 

lin_reg_model_stacked <- fit_resamples(linregwf_stacked, 
                                       resamples = folds, 
                                       metrics = metric_set(rmse, mae, rsq), 
                                       control=tunedmodel)

preg_model_stacked <- linear_reg(penalty=tune(),
                                 mixture=tune()) %>% 
  set_engine("glmnet") 


preg_wf_stacked <-workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(preg_model_stacked) 

preg_tuning_grid_stacked <- grid_regular(penalty(), 
                                         mixture(),
                                         levels= 10)


preg_models_stacked <- preg_wf_stacked %>%
  tune_grid(resamples=folds, 
            grid=preg_tuning_grid_stacked,
            metrics=metric_set(rmse, mae, rsq),
            control=untunedmodel)


random_forest_stacked <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

random_forest_wf_stacked <- workflow() %>%
  add_recipe(my_recipe_2) %>%
  add_model(random_forest_stacked)

tuning_grid_random_forest <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = (5))

random_forest_stacked <- random_forest_wf_stacked %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_random_forest,
            metrics = metric_set(rmse),
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



