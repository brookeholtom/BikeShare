library(tidyverse) #importing needed packages
library(tidymodels)
library(vroom)

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
  step_bin2factor(workingday) %>%
  step_rm(datetime)#modifying working day to factor from numbers 

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
