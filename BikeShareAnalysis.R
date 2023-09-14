library(tidyverse) #importing needed packages
library(tidymodels)
library(vroom)

bike <- vroom("train.csv") #importing the data via vroom


#cleaning section: changing the single "4" weather day to "3"
bike <- bike %>% 
         mutate(weather = ifelse(weather >= 4, 3, weather)) 


#feature engineering section via recipe:  
my_recipe <- recipe(count ~ ., data=bike) %>% 
  step_date(datetime, features="dow") %>% #adding day of week column 
  step_date(datetime, features="month") %>% #adding month column
  step_num2factor(season, transform = function(x) x, levels=c("spring", "summer", "fall", "winter")) %>% #modifying season column from numbers to a factor
  step_num2factor(weather, transform = function(x) x, levels=c("clear", "mist", "rain/snow")) %>% #modifying weather from numbers into factors
  step_bin2factor(holiday) %>% #modifying holiday to factor from numbers
  step_bin2factor(workingday) %>% #modifying working day to factor from numbers 
  step_rm(c(casual, registered)) #removing the casual and registered columns, not needed 

#updating table to reflect above changes, will be saved with original name "bike"
prepped_recipe <- prep(my_recipe) 
bike_clean <- bake(prepped_recipe, new_data=bike)




