library(randomForest)
library(tidyverse) #This library is needed

#importing the data
FetusDF = read.csv("C:/Users/Ronald Chitauro/Documents/Box-Docs/Python Projects/fetal_health.csv")
head(FetusDF) #for viewing just a sample of the dataset

#Checking if there is missing data
which(is.na(FetusDF))  #This shows that there is no missing Data
