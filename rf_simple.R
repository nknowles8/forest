catAspect <- function(df.Aspect){
#redefines numerical aspect (0-360 degrees) as categorical with 8 levels corresponding to cardinal directions
      
      df.Aspect[df.Aspect > 337.5] <- df.Aspect[df.Aspect > 337.5] - 360
      df.Aspect <- cut(df.Aspect, seq(-22.5, 337.5, 45), labels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW"))
      
      return(df.Aspect)
      
}

my_dir <- "/home/selwonk/Documents/indie/forest"   #set appropriate directory

setwd(my_dir)

library(plyr)
library(caret)
library(doParallel)

#for parallel cpu computing
cl <-makeCluster(detectCores()/2)  
registerDoParallel(cl)

data.train <- read.csv('train.csv') 
data.test <- read.csv('test.csv')

#redefine 'Aspect'
data.train$Aspect <- catAspect(data.train$Aspect)
data.test$Aspect <- catAspect(data.test$Aspect)

#rename Cover_Type levels
data.train$Cover_Type <- revalue(as.factor(data.train$Cover_Type) , c("1" = "Type1",        
                                                                      "2" = "Type2",
                                                                      "3" = "Type3",
                                                                      "4" = "Type4",
                                                                      "5" = "Type5",
                                                                      "6" = "Type6",
                                                                      "7" = "Type7"))

#drop sparse soil types with near zero variance and/or little predictive value
dropCols <- c(22, 23, 30, 31, 34, 35, 40, 42, 43, 49)
data.train <- data.train[ , -dropCols]

#drop hillshade_3pm to reduce collinearity
data.train <-data.train[ , -10]

#set cv parameters
rf.ctrl <- trainControl(method = "repeatedcv", repeats = 3, classProbs = TRUE)
#set tuning parameter
rf.grid <- expand.grid(.mtry = 2*(7:10))

#build model
set.seed(42)
rf.model <- train(Cover_Type ~ . - Id, data = data.train, method = "rf", trControl = rf.ctrl, tuneGrid = rf.grid, ntree = 500)