## ---------------------------
##
## Script name: classicML_msr
##
## Purpose of script: Running the classic models on the msr corpus
##
## ---------------------------

rm(list=ls())


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(stringi)
library(stringr)
library(tidyr)
library(strex)
library(caret)
library(data.table)
library(xtable)
library(yardstick)
library("psych")
source("classic_ML_functions.R")




# Directory ---------------------------------------------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


# Map to new partitioning -------------------------------------------------

msr<- read_rds("../data/msr.rds")

# msr_udel ----------------------------------------------------------------

library(party)
library(C50)


msr_udel <- read_rds("../data/msr_udel.rds")

msr_udel_train <- msr_udel %>% filter(type=="train") %>% select(-type)
msr_udel_test <- msr_udel %>% filter(type=="test") %>% select(-type)
msr_udel_dev <- msr_udel %>% filter(type=="dev") %>% select(-type)


set.seed(1958)

c5_udel_msr <- C5.0(refex ~ . , data=msr_udel_train, trials=3, rules=TRUE)

c5_udel_msr_pred <- predict(object=c5_udel_msr, newdata=msr_udel_test, type="class") 

c5_udel_msr_estimates <- truth_estimate(msr_udel_test$refex, c5_udel_msr_pred)

c5_udel_msr_stat <- stats_models(c5_udel_msr_estimates)


# msr_icsi ----------------------------------------------------------------

library(crfsuite)

msr_icsi <- read_rds("../data/msr_icsi.rds")

msr_icsi <- msr_icsi %>% 
  select(-c(ref_id, ID)) %>%
  mutate_if(is.factor,as.character) 

msr_icsi_train <- msr_icsi %>% filter(type == "train") %>% select(-type)
msr_icsi_test <- msr_icsi %>% filter(type == "test") %>% select(-type)
msr_icsi_dev <- msr_icsi %>% filter(type == "dev") %>% select(-type)


indep_var_msr <- c( "GrammaticalRole",  
                    "SemanticCategory", 
                    "uni_w_after",      
                    "uni_w_before", 
                    "bi_w_before",      
                    "bi_w_after",       
                    "ref_in_chain",     
                    "next_in_chain",    
                    "par_indic",       
                    "punct_before",     
                    "punct_after",      
                    "morph_before",     
                    "morph_after" )

iterations = 3000
method = "l2sgd"

set.seed(42)

crf_icsi_msr <- crf(y = msr_icsi_train$refex, 
                    x = msr_icsi_train[, indep_var_msr], 
                    group = msr_icsi_train$doc_id, 
                    method = method,
                    verbose = 'True',
                    all_possible_transitions= 'True', 
                    options = list(max_iterations = iterations))

crf_icsi_msr_pred <-predict(crf_icsi_msr, 
                            newdata = msr_icsi_test[, indep_var_msr], 
                            group = msr_icsi_test$doc_id)

crf_icsi_msr_estimates <- truth_estimate(msr_icsi_test$refex, crf_icsi_msr_pred$label)

crf_icsi_msr_stat <- stats_models(crf_icsi_msr_estimates)



# MSR OSU -----------------------------------------------------------------

library(nnet)

msr_osu <- read_rds("../data/msr_osu.rds")


msr_osu <- msr_osu %>% 
  select(-c(doc_id, ID)) %>%
  mutate_if(is.character,as.factor)

msr_osu_train <- msr_osu %>% filter(type == "train") %>% select(-type)
msr_osu_test <- msr_osu %>% filter(type == "test") %>% select(-type)
msr_osu_dev <- msr_osu %>% filter(type == "dev") %>% select(-type)

msr_formula <- "refex ~ GrammaticalRole+ SemanticCategory + txt_cmpt + 
prev_cmpt + btwn_cmpt + distance_w_disc + order + distance_sent_disc+ old_new"

set.seed(42)
maxent_osu_msr <- multinom(msr_formula, data=msr_osu_train, maxit=500)

maxent_osu_msr_pred <- predict(object=maxent_osu_msr, newdata=msr_osu_test, type="class")

maxent_osu_msr_estimates <- truth_estimate(msr_icsi_test$refex, maxent_osu_msr_pred)

maxent_osu_msr_stat <- stats_models(maxent_osu_msr_estimates)

#   1 accuracy      0.680
# 2 kap           0.465
# 3 precision     0.641
# 4 recall        0.595
# 5 f_meas        0.603

maxent_osu_msr_perclass <- perclass_all(maxent_osu_msr_estimates)

# MSR CNTS ----------------------------------------------------------------

library(caret)
library(class)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)

msr_cnts <- read_rds("../data/msr_cnts.rds")

msr_cnts_train <- msr_cnts2 %>% filter(type == "train") %>% select(-type)
msr_cnts_test <- msr_cnts2 %>% filter(type == "test") %>% select(-type)
msr_cnts_dev <- msr_cnts2 %>% filter(type == "dev") %>% select(-type)


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knn_cnts_msr <- train(refex ~ ., data = msr_cnts_train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

knn_cnts_msr_pred <- predict(knn_cnts_msr,newdata = msr_cnts_test) %>% as.data.frame()

knn_cnts_msr_estimates <- truth_estimate(msr_cnts_test$refex, knn_cnts_msr_pred$.)

knn_cnts_msr_stat <- stats_models(knn_cnts_msr_estimates)

knn_cnts_msr_perclass <- perclass_all(knn_cnts_msr_estimates)

# MSR ISG -----------------------------------------------------------------

library(keras)
library(recipes)

msr_isg <- read_rds("../data/msr_isg.rds")

msr_isg_train <- msr_isg2 %>% filter(type == "train") %>% select(-type)
msr_isg_test <- msr_isg2 %>% filter(type == "test") %>% select(-type)
msr_isg_dev <- msr_isg2 %>% filter(type == "dev") %>% select(-type)

recipe_msr <- recipe(refex ~ ., data = msr_isg_train) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = msr_isg_train)

baked_train_msr <- bake_recipe(msr_isg_train,recipe_msr)

baked_test_msr <- bake_recipe(msr_isg_test,recipe_msr)

target_train_msr <- target_handling(msr_isg_train)
vec_target_train_msr <- target_vector(target_train_msr)

target_test_msr <- target_handling(msr_isg_test)
vec_target_test_msr <- target_vector(target_test_msr)

mlp_isg_msr <- mlp_model(baked_train_msr)

set.seed(42)
fit_mlp_isg_msr <- fit_mlp(mlp_isg_msr,baked_train_msr, vec_target_train_msr)

mlp_isg_msr_pred <- mlp_isg_msr %>% 
  predict(as.matrix(baked_test_msr)) %>% 
  as.data.frame() %>% 
  rename(description = V1, name = V2, pronoun = V3) %>% 
  mutate(class_prediction = colnames(.)[max.col(.)])

mlp_isg_msr_estimates <- truth_estimate(msr_isg_test$refex, mlp_isg_msr_pred$class_prediction)

mlp_isg_msr_stat <- stats_models(mlp_isg_msr_estimates)

mlp_isg_msr_perclass <- perclass_all(mlp_isg_msr_estimates)
