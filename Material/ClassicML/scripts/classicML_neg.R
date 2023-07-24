## ---------------------------
##
## Script name: classicML_neg
##
## Purpose of script: Running the classic models on the neg corpus
##
## ---------------------------

rm(list=ls())

library(tidyverse)
library(stringi)
library(stringr)
library(tidyr)
library(strex)
library(caret)
library(data.table)
library(xtable)
library(yardstick)

source("classic_ML_functions.R")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# Map to new partitioning -------------------------------------------------

neg <- read_rds("../data/neg.rds")

# neg_udel ----------------------------------------------------------------

library(party)
library(C50)
library(yardstick)

neg_udel <- read_rds("../data/neg_udel.rds")

neg_udel <- neg_udel %>% 
  select(-c(doc_id, ref_id, distance_w, ID, ref_number)) %>%  
  mutate(mention_num = as.numeric(mention_num)) %>% 
  mutate(which_sent = as.numeric(which_sent)) %>% 
  mutate_if(is.character,as.factor) 

neg_udel_train <- neg_udel %>% filter(type=="train") %>% select(-type)
neg_udel_test <- neg_udel %>% filter(type=="test") %>% select(-type)
neg_udel_dev <- neg_udel %>% filter(type=="dev") %>% select(-type)


set.seed(1958)

c5_udel_neg <- C5.0(refex ~ . , data=neg_udel_train, trials=3, rules=TRUE)

c5_udel_neg_pred <- predict(object=c5_udel_neg, newdata=neg_udel_test, type="class") 

c5_udel_neg_estimates <- truth_estimate(neg_udel_test$refex, c5_udel_neg_pred)

table(c5_udel_neg_estimates)

c5_udel_neg_stat <- stats_models(c5_udel_neg_estimates)

c5_udel_neg_perclass <- perclass_all(c5_udel_neg_estimates)


# neg ICSI ----------------------------------------------------------------

library(crfsuite)

neg_icsi <- read_rds("../data/neg_icsi.rds")

neg_icsi <- neg_icsi %>% 
  select(-c(ref_id, ID)) %>%
  mutate_if(is.factor,as.character) %>% 
  mutate_each(funs(replace(., is.na(.), -1)))

neg_icsi_train <- neg_icsi %>% filter(type == "train") %>% select(-type)
neg_icsi_test <- neg_icsi %>% filter(type == "test") %>% select(-type)
neg_icsi_dev <- neg_icsi %>% filter(type == "dev") %>% select(-type)

indep_var_neg <- c( "GrammaticalRole",  
                    "SEMCAT", 
                    "uni_w_after",      
                    "uni_w_before", 
                    "bi_w_before",      
                    "bi_w_after",       
                    "mention_num",     
                    "mention_num_ahead",    
                    "paragraph_indicator",       
                    "punct_before",     
                    "punct_after",      
                    "morph_before",     
                    "morph_after")
#,"same_prev"

iterations = 3000
method = "l2sgd"

set.seed(42)

crf_icsi_neg <- crf(y = neg_icsi_train$refex, 
                    x = neg_icsi_train[, indep_var_neg], 
                    group = neg_icsi_train$doc_id, 
                    method = method,
                    verbose = 'True',
                    all_possible_transitions= 'True', 
                    options = list(max_iterations = iterations))

crf_icsi_neg_pred <-predict(crf_icsi_neg, 
                            newdata = neg_icsi_test[, indep_var_neg], 
                            group = neg_icsi_test$doc_id)

crf_icsi_neg_estimates <- truth_estimate(neg_icsi_test$refex, crf_icsi_neg_pred$label)

crf_icsi_neg_stat <- stats_models(crf_icsi_neg_estimates)

crf_icsi_neg_perclass <- perclass_all(crf_icsi_neg_estimates)

# neg OSU -----------------------------------------------------------------

library(nnet)

neg_osu <- read_rds("../data/neg_osu.rds")

neg_osu <- neg_osu %>% 
  select(-c(doc_id, ID)) %>%
  mutate_if(is.character,as.factor)

neg_osu_train <- neg_osu %>% filter(type == "train") %>% select(-type)
neg_osu_test <- neg_osu %>% filter(type == "test") %>% select(-type)
neg_osu_dev <- neg_osu %>% filter(type == "dev") %>% select(-type)

neg_formula <- "refex ~ GrammaticalRole+ txt_cmpt + 
prev_cmpt + btwn_cmpt + distance_w_disc + order +
distance_sent_disc"

set.seed(42)
maxent_osu_neg <- multinom(neg_formula, data=neg_osu_train, maxit=500)

maxent_osu_neg_pred <- predict(object=maxent_osu_neg, newdata=neg_osu_test, type="class")

maxent_osu_neg_estimates <- truth_estimate(neg_icsi_test$refex, maxent_osu_neg_pred)

maxent_osu_neg_stat <- stats_models(maxent_osu_neg_estimates)

maxent_osu_neg_perclass <- perclass_all(maxent_osu_neg_estimates)

# neg CNTS ----------------------------------------------------------------

library(caret)
library(class)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)

neg_cnts <- read_rds("../data/neg_cnts.rds")

neg_cnts_train <- neg_cnts2 %>% filter(type == "train") %>% select(-type)
neg_cnts_test <- neg_cnts2 %>% filter(type == "test") %>% select(-type)
neg_cnts_dev <- neg_cnts2 %>% filter(type == "dev") %>% select(-type)


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knn_cnts_neg <- train(refex ~ ., data = neg_cnts_train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

knn_cnts_neg_pred <- predict(knn_cnts_neg,newdata = neg_cnts_test) %>% as.data.frame()

knn_cnts_neg_estimates <- truth_estimate(neg_cnts_test$refex, knn_cnts_neg_pred$.)

knn_cnts_neg_stat <- stats_models(knn_cnts_neg_estimates)

knn_cnts_neg_perclass <- perclass_all(knn_cnts_neg_estimates)

# neg ISG -----------------------------------------------------------------

library(keras)
library(recipes)

neg_isg <- read_rds("../data/neg_isg.rds")

neg_isg_train <- neg_isg2 %>% filter(type == "train") %>% select(-type)
neg_isg_test <- neg_isg2 %>% filter(type == "test") %>% select(-type)
neg_isg_dev <- neg_isg2 %>% filter(type == "dev") %>% select(-type)


recipe_neg <- recipe(refex ~ ., data = neg_isg_train) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = neg_isg_train)

baked_train_neg <- bake_recipe(neg_isg_train,recipe_neg)

baked_test_neg <- bake_recipe(neg_isg_test,recipe_neg)

target_train_neg <- target_handling(neg_isg_train)
vec_target_train_neg <- target_vector(target_train_neg)

target_test_neg <- target_handling(neg_isg_test)
vec_target_test_neg <- target_vector(target_test_neg)

mlp_isg_neg <- mlp_model(baked_train_neg)

set.seed(42)
fit_mlp_isg_neg <- fit_mlp(mlp_isg_neg,baked_train_neg, vec_target_train_neg)

mlp_isg_neg_pred <- mlp_isg_neg %>% 
  predict(as.matrix(baked_test_neg)) %>% 
  as.data.frame() %>% 
  rename(description = V1, name = V2, pronoun = V3) %>% 
  mutate(class_prediction = colnames(.)[max.col(.)])

mlp_isg_neg_estimates <- truth_estimate(neg_isg_test$refex, mlp_isg_neg_pred$class_prediction)

mlp_isg_neg_stat <- stats_models(mlp_isg_neg_estimates)

mlp_isg_neg_perclass <- perclass_all(mlp_isg_neg_estimates)

