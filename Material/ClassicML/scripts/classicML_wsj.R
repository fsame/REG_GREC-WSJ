## ---------------------------
##
## Script name: classicML_wsj
##
## Purpose of script: Running the classic models on the wsj corpus
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
library(summarytools)
source("classic_ML_functions.R")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


# Map to new partitioning -------------------------------------------------

wsj <- read_rds("../data/wsj.rds")

# wsj_udel ----------------------------------------------------------------

library(party)
library(C50)
library(yardstick)

wsj_udel <- read_rds("../data/wsj_udel.rds")

wsj_udel <- wsj_udel %>% 
  select(-c(rowid, ref_id)) %>%  
  mutate_if(is.character,as.factor) 

wsj_udel_train <- wsj_udel %>% filter(type=="train") %>% select(-type)
wsj_udel_test <- wsj_udel %>% filter(type=="test") %>% select(-type)
wsj_udel_dev <- wsj_udel %>% filter(type=="dev") %>% select(-type)

set.seed(1958)

c5_udel_wsj <- C5.0(refex ~ . , data=wsj_udel_train, trials=3, rules=TRUE)

c5_udel_wsj_pred <- predict(object=c5_udel_wsj, newdata=wsj_udel_test, type="class") 

c5_udel_wsj_estimates <- truth_estimate(wsj_udel_test$refex, c5_udel_wsj_pred)

c5_udel_wsj_stat <- stats_models(c5_udel_wsj_estimates)

c5_udel_wsj_perclass <- perclass_all(c5_udel_wsj_estimates)


# wsj ICSI ----------------------------------------------------------------

library(crfsuite)

wsj_icsi <- read_rds("../data/wsj_icsi.rds")


wsj_icsi <- wsj_icsi %>% 
  select(-c(ref_id, rowid)) %>%
  mutate_if(is.factor,as.character)

wsj_icsi_train <- wsj_icsi %>% filter(type == "train") %>% select(-type)
wsj_icsi_test <- wsj_icsi %>% filter(type == "test") %>% select(-type)
wsj_icsi_dev <- wsj_icsi %>% filter(type == "dev") %>% select(-type)


indep_var_wsj <- c( "anim", "gm", "bi_w_before", "uni_w_before",
                    "uni_w_aft", "bi_w_aft", "ref_in_chain", "begin_sent_par", 
                    "punct_before", "punct_after", "morph_before", "morph_after",
                    "same_prv", "next_in_chain" )


iterations = 3000
method = "lbfgs"
##changed it from l2sgd to lbfgs

set.seed(42)

crf_icsi_wsj <- crf(y = wsj_icsi_train$refex, 
                    x = wsj_icsi_train[, indep_var_wsj], 
                    group = wsj_icsi_train$doc_id, 
                    method = method,
                    verbose = 'True',
                    all_possible_transitions= 'True', 
                    options = list(max_iterations = iterations))

crf_icsi_wsj_pred <-predict(crf_icsi_wsj, 
                            newdata = wsj_icsi_test[, indep_var_wsj], 
                            group = wsj_icsi_test$doc_id)

crf_icsi_wsj_estimates <- truth_estimate(wsj_icsi_test$refex, crf_icsi_wsj_pred$label)

crf_icsi_wsj_stat <- stats_models(crf_icsi_wsj_estimates)

crf_icsi_wsj_perclass <- perclass_all(crf_icsi_wsj_estimates)

# wsj OSU -----------------------------------------------------------------

library(nnet)

wsj_osu <- read_rds("../data/wsj_osu.rds")


wsj_osu <- wsj_osu %>% 
  select(-c(ref_id, rowid)) %>%
  mutate_if(is.character,as.factor)

wsj_osu_train <- wsj_osu %>% filter(type == "train") %>% select(-type)
wsj_osu_test <- wsj_osu %>% filter(type == "test") %>% select(-type)
wsj_osu_dev <- wsj_osu %>% filter(type == "dev") %>% select(-type)

wsj_formula <- "refex ~ anim + gm + cmpet_txt + prv_cmpet + cmpet_btwn + 
order_fctr + w_dist_fct + s_dist_fct + old_new"

set.seed(42)
maxent_osu_wsj <- multinom(wsj_formula, data=wsj_osu_train, maxit=500)

maxent_osu_wsj_pred <- predict(object=maxent_osu_wsj, newdata=wsj_osu_test, type="class")

maxent_osu_wsj_estimates <- truth_estimate(wsj_icsi_test$refex, maxent_osu_wsj_pred)

maxent_osu_wsj_stat <- stats_models(maxent_osu_wsj_estimates)

maxent_osu_wsj_perclass <- perclass_all(maxent_osu_wsj_estimates)

# wsj CNTS ----------------------------------------------------------------

library(caret)
library(class)
library(e1071)
library(FNN) 
library(gmodels) 
library(psych)

wsj_cnts <- read_rds("../data/wsj_cnts.rds")

wsj_cnts_train <- wsj_cnts2 %>% filter(type == "train") %>% select(-type)
wsj_cnts_test <- wsj_cnts2 %>% filter(type == "test") %>% select(-type)
wsj_cnts_dev <- wsj_cnts2 %>% filter(type == "dev") %>% select(-type)


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knn_cnts_wsj <- train(refex ~ ., data = wsj_cnts_train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

knn_cnts_wsj_pred <- predict(knn_cnts_wsj,newdata = wsj_cnts_test) %>% as.data.frame()

knn_cnts_wsj_estimates <- truth_estimate(wsj_cnts_test$refex, knn_cnts_wsj_pred$.)

knn_cnts_wsj_stat <- stats_models(knn_cnts_wsj_estimates)

knn_cnts_wsj_perclass <- perclass_all(knn_cnts_wsj_estimates)

# wsj ISG -----------------------------------------------------------------

library(keras)
library(recipes)

wsj_isg <- read_rds("../data/wsj_isg.rds")

wsj_isg_train <- wsj_isg2 %>% filter(type == "train") %>% select(-type)
wsj_isg_test <- wsj_isg2 %>% filter(type == "test") %>% select(-type)
wsj_isg_dev <- wsj_isg2 %>% filter(type == "dev") %>% select(-type)

recipe_wsj <- recipe(refex ~ ., data = wsj_isg_train) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = wsj_isg_train)

baked_train_wsj <- bake_recipe(wsj_isg_train,recipe_wsj)

baked_test_wsj <- bake_recipe(wsj_isg_test,recipe_wsj)

target_train_wsj <- target_handling(wsj_isg_train)
vec_target_train_wsj <- target_vector(target_train_wsj)

target_test_wsj <- target_handling(wsj_isg_test)
vec_target_test_wsj <- target_vector(target_test_wsj)

mlp_isg_wsj <- mlp_model(baked_train_wsj)

set.seed(42)
fit_mlp_isg_wsj <- fit_mlp(mlp_isg_wsj,baked_train_wsj, vec_target_train_wsj)

mlp_isg_wsj_pred <- mlp_isg_wsj %>% 
  predict(as.matrix(baked_test_wsj)) %>% 
  as.data.frame() %>% 
  rename(description = V1, name = V2, pronoun = V3) %>% 
  mutate(class_prediction = colnames(.)[max.col(.)])

mlp_isg_wsj_estimates <- truth_estimate(wsj_isg_test$refex, mlp_isg_wsj_pred$class_prediction)

mlp_isg_wsj_stat <- stats_models(mlp_isg_wsj_estimates)

mlp_isg_wsj_perclass <- perclass_all(mlp_isg_wsj_estimates)

