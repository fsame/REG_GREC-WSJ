## ---------------------------
##
## Script name: Ablation study
##
## ---------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


library(mlr3)
library(mlr3learners)
library(DALEXtra)
library(xgboost)
library(mlr)
library(tidyverse)
require(gridExtra)
library(SHAPforxgboost)

#library(SHAPforxgboost)

#___________________________________
# Directory, library, functions ----
#___________________________________

#***********************************
# read in whole data and extract features
#***********************************

#***********************************
# 3 way
#***********************************
tune_grid <- list(nrounds = 100,
                         max_depth = 5,
                         eta = 0.05,
                         gamma = 0.01,
                         colsample_bytree = 0.75,
                         min_child_weight = 0,
                         subsample = 0.5,
                         objective = "multi:softprob",  
                         num_class = 3)

explain_xgboost <- function(train,test,name){
  train_x <- train[,-1]
  train_y <- train[,1]
  encode_function <- function(X) {
    as.matrix(createDummyFeatures(X))
  }
  train_enc_x <- encode_function(train_x)
  train_enc_y <- as.numeric(train_y$refex)-1
  model<- xgboost::xgboost(data = train_enc_x, 
                   label = train_enc_y, 
                   verbose = TRUE, 
                   params = tune_grid,
                   nrounds = 1000)
  explainer <- DALEXtra::explain_xgboost(model, 
                                       train_x, 
                                       as.factor(train_y$refex), 
                                       encode_function = encode_function, 
                                       true_labels = train_y$refex, 
                                       verbose = FALSE, 
                                       label = "")
  varimp <- model_parts(explainer, B = 20,
                                type = "difference")
  #plot_varimp <- plot(varimp)
  plot_varimp <- plot(varimp,desc_sorting=TRUE , show_boxplots=FALSE, max_vars=8, title=paste("Model ", toupper(name),sep = ''), subtitle="")
  plot_varimp_2 <- plot_varimp +
    scale_y_continuous("Cross entropy loss after permutations") +
    theme(axis.text.y = element_text(angle = 0, hjust = 1, size=14,color="darkblue"))+
    theme(axis.text.x = element_text(angle = 0, hjust = 1, size=14,color="darkblue"))+
    theme(axis.title.x = element_text(color = "darkblue", size = 14, hjust = 0, vjust = 1, face = "plain"))
  ggsave(filename=paste("../featselect/varimp/", name,'_varimp.jpg',sep=''),plot_varimp_2,bg="white")
  shap <- predict_parts(explainer = explainer,
                                 new_observation = test,
                                 type = "shap",
                                 B = 2)
   plot_shap <- plot(shap)
  plot_shap_2 <- plot_shap +
    scale_y_continuous("Cross entropy loss") +
    theme(axis.text.y = element_text(angle = 0, hjust = 1, size=14,color="darkblue"))+
    theme(axis.text.x = element_text(angle = 0, hjust = 1, size=14,color="darkblue"))+
    theme(axis.title.x = element_text(color = "darkblue", size = 14, hjust = 1, vjust = 1, face = "plain"))
  ggsave(filename=paste("../featselect/shap/", name,'_shap.jpg',sep=''),plot_shap_2,width = 8,
         height = 6,units ="in",bg="white")
}



# neg_osu -----------------------------------------------------------------

feats_osu <- c("refex","GrammaticalRole","btwn_cmpt",
           "prev_cmpt","txt_cmpt","order","distance_w_disc", "distance_sent_disc",
           "old_new")

neg_osu_train <- readRDS("../data/feat_select_subsequent/neg_osu_train.rds")
neg_osu_test <- readRDS("../data/feat_select_subsequent/neg_osu_test.rds")


neg_osu_train <- neg_osu_train %>% 
  select(feats_osu) %>% 
  mutate_if(.,is.character,as.factor)

neg_osu_test <- neg_osu_test %>% 
  select(feats_osu) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(neg_osu_train,neg_osu_test,"neg_osu")


# neg_udel -----------------------------------------------------------------

neg_udel_train <- readRDS("../data/feat_select_subsequent/neg_udel_train.rds")
neg_udel_test <- readRDS("../data/feat_select_subsequent/neg_udel_test.rds")

feats_udel <- c("refex","GrammaticalRole","mention_num" ,
                "which_sent", "after_and","after_but"  ,       
                "after_then", "btwn_and","gm_prev_one",       
                "gm_prev_two","gm_prev_three", "prev_subj",         
                "btwn_cmpt","distance_sent_disc" ,"compt_in_sent" ,    
                "subj_cur", "subj_prev","subj_2prev" )


neg_udel_train <- neg_udel_train %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

neg_udel_test <- neg_udel_test %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(neg_udel_train,neg_udel_test,"neg_udel")

# neg_icsi -----------------------------------------------------------------


neg_icsi_train <- readRDS("../data/feat_select_subsequent/neg_icsi_train.rds")
neg_icsi_test <- readRDS("../data/feat_select_subsequent/neg_icsi_test.rds")

colnames(neg_icsi_train)

feats_icsi <- c("refex","GrammaticalRole",
                "uni_w_after","uni_w_before","bi_w_before","bi_w_after",
                "ref_in_chain", "next_in_chain", "par_indic", "punct_before",
                "punct_after", "morph_before", "morph_after", "same_prev")


neg_icsi_train <- neg_icsi_train %>% 
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor) 

neg_icsi_test <- neg_icsi_test %>% 
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(neg_icsi_train,neg_icsi_test,"neg_icsi")

# neg_isg -----------------------------------------------------------------


neg_isg_train <- readRDS("../data/feat_select_subsequent/neg_isg_train.rds")
neg_isg_test <- readRDS("../data/feat_select_subsequent/neg_isg_test.rds")

colnames(neg_isg_train)

feats_isg <- c("refex","GrammaticalRole","mention_num",
                "prev_refex","same_sentence","distance")

colnames(neg_isg_train)

neg_isg_train <- neg_isg_train %>% 
  rename(GrammaticalRole = SYNFUNC) %>% 
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor) 

neg_isg_test <- neg_isg_test %>% 
  rename(GrammaticalRole = SYNFUNC) %>%
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(neg_isg_train,neg_isg_test,"neg_isg")

# neg_cnts -----------------------------------------------------------------


neg_cnts_train <- readRDS("../data/feat_select_subsequent/neg_cnts_train.rds")
neg_cnts_test <- readRDS("../data/feat_select_subsequent/neg_cnts_test.rds")

colnames(neg_cnts_train)

feats_cnts <- c("refex"   ,           "GrammaticalRole" ,   "which_sent",        
                "np_num"   ,    "uni_pos_after" ,    
                "uni_pos_before"  ,   "uni_w_after"     ,   "uni_w_before",      
                "bi_w_before"     ,  "bi_pos_before"    ,  "bi_w_after"   ,     
                "bi_pos_after"    ,  "tri_w_before"     ,  "tri_pos_after" ,    
                "tri_w_after"     ,  "tri_pos_before"   ,  "root"      ,        
                "first_sent"      ,  "distance_sent"    ,  "dist_np"  ,         
                "trigram_prev_gm" ,  "diff_ref_prev_sent")


neg_cnts_train <- neg_cnts_train %>% 
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor) 

neg_cnts_test <- neg_cnts_test %>% 
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(neg_cnts_train,neg_cnts_test,"neg_cnts")

