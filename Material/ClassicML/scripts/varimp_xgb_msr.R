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



# msr_osu -----------------------------------------------------------------

feats_osu <- c("refex","GrammaticalRole","SemanticCategory","btwn_cmpt",
           "prev_cmpt","txt_cmpt","order","distance_w_disc", 
           "distance_sent_disc")

msr_osu_train <- readRDS("../data/feat_select_subsequent/msr_osu_train.rds")
msr_osu_test <- readRDS("../data/feat_select_subsequent/msr_osu_test.rds")


msr_osu_train <- msr_osu_train %>% 
  select(feats_osu) %>% 
  mutate_if(.,is.character,as.factor)

msr_osu_test <- msr_osu_test %>% 
  select(feats_osu) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(msr_osu_train,msr_osu_test,"msr_osu")


# msr_udel -----------------------------------------------------------------

msr_udel_train <- readRDS("../data/feat_select_subsequent/msr_udel_train.rds")
msr_udel_test <- readRDS("../data/feat_select_subsequent/msr_udel_test.rds")

feats_udel <- c("refex","GrammaticalRole","mention_num" ,
                "which_sent", "after_and","after_but"  ,       
                "after_then", "btwn_and","gm_prev_one",       
                "gm_prev_two","gm_prev_three", "prev_subj",         
                "btwn_cmpt","distance_sent_disc" ,"compt_in_sent" ,    
                "subj_cur", "subj_prev","subj_2prev" )


msr_udel_train <- msr_udel_train %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

msr_udel_test <- msr_udel_test %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(msr_udel_train,msr_udel_test,"msr_udel")

# msr_icsi -----------------------------------------------------------------


msr_icsi_train <- readRDS("../data/feat_select_subsequent/msr_icsi_train.rds")
msr_icsi_test <- readRDS("../data/feat_select_subsequent/msr_icsi_test.rds")

colnames(msr_icsi_train)

feats_icsi <- c("refex","GrammaticalRole","SemanticCategory",
                "uni_w_after","uni_w_before","bi_w_before","bi_w_after",
                "ref_in_chain", "next_in_chain", "par_indic", "punct_before",
                "punct_after", "morph_before", "morph_after")


msr_icsi_train <- msr_icsi_train %>% 
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor) 

msr_icsi_test <- msr_icsi_test %>% 
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(msr_icsi_train,msr_icsi_test,"msr_icsi")

# msr_isg -----------------------------------------------------------------


msr_isg_train <- readRDS("../data/feat_select_subsequent/msr_isg_train.rds")
msr_isg_test <- readRDS("../data/feat_select_subsequent/msr_isg_test.rds")

colnames(msr_isg_train)

feats_isg <- c("refex","GrammaticalRole","mention_num",
                "prev_refex","same_sentence","distance")


msr_isg_train <- msr_isg_train %>% 
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor) 

msr_isg_test <- msr_isg_test %>% 
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(msr_isg_train,msr_isg_test,"msr_isg")

# msr_cnts -----------------------------------------------------------------


msr_cnts_train <- readRDS("../data/feat_select_subsequent/msr_cnts_train.rds")
msr_cnts_test <- readRDS("../data/feat_select_subsequent/msr_cnts_test.rds")

colnames(msr_cnts_train)

feats_cnts <- c("refex"   ,           "GrammaticalRole" ,   "which_sent",        
                "np_num"   ,          "SemanticCategory",   "uni_pos_after" ,    
                "uni_pos_before"  ,   "uni_w_after"     ,   "uni_w_before",      
                "bi_w_before"     ,  "bi_pos_before"    ,  "bi_w_after"   ,     
                "bi_pos_after"    ,  "tri_w_before"     ,  "tri_pos_after" ,    
                "tri_w_after"     ,  "tri_pos_before"   ,  "root"      ,        
                "first_sent"      ,  "distance_sent"    ,  "dist_np"  ,         
                "trigram_prev_gm" ,  "diff_ref_prev_sent")


msr_cnts_train <- msr_cnts_train %>% 
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor) 

msr_cnts_test <- msr_cnts_test %>% 
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor)


explain_xgboost(msr_cnts_train,msr_cnts_test,"msr_cnts")

