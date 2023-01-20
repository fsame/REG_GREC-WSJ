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
tune_grid <- list(max_depth = 5,
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
  train_enc_y <- as.numeric(train_y)-1
  model<- xgboost::xgboost(data = train_enc_x,
                   label = train_enc_y,
                   verbose = TRUE,
                   params = tune_grid,
                   nrounds = 1000)
  explainer <- DALEXtra::explain_xgboost(model,
                                       train_x,
                                       as.factor(train_y),
                                       encode_function = encode_function,
                                       true_labels = train_y,
                                       verbose = FALSE,
                                       label = "")
  varimp <- model_parts(explainer, B = 20 ,type = "difference")
  #
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



# wsj_osu -----------------------------------------------------------------

feats_osu <- c("refex","GrammaticalRole","btwn_cmpt","SemanticCategory",
           "prev_cmpt","txt_cmpt","order","distance_w_disc", "distance_sent_disc")

wsj_osu_train <- readRDS("../data/feat_select_subsequent/wsj_osu_train.rds")
wsj_osu_test <- readRDS("../data/feat_select_subsequent/wsj_osu_test.rds")

name_change <- function(df){df %>% rename(GrammaticalRole = gm,
                                   btwn_cmpt = cmpet_btwn,
                                   prev_cmpt = prv_cmpet,
                                   txt_cmpt = cmpet_txt,
                                   order = order_fctr,
                                   distance_w_disc = w_dist_fct,
                                   distance_sent_disc = s_dist_fct,
                                   SemanticCategory = anim) %>% 
    filter(refex != "NA")
  }



wsj_osu_train <- wsj_osu_train %>%
  name_change(.) %>% 
  select(feats_osu) %>% 
  mutate_if(.,is.character,as.factor)

wsj_osu_test <- wsj_osu_test %>% 
  name_change(.) %>%
  select(feats_osu) %>% 
  filter(refex != "NA") %>% 
  mutate_if(.,is.character,as.factor)



explain_xgboost(wsj_osu_train,wsj_osu_test,"wsj_osu")


# wsj_udel -----------------------------------------------------------------

wsj_udel_train <- readRDS("../data/feat_select_subsequent/wsj_udel_train.rds")
wsj_udel_test <- readRDS("../data/feat_select_subsequent/wsj_udel_test.rds")

colnames(wsj_udel_train)
name_change <- function(df){df %>% rename(GrammaticalRole = gm,
                                          mention_num = ment_num,
                                          which_sent = sent_num,
                                          after_and = flw_and,
                                          after_but = flw_but,
                                          after_then = flw_then,
                                          "btwn_and" = btw_comma_and,
                                          "gm_prev_one" = gm_prv1 ,       
                                          "gm_prev_two" = gm_prv2,
                                          "gm_prev_three" = gm_prv3, 
                                          "prev_subj" = prv_gm_subj,         
                                          "btwn_cmpt" = interv_ref,
                                          "distance_sent_disc" = lng_short_dist ,
                                          "compt_in_sent" = interv_ent_sent,    
                                          "subj_cur" = subj_curS, 
                                          "subj_prev" = subj_prvS ,
                                          "subj_2prev" = subj_2prvS) %>% 
    filter(refex != "NA")
}

feats_udel <- c("refex","GrammaticalRole","mention_num" ,
                "which_sent", "after_and","after_but"  ,       
                "after_then", "btwn_and","gm_prev_one",       
                "gm_prev_two","gm_prev_three", "prev_subj",         
                "btwn_cmpt","distance_sent_disc" ,"compt_in_sent" ,    
                "subj_cur", "subj_prev","subj_2prev" )


wsj_udel_train <- wsj_udel_train %>% 
  name_change(.) %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

wsj_udel_test <- wsj_udel_test %>% 
  name_change(.) %>% 
  select(feats_udel) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(wsj_udel_train,wsj_udel_test,"wsj_udel")

# wsj_icsi -----------------------------------------------------------------


wsj_icsi_train <- readRDS("../data/feat_select_subsequent/wsj_icsi_train.rds")
wsj_icsi_test <- readRDS("../data/feat_select_subsequent/wsj_icsi_test.rds")

colnames(wsj_icsi_train)

name_change <- function(df){df %>% rename(GrammaticalRole = gm,
                                          SemanticCategory = anim,
                                          "uni_w_after" = uni_w_aft ,
                                          "uni_w_before" = uni_w_before ,
                                          "bi_w_before" = bi_w_before ,
                                          "bi_w_after" = bi_w_aft,
                                          "ref_in_chain" = ref_in_chain, 
                                          "next_in_chain" = next_in_chain , 
                                          "par_indic" = begin_sent_par, 
                                          "punct_before" = punct_before,
                                          "punct_after" = punct_after, 
                                          "morph_before" = morph_before , 
                                          "morph_after" = morph_after) %>% 
    filter(refex != "NA") %>% 
    mutate_each(funs(replace(., is.na(.), -1))) %>% 
    mutate(uni_w_after = as.factor(uni_w_after) %>% as.numeric(.), 
           uni_w_before = as.factor(uni_w_before) %>% as.numeric(.) ,
           bi_w_before = as.factor(bi_w_before) %>% as.numeric(.) , 
           bi_w_after = as.factor(bi_w_after) %>% as.numeric(.),
           ref_in_chain = as.numeric(ref_in_chain),
           next_in_chain = as.numeric(next_in_chain)) 
}

feats_icsi <- c("refex","GrammaticalRole", "SemanticCategory", 
                "uni_w_after","uni_w_before","bi_w_before","bi_w_after",
                "ref_in_chain", "next_in_chain", "par_indic", "punct_before",
                "punct_after", "morph_before", "morph_after")


wsj_icsi_train <- wsj_icsi_train %>% 
  name_change(.) %>% 
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor) 

wsj_icsi_test <- wsj_icsi_test %>% 
  name_change(.) %>%
  select(feats_icsi) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(wsj_icsi_train,wsj_icsi_test,"wsj_icsi")

# wsj_isg -----------------------------------------------------------------

wsj_isg_train <- readRDS("../data/feat_select_subsequent/wsj_isg_train.rds")
wsj_isg_test <- readRDS("../data/feat_select_subsequent/wsj_isg_test.rds")

colnames(wsj_isg_train)

name_change <- function(df){df %>% rename(GrammaticalRole = gm,
                                          mention_num = wch_ment,
                                          prev_refex = pr_typ,
                                          same_sentence = diff_sent,
                                          distance = dis_pr_nmrc) %>% 
    filter(refex != "NA")
}

feats_isg <- c("refex","GrammaticalRole","mention_num",
                "prev_refex","same_sentence","distance")

colnames(wsj_isg_train)

wsj_isg_train <- wsj_isg_train %>% 
  name_change(.) %>% 
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor) 

wsj_isg_test <- wsj_isg_test %>% 
  name_change(.) %>% 
  select(feats_isg) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(wsj_isg_train,wsj_isg_test,"wsj_isg")

# wsj_cnts -----------------------------------------------------------------

wsj_cnts_train <- readRDS("../data/feat_select_subsequent/wsj_cnts_train.rds")
wsj_cnts_test <- readRDS("../data/feat_select_subsequent/wsj_cnts_test.rds")


colnames(wsj_cnts_train)

name_change <- function(df){df %>% rename("SemanticCategory" = anim,
                                          "np_num"  = np_num ,
                                          "tri_w_before" = tri_w_before     ,
                                          "tri_pos_before" = tri_pos_before   ,
                                          "root" = root      ,        
                                          "first_sent"  = first_sent    ,
                                          "dist_np" = dist_np ,         
                                          "trigram_prev_gm" =  trigram_prev_gm,
                                          "diff_ref_prev_sent" = diff_ref_prev_sent) %>% 
    filter(refex != "NA")
}

feats_cnts <- c("refex"   ,           "GrammaticalRole" ,   "which_sent",        
                "np_num"   ,    "uni_pos_after" ,   "SemanticCategory", 
                "uni_pos_before"  ,   "uni_w_after"     ,   "uni_w_before",      
                "bi_w_before"     ,  "bi_pos_before"    ,  "bi_w_after"   ,     
                "bi_pos_after"    ,  "tri_w_before"     ,  "tri_pos_after" ,    
                "tri_w_after"     ,  "tri_pos_before"   ,  "root"      ,        
                "first_sent"      ,  "distance_sent"    ,  "dist_np"  ,         
                "trigram_prev_gm" ,  "diff_ref_prev_sent")


wsj_cnts_train <- wsj_cnts_train %>% 
  name_change(.) %>% 
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor) 

wsj_cnts_test <- wsj_cnts_test %>% 
  name_change(.) %>%
  select(feats_cnts) %>% 
  mutate_if(.,is.character,as.factor)

explain_xgboost(wsj_cnts_train,wsj_cnts_test,"wsj_cnts")

