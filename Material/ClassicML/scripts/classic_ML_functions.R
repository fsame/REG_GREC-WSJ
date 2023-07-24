options(yardstick.event_first = FALSE)

#Functions
stats_models <- function(model2) {model2 %>%
    conf_mat(original_value, class_prediction) %>%
    summary() %>%
    select(-.estimator) %>%
    filter(.metric %in%
             c("accuracy", "kap", "precision", "recall", "f_meas")) 
  #assign(deparse(substitute(model2)), model2, envir=.GlobalEnv)
}

truth_estimate <- function(truth, estimate) {
  tibble(
    original_value= as.factor(truth), 
    class_prediction= as.factor(estimate))
}

perclass_all <- function(dt){
  modelname = deparse(substitute(dt))
  cm = as.matrix(table(Actual = dt$original_value, Predicted = dt$class_prediction))
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  colsums[1] <- ifelse(colsums[1] == 0, 0.001, colsums)
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accuracy = round(sum(diag) / n , digits = 4)*100
  precision = round(diag / colsums, digits = 4)*100
  precision[1] <- ifelse(precision[1] == 0, 0.001, precision)
  recall = round(diag / rowsums, digits = 4) *100
  recall[1] <- ifelse(recall[1] == 0, 0.001, recall)
  f1 = round(2 * precision * recall / (precision + recall), digits = 4) 
  macroPrecision = round(mean(precision), digits=4)
  macroRecall = round(mean(recall), digits = 4)
  macroF1 = round(mean(f1), digits = 4)
  oneVsAll = lapply(1 : nc,
                    function(i){
                      v = c(cm[i,i],
                            rowsums[i] - cm[i,i],
                            colsums[i] - cm[i,i],
                            n-rowsums[i] - colsums[i] + cm[i,i]);
                      return(matrix(v, nrow = 2, byrow = T))})
  s = matrix(0, nrow = 2, ncol = 2)
  for(i in 1 : nc){s = s + oneVsAll[[i]]}
  avgAccuracy = sum(diag(s)) / sum(s)
  micro_prf = (diag(s) / apply(s,1, sum))[1];
  dt = data.frame(modelname,accuracy, classPreciosion = precision, classRecall = recall, classF1 = f1, macroPrecision, macroRecall, macroF1)
  dt <- dt %>% 
    rownames_to_column(.) %>% 
    rename(category = rowname)
}



# mlp ---------------------------------------------------------------------

bake_recipe <- function(dt,recipe){
  bake (recipe, new_data = dt) %>%
    select (-refex)
}

target_handling <- function(dt){
  dt %>% 
    dplyr::mutate(label = refex,
                  label= as.factor(label),
                  label= as.numeric(label)-1) %>% 
    select(label) 
}

target_vector <- function(dt){
  to_categorical(as.matrix(dt))
}

stats_mlp <- function(model2) {model2 %>%
    conf_mat(truth, estimate) %>%
    summary() %>%
    select(-.estimator) %>%
    filter(.metric %in%
             c("accuracy", "kap", "precision", "recall", "f_meas")) 
  #assign(deparse(substitute(model2)), model2, envir=.GlobalEnv)
}

mlp_model <- function(dt){
  keras_model_sequential() %>%  
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu", 
      input_shape        = ncol(dt)) %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(
      units              = 8,
      kernel_initializer = "uniform",
      activation         = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(
      units              = 3, 
      kernel_initializer = "uniform", 
      activation         = "sigmoid") %>% 
    compile(
      optimizer = 'adam',
      loss      = 'categorical_crossentropy',
      metrics   = c('accuracy')
    ) 
}
mlp_model_grec <- function(dt){
  keras_model_sequential() %>%  
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu", 
      input_shape        = ncol(dt)) %>% 
    layer_dropout(rate = 0.1) %>%
    # layer_dense(
    #   units              = 16,
    #   kernel_initializer = "uniform",
    #   activation         = "relu") %>%
    # layer_dropout(rate = 0.1) %>%
    layer_dense(
      units              = 8,
      kernel_initializer = "uniform",
      activation         = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(
      units              = 4, 
      kernel_initializer = "uniform", 
      activation         = "sigmoid") %>% 
    compile(
      optimizer = 'adam',
      loss      = 'categorical_crossentropy',
      metrics   = c('accuracy')
    ) 
}


fit_mlp <- function(model,dt,target){
  fit(
    object = model, 
    x = as.matrix(dt), 
    y = target,
    batch_size = 50,  epochs = 50, validation_split = 0.20
  )
}

