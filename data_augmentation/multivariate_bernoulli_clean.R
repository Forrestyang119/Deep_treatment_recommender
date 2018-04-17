###
# @Author: Sen Yang
# @Function: load alignment matrix and generate synthetic sequence data
###

# install.packages('cmm')
# install.packages('Rsolnp')
# install.packages('numDeriv')
# install.packages('mipfp')
# install.packages('BinNor')
# install.packages('Matrix')
# install.packages('mnormt')
# install.packages('corpcor')
# install.packages('mvtnorm')
# install.packages('psych')
# https://cran.r-project.org/web/packages/BinNor/BinNor.pdf

library('cmm')
library('Rsolnp')
library('numDeriv')
library('mipfp')
library('mnormt')
library('Matrix')
library('BinNor')
library('corpcor')
library('mvtnorm')
library('psych')

load_data <- function(file_path) {
  mydata = read.csv(file_path, header=FALSE)
  return(mydata)
}

prob_col <- function(data, size) {
  for (row in 1:nrow(data)) {
    for (col in 1:ncol(data)) {
      if (data[row, col] <= 0.5) {
        data[row, col] = 0;
      } else {
        data[row, col] = 1;
      }
    }
  }
  p <- colSums(data)
  p <- p/size
  return(p)
}

data_wrangling <- function(mydata) {
  mydata = mydata[, 2:ncol(mydata)] # drop the first column (i.e., case ids)
  mydata[mydata == 1] <- 0
  mydata[mydata == 2] <- 1
  
  p = prob_col(mydata, nrow(mydata)) # calculate the vector probability
  
  ones = which(p %in% c(0)) # find columns with prob == 1, note columns of all 1s has p == 0
  mydata[, ones] = 1
  
  # print(p)
  print(ones)
  print(length(ones))
  # print(ncol(mydata))
  if (length(ones) != 0){
    flt_data = mydata[, -ones] # to drop columns 
  } else {
    flt_data = mydata
  }
  
  flt_p = p[p != 0] # drop 1s
  
  print(ncol(flt_data))
  print(length(flt_p))
  print(ncol(mydata))
  print(length(p))

  return(list("flt_data" = flt_data, "flt_p" = flt_p, "mydata" = mydata, "p" = p, "ones" = ones))
}

cal_corr <- function(data) {
  # https://en.wikipedia.org/wiki/Phi_coefficient
  # phi correlation
  corr = matrix(0, ncol(data), ncol(data))
  for (col1 in 1:ncol(data)) {
    print(col1)
    for (col2 in col1:ncol(data)) {
      x_1_1 = 0
      x_1_0 = 0
      x_0_1 = 0
      x_0_0 = 0
      data_1 = data[,col1]
      data_2 = data[,col2]
      x_1_1 = sum(data_1 + data_2 == 2)
      # x_0_0 = sum(data_1 + data_2 == 0)
      x_1_dot = sum(data_1 == 1)
      x_0_dot = sum(data_1 == 0)
      x_dot_1 = sum(data_2 == 1)
      x_dot_0 = sum(data_2 == 0)
      n_x = nrow(data)
      
      corr[col1, col2] = (x_1_1 * n_x - x_1_dot * x_dot_1)/(sqrt(x_1_dot) * sqrt(x_dot_1) * sqrt(x_0_dot) * sqrt(x_dot_0))
    }  
  }
  return(corr)
}

toSymmetric <- function(m) {
  m[lower.tri(m)] <- t(m)[lower.tri(m)]
  return(m)
}

results_print <- function(synthetic, mydata) {
  data_row = rowSums(mydata != 0)
  syn_row = rowSums(synthetic != 0)
  cat(mean(data_row), sd(data_row))
  print(" ")
  cat(mean(syn_row), sd(syn_row))
  
  mean = mean(syn_row)
  std = sd(syn_row)
  
  return(list("mean" = mean, "std" = std))
}

generate_synthetic_sequences <- function(n, file_path) {
  # n: num of synthetic sequences to generate
  # file_path: alginment matrix file path
  
  mydata = load_data(file_path)
  mydata = data.matrix(mydata) # convert to matrix format

  lst = data_wrangling(mydata)

  flt_data = lst$flt_data
  flt_p = lst$flt_p
  data = lst$mydata
  p = lst$p
  ones = lst$ones

  corr = cal_corr(flt_data)
  corr = toSymmetric(corr) # calculate correlation matrix
  print(corr[3,9])
  print(flt_data[, 3])
  print(flt_data[, 9])
  
  synthetic = jointly.generate.binary.normal(n, ncol(flt_data), no.nor = 0, prop.vec.bin = flt_p,
                                        sigma_star=corr,
                                        continue.with.warning=TRUE)

  for (col in ones) { # insert the 1st back
    c = matrix(1, nrow(synthetic), 1) # create a column of 1s
    # print(ncol(synthetic))
    # print(col)
    if (col == 1) { # the 1st column
      synthetic = cbind(c, synthetic)
    } else if (col == ncol(synthetic) + 1) { # the last column
      synthetic = cbind(synthetic, c)
    } else {
      synthetic = cbind(synthetic[,1:col-1], c, synthetic[,col:ncol(synthetic)])
    }

  }

  res = results_print(synthetic, data)
  mean = res$mean
  std = res$std
  return(list("synthetic" = synthetic, "authetic" = data))
}

# print(ncol(trauma_mat)-2)
# print(length(p[3:length(p)-1]))
# print(ncol(corr))
# res = generate_synthetic_sequences(1000, "Files_synthetic/alignment.csv")
# res = generate_synthetic_sequences(5000, "Alignment_intubate_101/PIMA_alignment.csv")
# res = generate_synthetic_sequences(5000, "Alignment_trauma_99/PIMA_alignment.csv")
# 
# # check duplicates
# x = res$synthetic
# df = data.frame(x)
# dup = which(duplicated(df) | duplicated(df[nrow(df):1, ])[nrow(df):1])
# print(length(dup))

