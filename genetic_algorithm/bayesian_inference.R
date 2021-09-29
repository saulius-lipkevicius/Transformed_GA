#  Set seed hence analysis is reconstructable
set.seed(2021)
library(tidyverse)
library(LearnBayes)
library(data.table)

#  Determine what measure of kinetic energy is considered sufficient
#  to call an aptamer fit, in our case it is 51 for Albumin
p_albumin = 51
p_ehppdk = 35

#  Read data and add columns for later to determine proportion of fit aptamers
data = read.csv('sequences for bayes.csv')
data$albumin_prop <- ifelse(data$entropy_albumin >= p_albumin, 1, 0)
data$ehppdk_prop <- ifelse(data$entropy_ehppdk >= p_ehppdk, 1, 0)

#  Seperating target proteins data
df_albumin = as.vector(data[['albumin_prop']])
df_ehppdk = as.vector(data[['eh_prop']])

#  Measuring investigated data row number for beta distributions
data_len = len(df)
fit_albumin = sum(data$albumin_prop)
fit_ehppdk = sum(data$ehppdk_prop)


#  Beta prior - binomial likelihood case for Albumin

#  We have a believe that 0.9 quantile is value p = .006 and
#  quantile 0.5 (median) p = 0.03, using these values We can
#  determine beta(a,b) hyperparameters a and b using beta.select
quantile1 = list(p=.9, x=.006)
quantile2 = list(p=.5, x=.003)
parameters = beta.select(quantile1, quantile2)

#  Select range of p
p = seq(0, 0.015, by=0.0001)
dist1 = dbeta(p, parameters[1], parameters[2])
dist2 = dbeta(p, fit_albumin, data_len - fit_albumin)
dist3 = dbeta(p, parameters[1] + fit_albumin, parameters[2] + data_len - fit_albumin)

#  Plots prior, likelihood, posterior distributions and saves
#  it for later usage in .png format
png(filename="posterior_albumin.png")
plot(0, 0, ylim = c(0,350), col='#002733', xlim=c(0,0.015), xlab = "Proportion Coefficient p (to have a fit aptamer at random)", ylab="Density")
lines(p, dist1, lty=3, lwd = 7, col= "#1b8489")
lines(p, dist2, lty=2 , lwd = 7, col="#fccec0")
lines(p, dist3, lty=1 , lwd = 7, col="#054d54")
legend("topright", legend=c("Prior", "Likelihood", "Posterior"),
       col=c("#1b8489", "#fccec0", "#054d54"), lty = 3:1, lwd = 3, cex=1.1)
dev.off()

#  Inference from posterior distribution, determining a,b hyperparameters
#  of posterior distribution
ab = c(parameters[1] + fit_albumin, parameters[2] + data_len - fit_albumin)

#  m - number of aptamers we are randomly inferencing, ys - predicted number
#  of fit aptamers from m aptamers run
m = 1000
ys = 0:12

#  Inferencing from posterior distribution
pred = pbetap(ab, m, ys)
dt = cbind(ys, pred)

#  Density plot of predicted number of fit aptamers from m random aptamers
png(filename="aptamers_albumin.png")
plot(dt, type='h', col = '#054d54', lwd=5, ylab="Predictive Probability", xlab="Number of Fit Aptamers (per 1000 random aptamers)")
dev.off()

#  Average number of fit aptamers in m random aptamers iteration
samp = sample(size = 5000, x = dt[,1], prob=dt[,2], replace = TRUE)
mean(samp)

#  To see in what interval ypred falls with probability of 85%
discint(dt, 0.85)


##  Inferencing how many aptamers should stay in the TOP aptamer list to avoid
#   removing FIT sequences

#  Your model accuracy
model_accuracy = 0.846

#  Read data where every aptamer is compared with other and has label which is better
data = read.csv('model.csv')

#  Worksheet where you have every aptamer scored in an initial iteration, mostly it
#  should be EFBA output
pos_data = read.csv('position_analysis.csv')
true_position = pos_data[,'Power1']

#  Later this column is used to create dictionary
sequences_list = as.vector(pos_data['Sequence'])

#  Generating "possible" outcomes for prediction with NN error. We flip
#  random compared aptamers pair label using binomial distribution generated
#  0s and 1s list which later becomes new colun of Label
for (t in 1:1000) {
  
  #  Create a new column for every possible prediction with shifting error
  data[,i+3]  = rbinom(499500, size = 1, prob=c(model_accuracy))

  #  this dictionary will contain a list of aptamers with their ranking in
  #  the whole list 
  sequences <- c()

  for (i in 1:1000) {
     sequences[sequences_list[i,1]] <- 0.000
  }
  
  #  Calculating out which aptamer is superior using already compared aptamers data
  #  for more information refer to github folder ./model/README.md
  for (i in 1:dim(data)[1]) {
    if (data[i,t+3] == 1) {
      sequences[data[i,1]] <- as.numeric(sequences[data[i,1]]) + 0.001
    } else {
      sequences[data[i,2]] <- as.numeric(sequences[data[i,2]]) + 0.001
    }
    
  }  

  #  Create temp table for dictionary data in dataframe format
  temp = data.frame(names(sequences), as.numeric(sequences))

  #  Rename columns to follow up the structure
  colnames(temp) <- c("Sequence", "Power")

  #  Adding additional indexing to follow up true positioning
  temp$index <- as.numeric(row.names(temp))

  #  Order by current 'Power'
  temp = temp[order(-temp$Power), c(1,2,3)]
  
  #  Create new position column
  temp$position <- 1:nrow(temp)
  
  #  Reorder dataframe by the true index to concat dataframes easily
  temp = temp[order(temp$index), c(1,2,3,4)]

  #  Remove redundant columns, keep only Power and Position
  temp = temp[,c(2,4)]

  #  Connecting new scenario with the initial - true positioning of every aptamer
  true_position = cbind(true_position, data.frame(sequences))
  print(t) #  Follow the process since it takes ~5 hours to generate 1000 samples
}


#  Analysis of possible scenarios of aptamers in the list: how extreme can a change in
#  aptamers position in the list could be; on average, how much position changes;

#  Consider only top N aptamers, in this case N = 200
cols = seq(4,204,by=2)

#  Transpose data to easily follow specific aptamer positioning
data_t = transpose(pos_data[,cols])

#  Investigate fluctuations through quantiles of every aptamer positioning
quantile1 = stack(lapply(data_t[,c(1:200)], quantile, prob = 0.1, names = FALSE))
quantile2 = stack(lapply(data_t[,c(1:200)], quantile, prob = 0.9, names = FALSE))
top_N = 1:200

# Plot 90% confidence interval for aptamer position change
png(filename="true_error_albumin.png")
plot(y=quantile1[,1],x=top_N, type='l', col='#054d54', ylim=c(0,215), xlim=c(0,200),lwd=5, xlab="True Aptamer Position", ylab="Aptamer Position with Error")
lines(y=quantile2[,1],x=top_N, col='#054d54', lwd=5)
lines(y=top_N, x=top_N,col='#1b8489',lwd=5)
dev.off()

#  Find how many aptamers on average will be left out even if they are fit, always ~6-7
sum(ttt[,c(1:200)]>200)/100
sum(ttt[,c(1:100)]>100)/100


#  The last chapter of code can be repeated multiple times for different model accuracy