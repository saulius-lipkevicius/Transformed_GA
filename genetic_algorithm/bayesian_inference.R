#  Set seed hence analysis is reconstructable
set.seed(2021)
library(tidyverse)
library(LearnBayes)

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



