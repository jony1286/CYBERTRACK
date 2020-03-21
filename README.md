# CYBERTRACK
CYBERTRACK is an R package for CYtometry-Based Estimation and Reasoning for cell population TRACKing

###Depends:

R(>=3.5.2)

Rcpp, RcppARmadillo

###Authors:

Kodai Minoura and Ko Abe

Contact: minoura.kodai[at]e.mbox.nagoya-u.ac.jp and ko.abe[at]med.nagoya-u.ac.jp

## Installation

Install the latest version of this package from Github by pasting in the following.

~~~R
devtools::install_github("kodaim1115/CYBERTRACK")
~~~

## An example of synthetic data

~~~R
library(CYBERTRACK)
library(mvtnorm)

set.seed(1234)
sample <- 1000000

L <- 3 #number of clusters 
K <- 2 #number of variables 
N <- 1000 #samples at each timepoint
T <- 5 #number of timepoints
D <- 2 #number of cases

true_pi <- list()
true_pi[[1]] <- matrix(c(0.3,0.3,0.4,
                         0.3,0.3,0.4,
                         0.2,0.5,0.3,
                         0.2,0.5,0.3,
                         0.8,0.1,0.1),L,T,byrow=FALSE)
true_pi[[2]] <- matrix(c(0.4,0.3,0.3,
                         0.1,0.1,0.8,
                         0.1,0.1,0.8,
                         0.2,0.2,0.6,
                         0.2,0.2,0.6),L,T,byrow=FALSE)

true_mu <- matrix(c(-3,2,
                    2,1,
                    0,-3),L,K)
true_sigma <- array(,dim=c(K,K,L))
for(i in 1:L) true_sigma[,,i] <- diag(K)

Y <- t_id <- list()
for(d in 1:D){
  mn <- list()
  for(i in 1:L) mn[[i]] <- mvrnorm(sample,true_mu[i,],true_sigma[,,i]) 
  junk <- pi_id <- id <- list()
  for(t in 1:T){
    junk[[t]] <- matrix(NA_real_,N,K)
    pi_id[[t]] <- sample(1:L,N,replace=TRUE,prob=true_pi[[d]][,t])
    id[[t]] <- sample(1:sample,N,replace=TRUE)
    for(i in 1:N){
      junk[[t]][i,] <- mn[[pi_id[[t]][i]]][id[[t]][i],] 
    }
  }  
  Y[[d]] <- do.call(rbind,junk)
  t_id[[d]] <- rep(1:T,each=N)
}

kminit <- function(y,L,seed = sample.int(.Machine$integer.max, 1)){
  set.seed(seed)
  kmres <- kmeans(y,L,iter.max=100,nstart=3,algorithm="Lloyd")
  list(mean=t(kmres$centers),
       var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
       cluster=kmres$cluster)
}
kmY <- do.call(rbind,Y)

num_iter <- 100
tau <- 0.1
nu <- K+1
Lambda <- diag(K)
kmpar <- kminit(kmY,L,123)
piini <- matrix(1/L,T,L)
alphaini <- c(rep(1,T))
muini <- kmpar$mean
Sigmaini <- kmpar$var

result <- cybertrack(Y,L,D,piini,alphaini,muini,Sigmaini,tau,nu,Lambda,num_iter,t_id)
~~~

## Genral overview
We propose new statistical framework called CYBERTRACK.

CYBERTRACK enables cell clustering, population tracking, and change-point detection in overall mixture proportion for time-series flowc ytometry data.

In CYBERTRACK, flow cytometry data is assumed to be generated from a multivariate Gaussian mixture distribution.

## Reference
Minoura, K., Abe, K., Maeda, Y., Nishikawa, H., & Shimamura, T. (2019). Model-based cell clustering and population tracking for time-series flow cytometry data. BMC bioinformatics, 20(23), 1-10.
