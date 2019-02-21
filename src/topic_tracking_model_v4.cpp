// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(mvtnorm)]]
#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
//#include <progress.hpp>
//#include <mvtnormAPI.h>
#include<Rmath.h>
using namespace Rcpp;

const double log2pi = std::log(2.0 * M_PI);

arma::rowvec rcate(const arma::rowvec & p){
  int K = p.n_cols;
  arma::rowvec cump = cumsum(p);
  arma::rowvec x(K);
  x.fill(0);
  double U = R::runif(0,1);
  if(U<=cump[0]){
    x[0] = 1;
  }else{
    for(int k=1; k<K; k++){
      if(cump[k-1]<U & U<=cump[k]){
        x[k] = 1;
      }
    }
  }
  return(x);
}

double logsumexp(const arma::rowvec & x){
  double maxx = max(x);
  double out = maxx + std::log(sum(exp(x-maxx)));
  return out;
}

// [[Rcpp::export]]
arma::rowvec softmax(const arma::rowvec & x){
  double den = logsumexp(x);
  arma::rowvec res = x;
  if(arma::is_finite(den)){
    res = exp(res - den);
  }else{
    res.fill(0);
    res.elem(arma::find(x==max(x))).fill(1);
    res = res/sum(res);
  }
  return res;
}

arma::vec colSums(const arma::mat & X){
  int nCols = X.n_cols;
  arma::vec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i));
  }
  return(out);
}

arma::vec colMeans(const arma::mat & X){
  int nCols = X.n_cols;
  int nRows = X.n_rows;
  arma::vec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i))/nRows;
  }
  return(out);
}

//[[Rcpp::export]]
arma::vec weighted_colMeans(const arma::mat X, const arma::vec & w, const double & tau){
  int nCols = X.n_cols;
  arma::vec out(nCols);
  double den = sum(w);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(w % X.col(i))/(den+tau);
  }
  return(out);
}

//[[Rcpp::export]]
arma::mat weighted_colMeans_v2(Rcpp::List Y, Rcpp::List W, const double & tau, int L, int D){
  
  arma::mat tmpY = Y[0];
  int K = tmpY.n_cols; //variable number
  arma::mat out(K,L);
  for(int l=0;l<L;l++){
    for(int i=0;i<K;i++){
      double den = 0;
      double Sum = 0;
      for(int d=0;d<D;d++){
        arma::mat tmpY = Y[d];
        arma::mat tmpW = W[d];
        den += arma::sum(tmpW.col(l));
        Sum += arma::sum(tmpW.col(l) % tmpY.col(i));
      }
      out(i,l) = Sum/(den+tau);
    }  
  }
  return(out);
}

// [[Rcpp::export]]
double mvnorm_pdf(arma::vec x,
                  arma::vec mean,
                  arma::mat sigma,
                  bool log=true) {
  int xdim = x.n_rows;
  arma::mat out;
  double rootdet = -0.5 * std::log(arma::det(sigma));
  double constants = -0.5 * xdim * log2pi;
  arma::vec z = x - mean;
  out  = constants - 0.5 * z.t()*arma::inv_sympd(sigma)*z + rootdet;
  if(log){
    return arma::as_scalar(out);
  }else{
    return std::exp(arma::as_scalar(out));
  }
}

// [[Rcpp::export]]
double mvnorm_pdf_inv(arma::vec x,
                      arma::vec mean,
                      arma::mat invsigma,
                      bool log=true) {
  int xdim = x.n_rows;
  arma::mat out;
  double rootdet = 0.5 * std::log(det(invsigma));
  double constants = -0.5 * xdim * log2pi;
  arma::vec z = x - mean;
  out  = constants - 0.5 * z.t()*invsigma*z + rootdet;
  if(log){
    return arma::as_scalar(out);
  }else{
    return std::exp(arma::as_scalar(out));
  }
}

// [[Rcpp::export]]
double mvnorm_pdf_inv_det(arma::vec x,
                          arma::vec mean,
                          arma::mat invsigma,
                          double rootdet,
                          bool log=true) {
  int xdim = x.n_rows;
  double constants = -0.5 * xdim * log2pi;
  arma::vec z = x - mean;
  arma::mat out  = constants - 0.5 * z.t()*invsigma*z + rootdet;
  if(log){
    return arma::as_scalar(out);
  }else{
    return std::exp(arma::as_scalar(out));
  }
}

double mvnorm_lpdf_det(arma::vec x,
                       arma::vec mean,
                       arma::mat invsigma,
                       double rootdet){
  int xdim = x.n_rows;
  arma::mat out;
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  arma::vec A = x - mean;
  out  = constants - 0.5 * A.t()*invsigma*A + rootdet;
  return(arma::as_scalar(out));
}

// [[Rcpp::export]]
List simW(arma::mat Y, arma::mat pi, arma::mat mu,
          arma::cube sigma,int L, arma::mat minmax_id, int T){
  
  int N = Y.n_rows;
  arma::rowvec lp(L);
  arma::mat W(N,L);
  arma::vec rootdet(L);
  arma::cube invsigma=sigma;
  double ll=0;
  for(int l=0; l<L; l++){
    rootdet(l) = -std::log(det(sigma.slice(l)))/2.0;
    invsigma.slice(l) = arma::inv_sympd(sigma.slice(l)); //sigma inverse
  }
  for(int t=0;t<T;t++){
    int start = minmax_id(0,t);
    int end = minmax_id(1,t)+1;
    for(int n=start; n<end; n++){
      for(int l=0; l<L; l++){
        lp(l) = mvnorm_lpdf_det(Y.row(n).t(),mu.col(l),invsigma.slice(l),rootdet(l))+
          std::log(pi(t,l)); 
      }
      W.row(n) = rcate(softmax(lp)); //Gibbs sampling
      ll += logsumexp(lp);
    }
  }
  return List::create(W,ll);
}

// [[Rcpp::export]]
arma::rowvec pi_update(arma::mat W, arma::rowvec pre_pi, arma::vec minmax_id,
                       double alpha, int L, int T){
  
  int start = minmax_id(0);
  int end = minmax_id(1);
  arma::rowvec pi(L);
  arma::rowvec nl_t(L);
  for(int l=0; l<L; l++){
    nl_t(l) = arma::sum(W(arma::span(start,end),l));
  }
  int Nt = arma::sum(nl_t);
  for(int l=0; l<L; l++){
    pi(l) = (nl_t(l)+alpha*pre_pi(l))/(Nt+alpha);
  }
  return pi;
}

// [[Rcpp::export]]
double alpha_update(arma::mat W, double pre_alpha, arma::rowvec pre_pi, 
                    int L,arma::vec minmax_id){
  
  int start = minmax_id(0);
  int end = minmax_id(1);
  int Nt = end-start+1;
  double alpha;
  double tmp;
  arma::vec nl_t(L);
  
  for(int l=0;l<L;l++){
    nl_t(l) = arma::sum(W(arma::span(start,end),l));
  }
  tmp = 0;
  for(int l=0;l<L;l++){
    tmp += pre_pi(l)*(R::digamma(nl_t(l)+pre_alpha*pre_pi(l))-R::digamma(pre_alpha*pre_pi(l)));
  } 
  alpha = pre_alpha*tmp/(R::digamma(Nt+pre_alpha)-R::digamma(pre_alpha));
  return alpha;
}

// [[Rcpp::export]]
arma::cube sigma_update(arma::mat mu, arma::mat z, arma::mat w, int L,
                        double nu, arma::mat Lambda){
  int K = mu.n_rows;
  int N = z.n_rows;
  arma::cube Sigma(K,K,L);
  Sigma.fill(0);
  for(int l=0; l<L; l++){
    for(int n=0; n<N; n++){
      arma::vec d = z.row(n).t()-mu.col(l);
      Sigma.slice(l) += w(n,l)*d*d.t();
    }
  }
  for(int l=0; l<L; l++){
    Sigma.slice(l) = (Sigma.slice(l)+Lambda)/(sum(w.col(l)) + nu - K - 1);
  }
  return Sigma;
}

// [[Rcpp::export]]
arma::cube sigma_update_v2(arma::mat mu, Rcpp::List Y, Rcpp::List W, int L, int D,
                        double nu, arma::mat Lambda){
  int K = mu.n_rows;
  arma::cube Sigma(K,K,L);
  Sigma.fill(0);
  arma::vec nl(L);
  nl.fill(0);
  for(int l=0; l<L; l++){
    for(int d=0; d<D; d++){
      arma::mat tmpY = Y[d];
      arma::mat tmpW = W[d];
      nl(l) += arma::sum(tmpW.col(l));
      int N = tmpY.n_rows;
      for(int n=0; n<N; n++){
        arma::vec d = tmpY.row(n).t()-mu.col(l);
        Sigma.slice(l) += tmpW(n,l)*d*d.t();
      }
    }
  }
  for(int l=0; l<L; l++){
    Sigma.slice(l) = (Sigma.slice(l)+Lambda)/(nl(l) + nu - K - 1);
  }
  return Sigma;
}

// [[Rcpp::export]]
arma::vec rowSums(const arma::mat & X){
  int nRows = X.n_rows;
  arma::vec out(nRows);
  for(int i = 0; i < nRows; i++){
    out(i) = sum(X.row(i));
  }
  return(out);
}

// [[Rcpp::export]]
Rcpp::List cybertrack_noW(const arma::mat & Y, const int & L, arma::mat piini, arma::rowvec alphaini, 
                          const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, 
                          const arma::mat & Lambda, const int & num_iter, int T, arma::rowvec t_id){
  arma::mat pi = piini;
  arma::rowvec alpha = alphaini;
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  arma::vec llhist(num_iter-1);
  llhist.fill(0);
  //Progress prog(num_iter);
  List LW(2);
  
  arma::mat minmax_id(2,T);
  arma::rowvec unique_time = arma::unique(t_id);
  for(int t=0; t<T; t++){
    arma::uvec id = arma::find(t_id == unique_time(t));
    minmax_id(0,t) = arma::min(id);
    minmax_id(1,t) = arma::max(id);
  }
  for(int h=1;h<num_iter;h++){
    if(Sigma.has_nan()){
      break;
    }
    //Gibbs sampling of latent variable
    LW = simW(Y,pi,mu,Sigma,L,minmax_id,T); 
    arma::mat tmpW = LW[0];
    llhist(h-1) = LW[1];
    
    //Caluclate alpha and pi
    for(int t=0;t<T;t++){
      if(t==0){
        alpha(0) = 1;
        arma::rowvec nl(L);
        int start = minmax_id(0,t);
        int end = minmax_id(1,t);
        int Nt = end-start+1;
        for(int l=0;l<L;l++){
          nl(l) = arma::sum(tmpW(arma::span(start,end),l));
        }
        pi.row(0) = (nl+alpha(0))/(Nt+L*alpha(0));
      } else{
        alpha(t) = alpha_update(tmpW,alpha(t),pi.row(t-1),L,minmax_id.col(t));
        pi.row(t) = pi_update(tmpW,pi.row(t-1),minmax_id.col(t),alpha(t),L,T);
      }
    }
    for(int l=0;l<L;l++){
      mu.col(l) = weighted_colMeans(Y,tmpW.col(l),tau);
    }
    Sigma = sigma_update(mu,Y,tmpW,L,nu,Lambda);
    //prog.increment();
  }
  return Rcpp::List::create(Rcpp::Named("pi")=pi,_["alpha"]=alpha,_["Sigma"]=Sigma,
                            _["mu"]=mu,_["loglik"]=llhist);
}

// [[Rcpp::export]]
Rcpp::List cybertrack(const arma::mat & Y, const int & L, arma::mat piini, arma::rowvec alphaini, 
                      const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, 
                      const arma::mat & Lambda, const int & num_iter, int T, arma::rowvec t_id){
  arma::mat pi = piini;
  arma::rowvec alpha = alphaini;
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  int N = Y.n_rows;
  arma::mat W(N,L);
  arma::vec llhist(num_iter-1);
  llhist.fill(0);
  //Progress prog(num_iter);
  List LW(2);
  
  arma::mat minmax_id(2,T);
  arma::rowvec unique_time = arma::unique(t_id);
  for(int t=0; t<T; t++){
    arma::uvec id = arma::find(t_id == unique_time(t));
    minmax_id(0,t) = arma::min(id);
    minmax_id(1,t) = arma::max(id);
  }
  for(int h=1;h<num_iter;h++){
    if(Sigma.has_nan()){
      break;
    }
    //Gibbs sampling of latent variable
    LW = simW(Y,pi,mu,Sigma,L,minmax_id,T); 
    arma::mat tmpW = LW[0];
    llhist(h-1) = LW[1];
    
    //Caluclate alpha and pi
    for(int t=0;t<T;t++){
      if(t==0){
        alpha(0) = 1;
        arma::rowvec nl(L);
        int start = minmax_id(0,t);
        int end = minmax_id(1,t);
        int Nt = end-start+1;
        for(int l=0;l<L;l++){
          nl(l) = arma::sum(tmpW(arma::span(start,end),l));
        }
        pi.row(0) = (nl+alpha(0))/(Nt+L*alpha(0));
      } else{
        alpha(t) = alpha_update(tmpW,alpha(t),pi.row(t-1),L,minmax_id.col(t));
        pi.row(t) = pi_update(tmpW,pi.row(t-1),minmax_id.col(t),alpha(t),L,T);
      }
    }
    for(int l=0;l<L;l++){
      mu.col(l) = weighted_colMeans(Y,tmpW.col(l),tau);
    }
    Sigma = sigma_update(mu,Y,tmpW,L,nu,Lambda);
    W = tmpW;
    //prog.increment();
  }
  return Rcpp::List::create(Rcpp::Named("pi")=pi,_["alpha"]=alpha,_["Sigma"]=Sigma,
                            _["mu"]=mu,_["W"]=W,_["loglik"]=llhist);
}

// [[Rcpp::export]]
Rcpp::List cybertrack_v2(Rcpp::List & Y, const int & L, const int & D, arma::mat piini, arma::rowvec alphaini, 
                      const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, 
                      const arma::mat & Lambda, const int & num_iter, Rcpp::List t_id){
  
  Rcpp::List pi(D);
  Rcpp::List alpha(D);
  Rcpp::List W(D);
  for(int d=0;d<D;d++){
    pi[d] = piini;
    alpha[d] = alphaini;
  }
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  arma::mat llhist(num_iter-1,D);
  for(int h=1;h<num_iter;h++){
    for(int d=0;d<D;d++){
      arma::mat tmpY = Y[d];
      arma::mat tmppi = pi[d];
      arma::rowvec tmpalpha = alpha[d];
      arma::rowvec tmpt_id = t_id[d];
      arma::rowvec unique_time = arma::unique(tmpt_id);
      int T = unique_time.n_cols;
      arma::mat minmax_id(2,T);
      for(int t=0; t<T; t++){
        arma::uvec id = arma::find(tmpt_id == unique_time(t));
        minmax_id(0,t) = arma::min(id);
        minmax_id(1,t) = arma::max(id);
      }
      if(Sigma.has_nan()){
        break;
      }
      //Gibbs sampling of latent variable
      List LW(2);
      LW = simW(tmpY,pi[d],mu,Sigma,L,minmax_id,T); 
      arma::mat tmpW = LW[0];
      llhist(h-1,d) = LW[1];
      //Caluclate alpha and pi
      for(int t=0;t<T;t++){
        if(t==0){
          tmpalpha(0) = 1;
          int start = minmax_id(0,t);
          int end = minmax_id(1,t);
          int Nt = end-start+1;
          arma::rowvec nl(L);
          nl.fill(0);
          for(int l=0;l<L;l++){
            nl(l) = arma::sum(tmpW(arma::span(start,end),l));
          }
          tmppi.row(0) = (nl+tmpalpha(0))/(Nt+L*tmpalpha(0));
        } else{
          tmpalpha(t) = alpha_update(tmpW,tmpalpha(t),tmppi.row(t-1),L,minmax_id.col(t));
          tmppi.row(t) = pi_update(tmpW,tmppi.row(t-1),minmax_id.col(t),tmpalpha(t),L,T);
        }
      }
      pi[d] = tmppi;
      alpha[d] = tmpalpha;
      W[d] = tmpW;
    }
    mu = weighted_colMeans_v2(Y,W,tau,L,D);
    Sigma = sigma_update_v2(mu,Y,W,L,D,nu,Lambda);
  }
  return Rcpp::List::create(Rcpp::Named("pi")=pi,_["alpha"]=alpha,_["Sigma"]=Sigma,
                            _["mu"]=mu,_["W"]=W,_["loglik"]=llhist);
}

// [[Rcpp::export]]
double loglik_givW(Rcpp::List & Y, Rcpp::List & W, const int & L, const int & D, Rcpp::List & pi, 
                   const arma::mat & mu, const arma::cube & Sigma, Rcpp::List & t_id){
  //int N = Y.n_rows;
  double loglik = 0;
  arma::uvec w(1); 
  arma::vec rootdet(L);
  arma::cube invsigma=Sigma;
  //Rprintf("1 ");
  for(int l=0; l<L; l++){
    rootdet(l) = -std::log(det(Sigma.slice(l)))/2.0;
    invsigma.slice(l) = arma::inv_sympd(Sigma.slice(l)); //sigma inverse
  }
  //Rprintf("2 ");
  for(int d=0;d<D;d++){
    arma::mat tmpY = Y[d];
    arma::mat tmpW = W[d];
    arma::mat tmppi = pi[d];
    arma::rowvec tmpt_id = t_id[d];
    arma::rowvec unique_time = arma::unique(tmpt_id);
    int T = unique_time.n_cols;
    //arma::mat minmax_id(2,T);
    //minmax_id.fill(0);
    //Rprintf("3 ");
    for(int t=0; t<T; t++){
      arma::uvec id = arma::find(tmpt_id == unique_time(t));
      //minmax_id(0,t) = arma::min(id);
      //minmax_id(1,t) = arma::max(id);
      int start = arma::min(id);
      int end = arma::max(id);
      int Nt = end - start + 1;
      //Rprintf("4 ");
      for(int n=0; n<Nt; n++){
        w = arma::find(tmpW.row(n)==1);
        int z = w(0);
        loglik += mvnorm_lpdf_det(tmpY.row(n).t(),mu.col(z),invsigma.slice(z),rootdet(z))+
          std::log(tmppi(t,z));
        //Rprintf("5 ");
      }
    }
  }
  return loglik;
}

// [[Rcpp::export]]
double integ_comp_lik(Rcpp::List & Y, Rcpp::List & W, const int & L, const int & D, Rcpp::List & pi, 
                   const arma::mat & mu, const arma::cube & Sigma, Rcpp::List & t_id){
  //int N = Y.n_rows;
  double loglik = 0;
  arma::uvec w(1); 
  arma::vec rootdet(L);
  arma::cube invsigma=Sigma;
  //Rprintf("1 ");
  for(int l=0; l<L; l++){
    rootdet(l) = -std::log(det(Sigma.slice(l)))/2.0;
    invsigma.slice(l) = arma::inv_sympd(Sigma.slice(l)); //sigma inverse
  }
  //Rprintf("2 ");
  for(int d=0;d<D;d++){
    arma::mat tmpY = Y[d];
    arma::mat tmpW = W[d];
    arma::mat tmppi = pi[d];
    arma::rowvec tmpt_id = t_id[d];
    arma::rowvec unique_time = arma::unique(tmpt_id);
    int T = unique_time.n_cols;
    //arma::mat minmax_id(2,T);
    //minmax_id.fill(0);
    //Rprintf("3 ");
    for(int t=0; t<T; t++){
      arma::uvec id = arma::find(tmpt_id == unique_time(t));
      //minmax_id(0,t) = arma::min(id);
      //minmax_id(1,t) = arma::max(id);
      int start = arma::min(id);
      int end = arma::max(id);
      int Nt = end - start + 1;
      //Rprintf("4 ");
      for(int n=0; n<Nt; n++){
        w = arma::find(tmpW.row(n)==1);
        int z = w(0);
        loglik += mvnorm_lpdf_det(tmpY.row(n).t(),mu.col(z),invsigma.slice(z),rootdet(z))+
          std::log(tmppi(t,z));
        //Rprintf("5 ");
      }
    }
  }
  return loglik;
}
