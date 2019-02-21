\name{cybertrack_v2}
\alias{cybertrack_v2}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
parameter estimation
}
\description{
hoge
}
\usage{
cybertrack_v2(Y, L, D, piini, alphaini, muini, Sigmaini, tau, nu, Lambda, num_iter, t_id)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
%%     ~~Describe \code{Y} here~~
}
  \item{L}{
%%     ~~Describe \code{L} here~~
}
  \item{D}{
%%     ~~Describe \code{D} here~~
}
  \item{piini}{
%%     ~~Describe \code{piini} here~~
}
  \item{alphaini}{
%%     ~~Describe \code{alphaini} here~~
}
  \item{muini}{
%%     ~~Describe \code{muini} here~~
}
  \item{Sigmaini}{
%%     ~~Describe \code{Sigmaini} here~~
}
  \item{tau}{
%%     ~~Describe \code{tau} here~~
}
  \item{nu}{
%%     ~~Describe \code{nu} here~~
}
  \item{Lambda}{
%%     ~~Describe \code{Lambda} here~~
}
  \item{num_iter}{
%%     ~~Describe \code{num_iter} here~~
}
  \item{t_id}{
%%     ~~Describe \code{t_id} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (Y, L, D, piini, alphaini, muini, Sigmaini, tau, nu, 
    Lambda, num_iter, t_id) 
{
    .Call(`_CYBERTRACK_cybertrack_v2`, Y, L, D, piini, alphaini, 
        muini, Sigmaini, tau, nu, Lambda, num_iter, t_id)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
