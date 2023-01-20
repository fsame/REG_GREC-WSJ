require(magrittr)
require(dplyr)
require(purrr)
require(Rmpfr)
#install.packages("Rmpfr")
bfPr <- function(x) { print(x); x }
Precision <- 1024


##-----------------------------------------------------------------------##
## mkUniformN
## - makes a uniform distribution df$q over equally spaced values q between
##   0 and 1
## - df$n is uniformly N
##-----------------------------------------------------------------------##
## mkUniformN :: Integer -> DataFrame
#' Make a uniform spread of $N$ items between $0$ and $N$ in increments of $1$
#'
#' @param \code{N} an integer (default: 1000).
#' @return A dataframe with two columns: an index column \code{n} with a
#' copy of \code{N} on every row and \code{y}, and \code{q} the column with
#' values from $0.5$ to $N-0.5$.
#' @examples
#' df <- bfUniformN(100); print(as.numeric(df$q))
#' df <- bfUniformN(); print(as.numeric(df$q))
#' @export
bfUniformN <- function(N=1000) {
  data.frame( n = mpfrArray(rep(N, N), precBits=Precision, dim=c(N)),
              q = mpfrArray(((1:N) - 0.5)/N, precBits=Precision, dim=c(N)) )
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## bfBernoulliProcess
## - given the number of times each of the two
##   options occurs (cts), returns a data.frame with colName having the
##   Bernoulli probability of producing that frequency
##   of outcomes given the probability df$q of first outcome and 1-df$q of
##   the second outcome
##-----------------------------------------------------------------------##
## bfBernoulliProcess :: ColumnName -> (Integer,Integer) -> DataFrame -> DataFrame
#' Determine the probability of a particular data set for a given distribution
#' over binary outcomes given by the \code{q} column of a dataframe.
#'
#' @param \code{df} a data frame with column \code{q} of integers
#' @param \code{colName} a name for the new column of probability of
#' getting this data from a frequency \code{q}
#' @param \code{cts} a vector of two integer counts first of probability \code{1-q},
#' second of probability \code{q}
#' @return the same data frame with an added column called \code{colName}.
#' @examples
#' df1 <- data.frame(q=((1:100)-0.5)/100.0)
#' df <- bfBernoulliProcess(df1,"X",c(30,50)); print(as.numeric(df$X))
#' @export
bfBernoulliProcess <- function(df,colName,cts) {
  data.frame( df$q ** cts[1] * (1.0 - df$q) ** cts[2] ) -> dff
  colName -> colnames(dff)
  cbind(df,dff)
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## smearUnif
## - shift a column of probabilities down and up by fraction delta, add
##   the given column and put in 4 columns: nDiag (number of diagonal items
##   summed), Diag (the sum of probabilities in the diagonal strip), nOffDiag,
##   the number of off-diagonal items sumed, and OffDiag, the sum of probabilities
##   excluding those incorporated around the diagonal.
##-----------------------------------------------------------------------##
## bfSmearUnif :: DataFrame -> ColumnName -> ColumnName -> Double -> DataFrame
#' Smear a distribution uniformly over a range $\plusminus\delta$.
#'
#' @param \code{df} a data frame with column \code{q} of integers
#' @param \code{colNameX} a name of the column to be smeared
#' @param \code{delta} a real value giving the fraction of the size of the
#' dataframe to be treated as part of the diagonal
#' @return the same data frame with added columns called \code{colNameX}
#' with some suffixes: \code{.nDiag}, \code{.Diag}, \code{.nOffDiag}, \code{.OffDiag}
#' which give the count and the sum of the values for that column within and outside
#' of the diagonal band defined by \code{delta}
#' @examples
#' df1 <- data.frame(q=((1:100)-0.5)/100.0)
#' df2 <- bfBernoulliProcess(df1,"X",c(30,50))
#' df <- bfSmearUnif(df2,"X")
#' print(paste(df$q,df$X,df$X.nDiag,df$X.Diag,df$X.nOffDiag,df$X.OffDiag))
#'
#' @export
bfSmearUnif <- function(df, colNameX, delta=0.05) {
  dim(df)[1] -> N
  as.integer(delta * N) -> deltaN
  mpfrArray(rep(0,deltaN),precBits=Precision,dim=c(deltaN)) -> padding
  c(padding,df[,colNameX],padding) -> paddedCells
  sum(df[,colNameX]) -> totes
  c(padding,mpfrArray(rep(1,N),precBits=Precision,dim=c(N)),padding) -> paddedCts
  ##
  mpfrArray(rep(0,N),precBits=Precision,dim=c(N)) -> v
  for (o in (-deltaN:deltaN)) { v + paddedCells[(o+deltaN+1):(o+deltaN+N)] -> v}
  mpfrArray(rep(0,N),precBits=Precision,dim=c(N)) -> vct
  for (o in (-deltaN:deltaN)) { vct + paddedCts[(o+deltaN+1):(o+deltaN+N)] -> vct}
  totes - v -> cv
  ##
  data.frame( vct, v, mpfr(N,precBits=Precision) - vct, cv ) -> dff
  paste(colNameX,c("nDiag","Diag","nOffDiag","OffDiag"),sep=".") -> colnames(dff)
  cbind(df,dff)
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## mkBF
## - calculate the Bayes Factor of the diagonal-hugging combination of
##   parameters versus the rest
##-----------------------------------------------------------------------##
## mkBF :: DataFrame -> Real
#' Creates the Bayes' Factor comparison of two hypotheses - one that the
#'
#' @param \code{N} an integer (default: 1000).
#' @return A dataframe with two columns: an index column \code{n} with a
#' copy of \code{N} on every row and \code{y}, and \code{q} the column with
#' values from $0.5$ to $N-0.5$.
#' @examples
#' df1 <- data.frame(q=((1:100)-0.5)/100.0)
#' df2 <- bfBernoulliProcess(df1,"Y",c(30,50))
#' df3 <- bfSmearUnif(df2,"Y")
#' df4 <- bfBernoulliProcess(df3,"X",c(50,30))
#' bfBF(df4,"X","Y")
#' @export
bfBF <- function(df,colNameX,colNameY) {
  dw <- data.frame(X = df[,colNameX],
                   Diag = df[,paste(colNameY,"Diag",sep=".")],
                   nDiag = df[,paste(colNameY,"nDiag",sep=".")],
                   OffDiag = df[,paste(colNameY,"OffDiag",sep=".")],
                   nOffDiag = df[,paste(colNameY,"nOffDiag",sep=".")])
  sum( dw$X * dw$Diag    ) / sum( dw$nDiag ) -> hDiag
  sum( dw$X * dw$OffDiag ) / sum( dw$nOffDiag ) -> hOffDiag
  hDiag / hOffDiag  ## Bayes' Factor of the 1-parameter :: 2-parameter solution
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## bfEvaluation
## - turn the bases factor into a textual evaluation of the comparison
##   given statements of the two hypotheses
##-----------------------------------------------------------------------##
## bfEvaluation :: Real -> String -> String -> String
#' Maps a raw Bayes Factor ratio of probabilities onto evaluations following
#' the scale defined by Kass \and Rafferty (1995).
#'
#' @param \code{bf} a raw (not logarithmic) Bayes' Factor.
#' @param \code{numeratorHypothesis} is a string naming the numerator hypothesis.
#' @param \code{denominatorHypothesis} is a string naming the denominator hypothesis.
#' @return A string assessing the weight of the evidence for the numerator
#' or denominator hypothesis
#' @examples
#' bf <- 1.0 / (5.0 ** 6)
#' for (i in 1:13) {
#'   print(paste(bf,bfEvaluation(bf,"Numerator Hypothesis","Denominator Hypothesis")))
#'   bf <- bf * 5
#' }
#' @export
bfEvaluation <- function(bf,numeratorHypothesis,denominatorHypothesis) {
  asNumeric(bf) -> bf
  paste(ifelse((bf > 150) | (1.0/bf > 150), "Very Strong Evidence for",
               ifelse((bf >= 20) | (1.0/bf >= 20), "Strong Evidence for",
                      ifelse((bf >= 3) | (1.0/bf >= 3), "Positive Evidence for",
                             ifelse((bf >= 3) | (1.0/bf >= 3), "Barely Mentionable Evidence for",
                                    "No evidence for"
                             )))),
        ifelse(bf > 1.0,numeratorHypothesis,denominatorHypothesis),
        " (BF =",ifelse(bf > 1.0,bf,1.0/bf),")"
  )
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## bfBernoulliBF
## - puts it all togeter
## - given two absolute frequency distributions
##   a number N of values for the parameter to try,
##   and a delta to indicate how wide the diagonal belt it
##   calculate and print the Bayes' Factor evaluation of
##   the two hypotheses
##-----------------------------------------------------------------------##
## bfBernoulliBF :: Distribution2 -> Distribution2 -> Integer -> Real -> Real
#' Make a uniform spread of $N$ items between $0$ and $N$ in increments of $1$
#'
#' @param \code{ctsA} an integer (default: 1000).
#' @param \code{ctsB} an integer (default: 1000).
#' @param \code{N} as the number of divisions of the unit interval defining
#' possible parameters to the Bernoulli distribution.
#' @param \code{delta} giving the maximum difference between the two generating
#' parameters so that they still count as the same.
#' @return A string describing the weight of evidence for the binary samples
#' being from the same or different distributions (to within \code{delta}).
#' @examples
#' bfBernoulliBF(c(10,50),c(50,10))
#' bfBernoulliBF(c(50,50),c(50,50))
#' @export
bfBernoulliBF <- function(ctsA, ctsB, N=1000, delta=0.5) {
  N %>%
    bfUniformN() %>%
    bfBernoulliProcess("A",ctsA) %>%
    bfBernoulliProcess("B",ctsB) %>%
    bfSmearUnif("B",delta=delta) %>%
    bfBF("A","B")
}
##-----------------------------------------------------------------------##

##-----------------------------------------------------------------------##
## bfBernoulli
## - runs bfBernoulliBF, then produces an evaluation based on
##    Kass \and Rafferty (1995).
##-----------------------------------------------------------------------##
## bfBernoulli :: Distribution2 -> Distribution2 -> Integer -> Real -> String
#' Make a uniform spread of $N$ items between $0$ and $N$ in increments of $1$
#'
#' @param \code{ctsA} an integer (default: 1000).
#' @param \code{ctsB} an integer (default: 1000).
#' @param \code{N} as the number of divisions of the unit interval defining
#' possible parameters to the Bernoulli distribution.
#' @param \code{delta} giving the maximum difference between the two generating
#' parameters so that they still count as the same.
#' @return A string describing the weight of evidence for the binary samples
#' being from the same or different distributions (to within \code{delta}).
#' @examples
#' bfBernoulli(c(10,50),c(50,10))
#' bfBernoulli(c(50,50),c(50,50))
#' @export
bfBernoulli <- function(ctsA, ctsB, N=1000, delta=0.01) {
  bfBernoulliBF(ctsA, ctsB, N, delta) %>%
    bfEvaluation("no effect (same distribution)",
                 "effect (different distributions)")
}

##-----------------------------------------------------------------------##

