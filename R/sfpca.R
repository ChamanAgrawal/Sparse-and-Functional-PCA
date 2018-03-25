# Calculates rank-1 approximated principal components

NULL
library(RcppEigen)

sfpca <- function(X,
                  lamu = 0,
                  lamv = 0,
                  alphau = 0,
                  alphav = 0,
                  Omegu = diag(nrow(X)),
                  Omegv = diag(ncol(X))){
  n <- nrow(X)
  p <- ncol(X)
  
  stopifnot(lamu >= 0)
  stopifnot(lamv >= 0)
  stopifnot(alphau >= 0)
  stopifnot(alphav >= 0)
  stopifnot(all(eigen(Omegu)$values >= 0))  # check for positive semi-definite
  stopifnot(all(eigen(Omegv)$values >= 0))
  stopifnot(dim(Omegu) == c(n, n))
  stopifnot(dim(Omegv) == c(p, p))
  
  sfpca_fixed(X, lamu, lamv, alphau, alphav, Omegu, Omegv)
}
