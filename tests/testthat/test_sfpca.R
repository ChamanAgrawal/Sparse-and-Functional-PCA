context("test_sfpca.R")


check_with_matlab <- function(testfile){
  
  test <- R.matlab::readMat(testfile)
  result <- sfpca_fixed(X = test$x,
                        lam_u = as.numeric(test$lamu),
                        lamv = as.numeric(test$lamv),
                        alphau = as.numeric(test$alphau),
                        alphav = as.numeric(test$alphav),
                        Omegu = test$Omegu,
                        Omegv = test$Omegv)
  
  expect_equal(test$U, res_r$u, tolerance = 1e-5)
  expect_equal(test$V, res_r$v, tolerance = 1e-5)
  expect_equal(as.numeric(test$d), res_r$d, tolerance = 1e-5)
}

test_that("Input checking", {
  
  X <- as.matrix(iris[, 1:4])
  
  expect_error(sfpca(X, lambda_u = -1))
  expect_error(sfpca(X, lambda_v = -1))
  expect_error(sfpca(X, alpha_u = -1))
  expect_error(sfpca(X, alpha_v = -1))
  
  n <- nrow(X)
  p <- ncol(X)
  
  expect_error(sfpca(X, Omega_u = -1 * diag(n)))  # negative definite
  expect_error(sfpca(X, Omega_v = -1 * diag(p)))
  expect_error(sfpca(X, Omega_u = diag(n + 1)))  # wrong size
  expect_error(sfpca(X, Omega_u = diag(p + 1)))
})

test_that("Matches PCA when not regularization applied", {
  X <- as.matrix(iris[, 1:4])
  
  res_sfpca <- sfpca(X)
  res_svd <- svd(X)
  
  expect_equivalent(res_svd$u[, 1], res_sfpca$u)
  expect_equivalent(res_svd$v[, 1], res_sfpca$v)
  expect_equal(res_svd$d[1], res_sfpca$d)
})

test_that("Agrees with MATLAB implementation", {
  expect_agrees_with_matlab("test1.mat")
  expect_agrees_with_matlab("test2.mat")
  expect_agrees_with_matlab("test3.mat")
})