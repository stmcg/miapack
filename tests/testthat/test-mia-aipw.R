test_that("mia_aipw returns finite point estimates with GLM nuisances", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1), X_values_2 = c(0, 0),
                  contrast_type = 'difference',
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 5)

  expect_true(is.finite(res$mean_est_1))
  expect_true(is.finite(res$mean_est_2))
  expect_true(is.finite(res$contrast_est))
  expect_equal(res$contrast_est,
               res$mean_est_1 - res$mean_est_2,
               tolerance = 1e-12)
  expect_equal(length(res$fit_Y), 5)
  expect_equal(length(res$fit_pi), 5)
  expect_equal(length(res$fit_outer), 5)
})

test_that("mia_aipw estimate is the average of fold-specific Step-B predictions", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 5)

  fold_predictions <- vapply(
    res$fit_outer,
    function(fit_outer){
      get_aipw_mean(X_values = c(0, 1),
                    X_names = c("X1", "X2"),
                    fit_outer = fit_outer)
    },
    numeric(1)
  )

  expect_equal(res$mean_est_1, mean(fold_predictions), tolerance = 1e-12)
  expect_equal(names(res$mean_est_1_fold),
               paste0('fold_', seq_len(res$n_folds)))
  expect_equal(res$mean_est_1_fold, fold_predictions, tolerance = 1e-12)
})

test_that("mia_aipw pseudo-outcomes are aligned with the original data", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 5)

  eligible <- stats::complete.cases(dat.sim[, c("X1", "X2", "W")])

  expect_equal(length(res$Q_hat), nrow(dat.sim))
  expect_equal(length(res$pi_hat), nrow(dat.sim))
  expect_equal(length(res$psi_hat), nrow(dat.sim))
  expect_equal(length(res$fold_id), nrow(dat.sim))

  expect_true(all(is.na(res$Q_hat[!eligible])))
  expect_true(all(is.na(res$pi_hat[!eligible])))
  expect_true(all(is.na(res$psi_hat[!eligible])))
  expect_true(all(is.na(res$fold_id[!eligible])))

  expect_true(all(is.finite(res$Q_hat[eligible])))
  expect_true(all(is.finite(res$pi_hat[eligible])))
  expect_true(all(is.finite(res$psi_hat[eligible])))
})

test_that("mia_aipw sets the augmentation contribution to zero when Y is missing", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 5)

  eligible_missing_Y <- stats::complete.cases(dat.sim[, c("X1", "X2", "W")]) &
    is.na(dat.sim$Y)

  expect_equal(res$psi_hat[eligible_missing_Y],
               res$Q_hat[eligible_missing_Y],
               tolerance = 1e-12)
})

test_that("mia_aipw supports SuperLearner nuisances", {
  skip_if_not_installed("SuperLearner")

  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  Y_method = 'SuperLearner',
                  pi_method = 'SuperLearner',
                  Y_SL_library = c('SL.glm', 'SL.gam'),
                  pi_SL_library = c('SL.glm', 'SL.gam'),
                  n_folds = 2)

  expect_true(is.finite(res$mean_est_1))
  expect_true(all(is.finite(res$psi_hat[!is.na(res$psi_hat)])))
})

test_that("mia_aipw errors on missing pi_model", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             outer_model = psi_hat ~ X1 * X2),
    "pi_model must be supplied"
  )
})

test_that("mia_aipw errors on pi_model with wrong LHS", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = missing_Y ~ W + X1 + X2,
             outer_model = psi_hat ~ X1 * X2),
    "The left-hand side of pi_model must be 'R_Y'"
  )
})

test_that("mia_aipw errors when pi_model introduces an auxiliary not in Y_model", {
  dat_temp <- dat.sim
  dat_temp$W2 <- stats::rnorm(nrow(dat_temp))

  expect_error(
    mia_aipw(data = dat_temp,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + W2 + X1 + X2,
             outer_model = psi_hat ~ X1 * X2),
    "The right-hand side of pi_model can only depend"
  )
})

test_that("mia_aipw errors on outer_model with wrong LHS", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + X1 + X2,
             outer_model = g_hat ~ X1 * X2),
    "The left-hand side of outer_model must be 'psi_hat'"
  )
})

test_that("mia_aipw errors when outer_model RHS is not in X_names", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + X1 + X2,
             outer_model = psi_hat ~ W),
    "The right-hand side of outer_model can only depend on the predictor"
  )
})

test_that("mia_aipw errors on invalid nuisance estimation methods", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + X1 + X2,
             outer_model = psi_hat ~ X1 * X2,
             Y_method = 'invalid'),
    "Y_method must be set"
  )

  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + X1 + X2,
             outer_model = psi_hat ~ X1 * X2,
             pi_method = 'invalid'),
    "pi_method must be set"
  )
})

test_that("mia_aipw errors on invalid number of folds", {
  expect_error(
    mia_aipw(data = dat.sim,
             X_names = c("X1", "X2"), X_values_1 = c(0, 1),
             Y_model = Y ~ W + X1 + X2,
             pi_model = R_Y ~ W + X1 + X2,
             outer_model = psi_hat ~ X1 * X2,
             n_folds = 0),
    "n_folds must be an integer greater than or equal to 1"
  )
})


test_that("get_CI clearly rejects mia_aipw objects while inference is not implemented", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 2)

  expect_error(
    get_CI(res, n_boot = 10, show_progress = FALSE),
    "Confidence intervals are not currently implemented for mia_aipw objects"
  )
})


test_that("mia_aipw constructs the AIPW pseudo-outcome correctly", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  n_folds = 5)

  eligible_observed_Y <- stats::complete.cases(dat.sim[, c("X1", "X2", "W", "Y")])
  expected <- res$Q_hat[eligible_observed_Y] +
    (dat.sim$Y[eligible_observed_Y] - res$Q_hat[eligible_observed_Y]) /
    res$pi_hat[eligible_observed_Y]

  expect_equal(res$psi_hat[eligible_observed_Y], expected, tolerance = 1e-12)
})

test_that("mia_aipw handles fully observed Y correctly", {
  dat_complete_Y <- dat.sim
  dat_complete_Y$Y[is.na(dat_complete_Y$Y)] <- 0

  set.seed(1234)

  res <- mia_aipw(
    data = dat_complete_Y,
    X_names = c("X1", "X2"),
    X_values_1 = c(0, 1),
    Y_model = Y ~ W + X1 + X2,
    pi_model = R_Y ~ W + X1 + X2,
    outer_model = psi_hat ~ X1 * X2,
    n_folds = 5
  )

  eligible <- complete.cases(
    dat_complete_Y[, c("X1", "X2", "W")]
  )

  expect_length(res$fit_pi, 5)
  expect_equal(
    names(res$fit_pi),
    paste0("fold_", 1:5)
  )
  expect_true(
    all(vapply(res$fit_pi, is.null, logical(1)))
  )

  expect_true(all(res$pi_hat[eligible] == 1))

  expect_equal(
    res$psi_hat[eligible],
    dat_complete_Y$Y[eligible],
    tolerance = 1e-12
  )
})


test_that("mia_aipw supports the marginal-mean special case", {
  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = character(0),
                  X_values_1 = numeric(0),
                  Y_model = Y ~ W,
                  pi_model = R_Y ~ W,
                  outer_model = psi_hat ~ 1,
                  n_folds = 5)

  expect_true(is.finite(res$mean_est_1))
  expect_equal(res$mean_est_1,
               mean(res$mean_est_1_fold),
               tolerance = 1e-12)
})


test_that("mia_aipw defaults to no cross-fitting for GLM nuisances", {
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2)

  eligible <- stats::complete.cases(dat.sim[, c("X1", "X2", "W")])

  expect_equal(res$n_folds, 1L)
  expect_equal(res$Y_method, 'glm')
  expect_equal(res$pi_method, 'glm')
  expect_length(res$fit_Y, 1)
  expect_length(res$fit_pi, 1)
  expect_length(res$fit_outer, 1)
  expect_true(all(res$fold_id[eligible] == 1L))
})


test_that("mia_aipw defaults to five-fold cross-fitting with SuperLearner", {
  skip_if_not_installed("SuperLearner")

  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  Y_method = 'SuperLearner')

  expect_equal(res$n_folds, 5L)
  expect_equal(res$Y_method, 'SuperLearner')
  expect_equal(res$pi_method, 'glm')
  expect_equal(res$Y_SL_library, c('SL.glm', 'SL.gam'))
  expect_length(res$fit_Y, 5)
  expect_length(res$fit_pi, 5)
  expect_length(res$fit_outer, 5)
})


test_that("mia_aipw defaults to five-fold cross-fitting when pi uses SuperLearner", {
  skip_if_not_installed("SuperLearner")

  set.seed(1234)
  res <- mia_aipw(data = dat.sim,
                  X_names = c("X1", "X2"),
                  X_values_1 = c(0, 1),
                  Y_model = Y ~ W + X1 + X2,
                  pi_model = R_Y ~ W + X1 + X2,
                  outer_model = psi_hat ~ X1 * X2,
                  pi_method = 'SuperLearner')

  expect_equal(res$n_folds, 5L)
  expect_equal(res$Y_method, 'glm')
  expect_equal(res$pi_method, 'SuperLearner')
  expect_equal(res$pi_SL_library, c('SL.glm', 'SL.gam'))
  expect_length(res$fit_Y, 5)
  expect_length(res$fit_pi, 5)
  expect_length(res$fit_outer, 5)
})


test_that("mia_aipw point estimate matches saturated cell mean", {
  set.seed(1234)

  res <- mia_aipw(
    data = dat.sim,
    X_names = c("X1", "X2"),
    X_values_1 = c(0, 1),
    X_values_2 = c(0, 0),
    contrast_type = "none",
    Y_model = Y ~ W * X1 * X2,
    pi_model = R_Y ~ W * X1 * X2,
    outer_model = psi_hat ~ X1 * X2,
    n_folds = 1
  )

  expect_equal(res$mean_est_1, 1.976309306, tolerance = 1e-5)
  expect_equal(res$mean_est_2, -0.03225690983, tolerance = 1e-5)
})
