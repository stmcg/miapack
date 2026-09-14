#' MIA Method via the Augmented Inverse Probability Weighted Approach
#'
#' This function implements the marginalization over incomplete auxiliaries
#' (MIA) method (Mathur et al. 2026) via an augmented inverse probability
#' weighted (AIPW) approach. For an outcome variable \eqn{Y}, predictor
#' variable \eqn{X}, and auxiliary variable \eqn{W}, this function estimates
#' the conditional outcome mean identified by
#' \deqn{
#' \mu_{\text{MIA}}(x) =
#' E [ \, E [ Y | X, W, R_X = R_W = R_Y = 1 ]
#' \mid X=x, R_W = R_X = 1 \, ].
#' }
#' where \eqn{R_W}, \eqn{R_X}, and \eqn{R_Y} are indicators of non-missing values of \eqn{W}, \eqn{X}, and \eqn{Y}, respectively.
#' The AIPW approach creates a pseudo-outcome using estimates
#' of the conditional outcome mean and the probability that \eqn{Y} is
#' observed, and then regresses this pseudo-outcome on \eqn{X}
#' (see the \strong{Estimation algorithm} section for more details).
#' The function supports estimating \eqn{\mu_{\text{MIA}}(x_1)} and
#' \eqn{\mu_{\text{MIA}}(x_2)}, as well as differences and ratios between these
#' quantities. \emph{This function currently implements point estimation only.}
#'
#' @param data Data frame containing the observed data.
#' @param X_names Vector of character strings specifying the name(s) of the
#' predictor variable(s) \eqn{X}.
#' @param X_values_1 Numeric vector specifying the value of the predictor
#' variable(s) \eqn{X}, i.e. \eqn{x_1} in
#' \eqn{\mu_{\text{MIA}}(x_1)}.
#' @param X_values_2 (Optional) Numeric vector specifying an additional value
#' of the predictor variable(s) \eqn{X}, i.e. \eqn{x_2} in
#' \eqn{\mu_{\text{MIA}}(x_2)}.
#' @param contrast_type (Optional) Character string specifying the type of
#' contrast to use when comparing \eqn{\mu_{\text{MIA}}(x_1)} and
#' \eqn{\mu_{\text{MIA}}(x_2)}. Options are \code{"difference"},
#' \code{"ratio"}, and \code{"none"}.
#' @param Y_model Formula for the outcome model \eqn{Q(X,W)}. As in
#' \code{\link{mia_ice}}, the left-hand side must be \code{Y}. The auxiliary
#' variables \eqn{W} are taken to be the variables on the right-hand side that
#' are not listed in \code{X_names}.
#' @param pi_model Formula for the outcome-observation model
#' \eqn{\pi(X,W)}. The left-hand side must be \code{R_Y}, which is generated
#' internally and indicates whether \code{Y} is observed. The right-hand side
#' can only depend on variables in \code{X_names} and the auxiliary variables
#' identified from \code{Y_model}.
#' @param outer_model Formula for the outer regression of the estimated
#' pseudo-outcomes on the predictor(s) \eqn{X}. The left-hand side must be
#' \code{psi_hat}, which denotes the estimated pseudo-outcomes.
#' For example, \code{psi_hat ~ X1 * X2} specifies a model that is saturated
#' with respect to binary predictors \code{X1} and \code{X2}.
#' @param Y_type (Optional) Character string specifying the "type" of the
#' outcome variable. Options are \code{"binary"} and \code{"continuous"}. If
#' this is not supplied, the type will be inferred from the corresponding
#' column in \code{data}.
#' @param Y_method Character string specifying how to estimate
#' \eqn{Q(X,W)}. Options are \code{"glm"} and
#' \code{"SuperLearner"}. The default is \code{"glm"}.
#' @param pi_method Character string specifying how to estimate
#' \eqn{\pi(X,W)}. Options are \code{"glm"} and
#' \code{"SuperLearner"}. The default is \code{"glm"}.
#' @param Y_SL_library Library passed to the \code{SL.library} argument of
#' \code{\link[SuperLearner]{SuperLearner}} when
#' \code{Y_method = "SuperLearner"}. The default is
#' \code{c("SL.glm", "SL.gam")}.
#' @param pi_SL_library Library passed to the \code{SL.library} argument of
#' \code{\link[SuperLearner]{SuperLearner}} when
#' \code{pi_method = "SuperLearner"}. The default is
#' \code{c("SL.glm", "SL.gam")}.
#' @param n_folds (Optional) Integer specifying the number of folds used for
#' cross fitting. The default is \code{1} (i.e., no cross fitting) if both
#' \code{Y_method} and \code{pi_method} are \code{"glm"}, and
#' \code{5} if either nuisance function is estimated using
#' \code{"SuperLearner"}.
#'
#' @return An object of class "mia". This object is a list with the following
#' elements:
#' \item{mean_est_1}{conditional outcome mean estimate under
#' \code{X_values_1}}
#' \item{mean_est_2}{conditional outcome mean estimate under
#' \code{X_values_2}}
#' \item{contrast_est}{contrast of conditional outcome mean estimates between
#' \code{X_values_1} and \code{X_values_2}}
#' \item{fit_Y}{list of fitted models for \eqn{Q(X,W)}, with one model per
#' fold or one model when cross fitting is not performed}
#' \item{fit_pi}{list of fitted models for \eqn{\pi(X,W)}, with one model per
#' fold or one model when cross fitting is not performed}
#' \item{fit_outer}{list of fitted outer regressions of the estimated
#' pseudo-outcomes on \eqn{X}, with one model per fold or one
#' model when cross fitting is not performed.}
#' \item{Q_hat}{estimated values of \eqn{Q(X,W)}, aligned with the rows
#' of \code{data}; \code{NA} for rows with incomplete \eqn{X} or \eqn{W}}
#' \item{pi_hat}{estimated values of \eqn{\pi(X,W)}, aligned with the
#' rows of \code{data}; \code{NA} for rows with incomplete \eqn{X} or
#' \eqn{W}}
#' \item{psi_hat}{estimated pseudo-outcomes, aligned with the rows
#' of \code{data}; \code{NA} for rows with incomplete \eqn{X} or \eqn{W}}
#' \item{fold_id}{fold assignments in cross fitting, aligned with the rows of \code{data};
#' all equal to 1 when cross fitting is not performed, and \code{NA} for rows with
#' incomplete \eqn{X} or \eqn{W}}
#' \item{...}{additional elements}
#'
#' @seealso \code{\link{mia_ice}}, \code{\link{mia_nice}},
#' \code{\link{print.mia}}
#'
#' @details
#'
#' \strong{Nuisance function estimation:}
#'
#' Define
#' \deqn{
#' Q(X,W) = E[Y \mid X,W,R_X=R_W=R_Y=1]
#' }
#' and
#' \deqn{
#' \pi(X,W) = P(R_Y=1 \mid X,W,R_X=R_W=1).
#' }
#' The model formulas for these nuisance function models are specified by
#' \code{Y_model} and \code{pi_model}. When \code{Y_method = "glm"},
#' \eqn{Q(X,W)} is estimated using linear regression for a continuous outcome
#' and logistic regression for a binary outcome. When
#' \code{pi_method = "glm"}, \eqn{\pi(X,W)} is estimated using logistic
#' regression. Either nuisance function can instead be estimated using SuperLearner
#' by setting the corresponding method to \code{"SuperLearner"}; the
#' candidate learners are specified through \code{Y_SL_library} and
#' \code{pi_SL_library}. See the \link[SuperLearner]{SuperLearner} package for more details on
#' available candidate learners.
#'
#' The nuisance functions can be estimated with or without cross fitting. If
#' cross fitting is not performed (i.e., \code{n_folds = 1}), the same observations are used for
#' estimating the nuisance functions and for estimating the MIA functional.
#' If cross fitting is performed (i.e., \code{n_folds > 1}),
#' the sample is randomly partitioned into folds where the nuisance functions are
#' estimated in a separate subsample from that used to estimate the MIA functional.
#' By default, cross fitting is not performed when both nuisance
#' functions are estimated using GLMs and cross fitting with 5 folds is performed
#' when SuperLearner is used to estimate the nuisance functions.
#'
#' \strong{Estimation algorithm:}
#'
#' \emph{Step 1:} Estimate the nuisance functions \eqn{Q(X,W)} and
#' \eqn{\pi(X,W)} using the approach specified by \code{Y_method},
#' \code{pi_method}, and \code{n_folds}.
#'
#' \emph{Step 2:} Use the fitted nuisance functions to construct the estimated
#' pseudo-outcome
#' \deqn{
#' \hat{\psi} =
#' \hat{Q}(X,W) +
#' \frac{R_Y}{\hat{\pi}(X,W)}
#' \{Y-\hat{Q}(X,W)\}.
#' }
#' When cross-fitting is used, the pseudo-outcomes for each fold are
#' constructed using nuisance functions estimated without that fold.
#'
#' \emph{Step 3:} Regress the estimated pseudo-outcomes on \eqn{X} using the
#' model specified by \code{outer_model}, and evaluate the fitted
#' model at the target predictor value(s).
#'
#' \emph{Step 4:} When cross fitting is used, repeat Steps 1--3 for each fold and
#' average the fold-specific estimates at each target predictor value. Otherwise,
#' return the corresponding full-sample estimate.
#'
#' @references
#' Mathur MB, Seaman S, Zhang W, McGrath S, Shpitser I. (2026). \emph{Estimating conditional means under missingness-not-at-random with incomplete auxiliary variables}. \doi{10.13140/RG.2.2.30750.19524}.
#'
#' @examples
#' # Generalized linear models for both nuisance functions
#' mia_aipw(data = dat.sim,
#'          X_names = c("X1", "X2"),
#'          X_values_1 = c(0, 1), X_values_2 = c(0, 0),
#'          Y_model = Y ~ W + X1 + X2,
#'          pi_model = R_Y ~ W + X1 + X2,
#'          outer_model = psi_hat ~ X1 * X2)
#'
#' \dontrun{
#' # Flexible nuisance function estimation using SuperLearner
#' mia_aipw(data = dat.sim,
#'          X_names = c("X1", "X2"),
#'          X_values_1 = c(0, 1), X_values_2 = c(0, 0),
#'          Y_model = Y ~ W + X1 + X2,
#'          pi_model = R_Y ~ W + X1 + X2,
#'          outer_model = psi_hat ~ X1 * X2,
#'          Y_method = "SuperLearner",
#'          pi_method = "SuperLearner",
#'          Y_SL_library = c("SL.glm", "SL.gam"),
#'          pi_SL_library = c("SL.glm", "SL.gam"))
#' }
#'
#' @export
mia_aipw <- function(data, X_names, X_values_1, X_values_2 = NULL,
                     contrast_type,
                     Y_model, Y_type,
                     pi_model,
                     outer_model,
                     Y_method = 'glm',
                     pi_method = 'glm',
                     Y_SL_library = c('SL.glm', 'SL.gam'),
                     pi_SL_library = c('SL.glm', 'SL.gam'),
                     n_folds = NULL) {
  # Checking that data has the correct column names
  missing_cols <- setdiff(X_names, colnames(data))
  if (length(missing_cols) > 0) {
    stop(paste("The following columns are listed in X_names but missing from the data:",
               paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  if (!'Y' %in% colnames(data)){
    stop("The observed data must include a column called 'Y' indicating the outcome variable.", call. = FALSE)
  }
  reserved_cols <- intersect(
    c("R_Y", "psi_hat"),
    colnames(data)
  )
  if (length(reserved_cols) > 0) {
    stop(
      paste0(
        "The following column name(s) are reserved for internal use by ",
        "mia_aipw(): ",
        paste(reserved_cols, collapse = ", "),
        ". Please rename these variable(s) in the input data."
      ),
      call. = FALSE
    )
  }

  # Checking the model formula for Y
  if (!inherits(Y_model, "formula")) {
    stop("Y_model must be a formula, e.g., Y ~ W + X", call. = FALSE)
  }
  lhs_Y <- all.vars(Y_model[[2]])
  if (length(lhs_Y) != 1 || lhs_Y != "Y") {
    stop("The left-hand side of Y_model must be the variable 'Y'.", call. = FALSE)
  }

  # Identifying the auxiliary variable(s) in Y_model, as in mia_ice()
  W_names <- setdiff(all.vars(Y_model[[3]]), X_names)
  missing_W_cols <- setdiff(W_names, colnames(data))
  if (length(missing_W_cols) > 0) {
    stop(paste("The following auxiliary variables identified from Y_model are missing from the data:",
               paste(missing_W_cols, collapse = ", ")), call. = FALSE)
  }

  # Checking the model formula for pi
  if (missing(pi_model)){
    stop("pi_model must be supplied, e.g., R_Y ~ W + X", call. = FALSE)
  }
  if (!inherits(pi_model, "formula")) {
    stop("pi_model must be a formula, e.g., R_Y ~ W + X", call. = FALSE)
  }
  lhs_pi <- all.vars(pi_model[[2]])
  if (length(lhs_pi) != 1 || lhs_pi != "R_Y") {
    stop("The left-hand side of pi_model must be 'R_Y', which denotes whether Y is observed.", call. = FALSE)
  }
  rhs_pi <- all.vars(pi_model[[3]])
  bad_pi <- setdiff(rhs_pi, c(X_names, W_names))
  if (length(bad_pi) > 0){
    stop(paste0("The right-hand side of pi_model can only depend on the predictor(s) in X_names and the auxiliary variable(s) identified from Y_model. The following term(s) are not allowed: ",
                paste(bad_pi, collapse = ", "), "."), call. = FALSE)
  }

  # Checking the model formula for the outer regression
  if (missing(outer_model)){
    stop("outer_model must be supplied, e.g., psi_hat ~ X1 * X2", call. = FALSE)
  }
  if (!inherits(outer_model, "formula")) {
    stop("outer_model must be a formula, e.g., psi_hat ~ X1 * X2", call. = FALSE)
  }
  lhs_outer <- all.vars(outer_model[[2]])
  if (length(lhs_outer) != 1 || lhs_outer != "psi_hat") {
    stop("The left-hand side of outer_model must be 'psi_hat', which denotes the estimated pseudo-outcomes.", call. = FALSE)
  }
  rhs_outer <- all.vars(outer_model[[3]])
  bad_outer <- setdiff(rhs_outer, X_names)
  if (length(bad_outer) > 0){
    stop(paste0("The right-hand side of outer_model can only depend on the predictor(s) in X_names. The following term(s) are not in X_names: ",
                paste(bad_outer, collapse = ", "), "."), call. = FALSE)
  }

  # Checking X_values_1 and X_values_2
  if (length(X_names) != length(X_values_1)){
    stop("The arguments 'X_names' and 'X_values_1' must be of the same length.", call. = FALSE)
  }
  if (!is.null(X_values_2)){
    if (length(X_names) != length(X_values_2)){
      stop("The arguments 'X_names' and 'X_values_2' must be of the same length.", call. = FALSE)
    }
    if (missing(contrast_type)){
      contrast_type <- 'difference'
    }
  } else {
    contrast_type <- 'none'
  }
  if (!contrast_type %in% c('difference', 'ratio', 'none')){
    stop("contrast_type must be set to either 'difference', 'ratio', or 'none'.", call. = FALSE)
  }

  # Validating X_values for categorical predictors
  if (length(X_names) > 0){
    for (i in 1:length(X_names)){
      if (is.factor(data[[X_names[i]]])){
        valid_levels <- levels(data[[X_names[i]]])
        # Check X_values_1
        if (!X_values_1[i] %in% valid_levels){
          stop(paste0("The value ", X_values_1[i], " specified for predictor '", X_names[i],
                      "' in X_values_1 is not a valid level. Valid levels are: ",
                      paste(valid_levels, collapse = ", "), "."), call. = FALSE)
        }
        # Check X_values_2 if provided
        if (!is.null(X_values_2)){
          if (!X_values_2[i] %in% valid_levels){
            stop(paste0("The value ", X_values_2[i], " specified for predictor '", X_names[i],
                        "' in X_values_2 is not a valid level. Valid levels are: ",
                        paste(valid_levels, collapse = ", "), "."), call. = FALSE)
          }
        }
      }
    }
  }

  # Checking variable types are appropriately set
  if (!missing(Y_type)){
    if (!Y_type %in% c('binary', 'continuous')){
      stop("Y_type must be set to either 'binary' or 'continuous'.", call. = FALSE)
    }
  }
  if (length(Y_method) != 1 || !Y_method %in% c('glm', 'SuperLearner')){
    stop("Y_method must be set to either 'glm' or 'SuperLearner'.", call. = FALSE)
  }
  if (length(pi_method) != 1 || !pi_method %in% c('glm', 'SuperLearner')){
    stop("pi_method must be set to either 'glm' or 'SuperLearner'.", call. = FALSE)
  }
  if (Y_method == 'SuperLearner' && length(Y_SL_library) == 0){
    stop("Y_SL_library must contain at least one learner when Y_method = 'SuperLearner'.", call. = FALSE)
  }
  if (pi_method == 'SuperLearner' && length(pi_SL_library) == 0){
    stop("pi_SL_library must contain at least one learner when pi_method = 'SuperLearner'.", call. = FALSE)
  }

  # Setting and checking n_folds
  if (is.null(n_folds)){
    if (Y_method == 'glm' && pi_method == 'glm'){
      n_folds <- 1L
    } else {
      n_folds <- 5L
    }
  }
  if (length(n_folds) != 1 || !is.numeric(n_folds) ||
      is.na(n_folds) || n_folds < 1 || n_folds != as.integer(n_folds)){
    stop("n_folds must be an integer greater than or equal to 1.", call. = FALSE)
  }
  n_folds <- as.integer(n_folds)

  # Creating dataset for AIPW estimation
  if (length(X_names) == 0){
    R_X <- rep(TRUE, nrow(data))
  } else {
    R_X <- stats::complete.cases(data[, X_names, drop = FALSE])
  }
  if (length(W_names) == 0){
    R_W <- rep(TRUE, nrow(data))
  } else {
    R_W <- stats::complete.cases(data[, W_names, drop = FALSE])
  }
  R_Y <- !is.na(data$Y)

  outer_ind <- which(R_X == 1 & R_W == 1)
  data_outer <- data[outer_ind, , drop = FALSE]
  data_outer$R_Y <- as.integer(R_Y[outer_ind])

  # Checking for empty datasets after filtering
  if (nrow(data_outer) == 0){
    stop("No cases found with X and W observed for AIPW estimation. All observations are missing at least one of: X or W. Please check the missingness patterns in your data.", call. = FALSE)
  }
  if (sum(data_outer$R_Y) == 0){
    stop("No complete cases found for fitting the outcome model (Y). All observations with X and W observed are missing Y. Please check the missingness patterns in your data.", call. = FALSE)
  }
  if (n_folds > nrow(data_outer)){
    stop("n_folds cannot be larger than the number of observations with X and W observed.", call. = FALSE)
  }

  # Inferring Y type
  if (missing(Y_type)){
    Y_levels <- unique(stats::na.omit(data$Y))
    if (length(Y_levels) == 2){
      Y_type <- 'binary'
    } else if (is.numeric(data$Y)){
      Y_type <- 'continuous'
    } else {
      Y_type <- NA
    }
    # Check if type inference failed
    if (is.na(Y_type)){
      stop("Unable to infer the type for the outcome variable Y. Please explicitly specify Y_type. Valid options are 'binary' or 'continuous'.", call. = FALSE)
    }
  }

  # SuperLearner is an optional dependency
  if ((Y_method == 'SuperLearner' || pi_method == 'SuperLearner') &&
      !requireNamespace("SuperLearner", quietly = TRUE)){
    stop("The SuperLearner package must be installed when Y_method or pi_method is set to 'SuperLearner'.", call. = FALSE)
  }

  # Assigning folds among observations with X and W observed. When
  # n_folds = 1, every eligible observation is assigned to the single
  # full-sample fit.
  if (n_folds == 1){
    fold_id_outer <- rep(1L, nrow(data_outer))
  } else {
    fold_id_outer <- sample(rep(seq_len(n_folds), length.out = nrow(data_outer)))
  }

  fit_Y <- vector('list', n_folds)
  fit_pi <- vector('list', n_folds)
  fit_outer <- vector('list', n_folds)
  names(fit_Y) <- names(fit_pi) <- names(fit_outer) <-
    paste0('fold_', seq_len(n_folds))

  Q_hat_outer <- rep(NA_real_, nrow(data_outer))
  pi_hat_outer <- rep(NA_real_, nrow(data_outer))
  psi_hat_outer <- rep(NA_real_, nrow(data_outer))
  mean_est_1_fold <- rep(NA_real_, n_folds)
  names(mean_est_1_fold) <- paste0('fold_', seq_len(n_folds))
  mean_est_2_fold <- if (!is.null(X_values_2)) rep(NA_real_, n_folds) else NULL
  if (!is.null(mean_est_2_fold)){
    names(mean_est_2_fold) <- paste0('fold_', seq_len(n_folds))
  }

  all_Y_observed <- all(data_outer$R_Y == 1)

  for (j in seq_len(n_folds)){
    if (n_folds == 1){
      # No cross-fitting: fit and evaluate on the full eligible sample.
      ind_train <- seq_len(nrow(data_outer))
      ind_test <- seq_len(nrow(data_outer))
    } else {
      ind_test <- which(fold_id_outer == j)
      ind_train <- which(fold_id_outer != j)
    }

    data_train <- data_outer[ind_train, , drop = FALSE]
    data_test <- data_outer[ind_test, , drop = FALSE]
    data_train_Y <- data_train[data_train$R_Y == 1, , drop = FALSE]

    if (nrow(data_train_Y) == 0){
      stop(paste0("No observations with Y observed are available for fitting the outcome model in the training sample for fold ", j, ". Consider using fewer folds or a different random seed."), call. = FALSE)
    }

    # Fitting Q(X,W) and predicting in the evaluation sample
    if (Y_method == 'glm'){
      fit_Y[[j]] <- safe_fit(variable_name = 'Y', variable_type = Y_type,
                             formula = Y_model, data = data_train_Y)
      Q_hat_test <- get_Y_pred(df = data_test, fit_Y = fit_Y[[j]],
                               Y_type = Y_type)
    } else if (Y_method == 'SuperLearner'){
      Y_SL <- fit_aipw_SL(
        formula = Y_model,
        data_train = data_train_Y,
        data_test = data_test,
        outcome = get_aipw_SL_Y(data_train_Y$Y, Y_type = Y_type),
        family = if (Y_type == 'binary') stats::binomial() else stats::gaussian(),
        SL_library = Y_SL_library,
        variable_name = 'Y'
      )
      fit_Y[[j]] <- Y_SL$fit
      Q_hat_test <- Y_SL$pred
    }

    # Fitting pi(X,W) and predicting in the evaluation sample
    if (all_Y_observed){
      # When Y is observed for every eligible unit, pi(X,W) = 1.
      # Use single-bracket assignment so that the NULL is stored in the
      # existing list element rather than removing that element.
      fit_pi[j] <- list(NULL)
      pi_hat_test <- rep(1, nrow(data_test))
    } else {
      if (length(unique(data_train$R_Y)) < 2){
        stop(paste0("The outcome-observation indicator R_Y has only one level in the training sample for fold ", j, ". The model for pi(X,W) cannot be estimated. Consider using fewer folds or a different random seed."), call. = FALSE)
      }

      if (pi_method == 'glm'){
        fit_pi[[j]] <- safe_fit(variable_name = 'R_Y', variable_type = 'binary',
                                formula = pi_model, data = data_train)
        pi_hat_test <- stats::predict(fit_pi[[j]], type = 'response',
                                      newdata = data_test)
      } else if (pi_method == 'SuperLearner'){
        pi_SL <- fit_aipw_SL(
          formula = pi_model,
          data_train = data_train,
          data_test = data_test,
          outcome = data_train$R_Y,
          family = stats::binomial(),
          SL_library = pi_SL_library,
          variable_name = 'R_Y'
        )
        fit_pi[[j]] <- pi_SL$fit
        pi_hat_test <- pi_SL$pred
      }
    }

    Q_hat_test <- as.numeric(Q_hat_test)
    pi_hat_test <- as.numeric(pi_hat_test)

    if (length(Q_hat_test) != nrow(data_test) ||
        any(!is.finite(Q_hat_test))){
      stop(paste0("Non-finite or incorrectly sized predictions were obtained from the outcome model in fold ", j, "."), call. = FALSE)
    }
    if (length(pi_hat_test) != nrow(data_test) ||
        any(!is.finite(pi_hat_test))){
      stop(paste0("Non-finite or incorrectly sized predictions were obtained from the model for pi(X,W) in fold ", j, "."), call. = FALSE)
    }
    if (any(pi_hat_test <= 0) || any(pi_hat_test > 1)){
      stop(paste0("Predicted values of pi(X,W) must lie in (0, 1]. Invalid predictions were obtained in fold ", j, "."), call. = FALSE)
    }

    # Computing the pseudo-outcomes in the evaluation sample.
    # The augmentation term is set explicitly to zero when Y is missing because 0 * NA is NA.
    augmentation <- rep(0, nrow(data_test))
    ind_Y_obs_test <- which(data_test$R_Y == 1)
    if (length(ind_Y_obs_test) > 0){
      augmentation[ind_Y_obs_test] <-
        (data_test$Y[ind_Y_obs_test] - Q_hat_test[ind_Y_obs_test]) /
        pi_hat_test[ind_Y_obs_test]
    }
    psi_hat_test <- Q_hat_test + augmentation

    Q_hat_outer[ind_test] <- Q_hat_test
    pi_hat_outer[ind_test] <- pi_hat_test
    psi_hat_outer[ind_test] <- psi_hat_test

    # Outer regression in the evaluation sample
    data_test$psi_hat <- psi_hat_test
    fit_outer[[j]] <- tryCatch({
      stats::lm(outer_model, data = data_test)
    },
    error = function(e) {
      stop(paste0("Error in fitting the outer regression in fold ", j, ": ",
                  conditionMessage(e),
                  ". Consider using fewer folds or simplifying outer_model."),
           call. = FALSE)
    })

    # Fold-specific prediction at X_values_1
    mean_est_1_fold[j] <- get_aipw_mean(
      X_values = X_values_1, X_names = X_names,
      fit_outer = fit_outer[[j]]
    )

    # Fold-specific prediction at X_values_2
    if (!is.null(X_values_2)){
      mean_est_2_fold[j] <- get_aipw_mean(
        X_values = X_values_2, X_names = X_names,
        fit_outer = fit_outer[[j]]
      )
    }
  }

  if (any(!is.finite(mean_est_1_fold))){
    stop("At least one fold produced a non-finite estimate under X_values_1. Consider using fewer folds or simplifying outer_model.", call. = FALSE)
  }

  Y_mean <- mean(mean_est_1_fold)

  if (!is.null(X_values_2)){
    if (any(!is.finite(mean_est_2_fold))){
      stop("At least one fold produced a non-finite estimate under X_values_2. Consider using fewer folds or simplifying outer_model.", call. = FALSE)
    }
    Y_mean_2 <- mean(mean_est_2_fold)

    if (contrast_type == 'difference'){
      contrast_est <- Y_mean - Y_mean_2
    } else if (contrast_type == 'ratio'){
      contrast_est <- Y_mean / Y_mean_2
    } else if (contrast_type == 'none'){
      contrast_est <- NA
    }
  } else {
    Y_mean_2 <- contrast_est <- NA
  }

  # Storing nuisance predictions and pseudo-outcomes aligned with the rows of
  # the original data
  Q_hat <- pi_hat <- psi_hat <- rep(NA_real_, nrow(data))
  fold_id <- rep(NA_integer_, nrow(data))
  Q_hat[outer_ind] <- Q_hat_outer
  pi_hat[outer_ind] <- pi_hat_outer
  psi_hat[outer_ind] <- psi_hat_outer
  fold_id[outer_ind] <- fold_id_outer

  out <- list(
    mean_est_1 = Y_mean,
    mean_est_2 = Y_mean_2,
    contrast_est = contrast_est,
    fit_Y = fit_Y,
    fit_pi = fit_pi,
    fit_outer = fit_outer,
    Q_hat = Q_hat,
    pi_hat = pi_hat,
    psi_hat = psi_hat,
    fold_id = fold_id,
    mean_est_1_fold = mean_est_1_fold,
    mean_est_2_fold = mean_est_2_fold,
    Y_type = Y_type,
    X_names = X_names,
    W_names = W_names,
    X_values_1 = X_values_1, X_values_2 = X_values_2,
    contrast_type = contrast_type,
    Y_model = Y_model,
    pi_model = pi_model,
    outer_model = outer_model,
    Y_method = Y_method,
    pi_method = pi_method,
    Y_SL_library = Y_SL_library,
    pi_SL_library = pi_SL_library,
    n_folds = n_folds,
    method = 'aipw',
    data = data
  )
  class(out) <- 'mia'
  return(out)
}

get_aipw_mean <- function(X_values, X_names, fit_outer){
  # Prediction from the fold-specific outer regression at the target X value.
  if (length(X_names) == 0){
    # For the marginal-mean special case, outer_model must be intercept-only.
    mean_est <- as.numeric(stats::coef(fit_outer)[1])
  } else {
    # This follows the same construction used by get_ice_mean().
    df_pred <- data.frame(matrix(X_values, nrow = 1))
    colnames(df_pred) <- X_names
    mean_est <- as.numeric(stats::predict(fit_outer, newdata = df_pred))
  }

  return(mean_est)
}

get_aipw_SL_Y <- function(Y, Y_type){
  if (Y_type == 'continuous'){
    if (!is.numeric(Y)){
      stop("When Y_method = 'SuperLearner' and Y_type = 'continuous', Y must be numeric.", call. = FALSE)
    }
    return(as.numeric(Y))
  }

  if (is.factor(Y)){
    Y_drop <- droplevels(Y)
    if (nlevels(Y_drop) != 2){
      stop("When Y_method = 'SuperLearner' and Y_type = 'binary', a factor Y must have exactly two observed levels in each training sample.", call. = FALSE)
    }
    return(as.numeric(Y_drop == levels(Y_drop)[2]))
  }

  if (is.logical(Y)){
    return(as.numeric(Y))
  }

  if (is.numeric(Y)){
    Y_values <- sort(unique(Y))
    if (all(Y_values %in% c(0, 1))){
      return(as.numeric(Y))
    }
  }

  stop("When Y_method = 'SuperLearner' and Y_type = 'binary', Y must be coded as 0/1, logical, or a two-level factor.", call. = FALSE)
}

get_aipw_SL_X <- function(formula, data_train, data_test){
  terms_rhs <- stats::delete.response(stats::terms(formula))

  X_train <- stats::model.matrix(terms_rhs, data = data_train)
  X_test <- stats::model.matrix(terms_rhs, data = data_test)

  # SuperLearner does not require an intercept column; individual learners
  # handle intercepts themselves.
  if ('(Intercept)' %in% colnames(X_train)){
    X_train <- X_train[, colnames(X_train) != '(Intercept)', drop = FALSE]
  }
  if ('(Intercept)' %in% colnames(X_test)){
    X_test <- X_test[, colnames(X_test) != '(Intercept)', drop = FALSE]
  }

  # Accommodate intercept-only nuisance formulas.
  if (ncol(X_train) == 0){
    X_train <- matrix(1, nrow = nrow(data_train), ncol = 1,
                      dimnames = list(NULL, '.intercept'))
    X_test <- matrix(1, nrow = nrow(data_test), ncol = 1,
                     dimnames = list(NULL, '.intercept'))
  }

  if (!identical(colnames(X_train), colnames(X_test))){
    stop(paste0(
      "The model matrices for the SuperLearner training and evaluation samples have different columns. ",
      "Categorical predictors should be encoded as factors with fixed levels before calling mia_aipw."
    ), call. = FALSE)
  }

  return(list(
    X_train = as.data.frame(X_train),
    X_test = as.data.frame(X_test)
  ))
}

fit_aipw_SL <- function(formula, data_train, data_test, outcome, family,
                        SL_library, variable_name){
  SL_X <- get_aipw_SL_X(formula = formula,
                        data_train = data_train,
                        data_test = data_test)

  fit <- tryCatch({
    SuperLearner::SuperLearner(
      Y = outcome,
      X = SL_X$X_train,
      newX = SL_X$X_test,
      family = family,
      SL.library = SL_library,
      env = asNamespace('SuperLearner')
    )
  },
  error = function(e) {
    stop(paste0("Error in fitting the SuperLearner model for ",
                variable_name, ": ", conditionMessage(e)), call. = FALSE)
  })

  pred <- as.numeric(fit$SL.predict)

  return(list(fit = fit, pred = pred))
}
