#' Print method for objects of class "mia"
#'
#' Print method for objects of class "mia"
#'
#' @param x Object of class "mia".
#' @param digits Integer specifying the number of decimal places to display.
#' @param ... Other arguments (ignored).
#' @return No value is returned.
#' @seealso \code{\link{mia_nice}}, \code{\link{mia_ice}},
#' \code{\link{mia_aipw}}
#'
#' @examples
#' res <- mia_nice(data = dat.sim,
#'                 X_names = c("X1", "X2"),
#'                 X_values_1 = c(0, 1), X_values_2 = c(0, 0),
#'                 Y_model = Y ~ W + X1 + X2, W_model = W ~ X1 + X2)
#' print(res)
#'
#' @export

print.mia <- function(x, digits = 4, ...){
  if (!inherits(x, "mia")){
    stop("Argument 'x' must be an object of class \"mia\".")
  }

  cat('MIA METHOD FOR CONDITIONAL MEAN ESTIMATION\n')
  cat("==========================================\n\n")
  cat("Setting:\n")
  method_label <- if (!is.null(x$method) && x$method == 'ice'){
    'Iterative conditional expectation (ICE)'
  } else if (!is.null(x$method) && x$method == 'aipw'){
    'Augmented inverse probability weighting (AIPW)'
  } else {
    'Noniterative conditional expectation (NICE)'
  }
  cat(sprintf("  %-28s %s\n", "Method:", method_label))
  cat(sprintf("  %-28s %s\n", "Outcome variable type:", x$Y_type))
  if (is.null(x$method) || x$method == 'nice'){
    W_info_components <- paste0(x$W_type, " (", x$W_names, ")")
    W_info <- paste(W_info_components, collapse = ", ")
    cat(sprintf("  %-28s %s\n", "Auxiliary variable(s) type:", W_info))
  }


  cat("\nResults:\n")
  if (length(x$X_names) == 0){
    X_temp <- "(marginal)"
  } else {
    X_temp <- paste(paste0(x$X_names, "=", x$X_values_1), collapse = ", ")
  }
  cat(sprintf("  %-28s %s\n", "Predictor values:", X_temp))
  cat(sprintf("  %-28s %s\n", "Mean estimate:",
              formatC(x$mean_est_1, digits = digits, format = "f")))

  if (!is.null(x$X_values_2)){
    if (length(x$X_names) == 0){
      X_temp_2 <- "(marginal)"
    } else {
      X_temp_2 <- paste(paste0(x$X_names, "=", x$X_values_2), collapse = ", ")
    }
    cat(sprintf("\n  %-28s %s\n", "Predictor values:", X_temp_2))
    cat(sprintf("  %-28s %s\n", "Mean estimate:",
                formatC(x$mean_est_2, digits = digits, format = "f")))

    if (!is.na(x$contrast_est)){
      cat(sprintf("\n  %-28s %s\n", paste0('Mean ', x$contrast_type, ' estimate:'),
                  formatC(x$contrast_est, digits = digits, format = "f")))
    }
  }
}


#' Print method for objects of class "mia_ci"
#'
#' Print method for objects of class "mia_ci"
#'
#' @param x Object of class "mia_ci".
#' @param digits Integer specifying the number of decimal places to display.
#' @param ... Other arguments (ignored).
#' @return No value is returned.
#' @seealso \code{\link{get_CI}}
#'
#' @examples
#' set.seed(1234)
#' res <- mia_nice(data = dat.sim,
#'            X_names = c("X1", "X2"),
#'            X_values_1 = c(0, 1), X_values_2 = c(0, 0),
#'            Y_model = Y ~ W + X1 + X2, W_model = W ~ X1 + X2)
#' res_ci <- get_CI(res, n_boot = 100, type = 'perc')
#' print(res_ci)
#'
#' @export

print.mia_ci <- function(x, digits = 4, ...){
  if (!inherits(x, "mia_ci")){
    stop("Argument 'x' must be an object of class \"mia_ci\".")
  }

  cat('BOOTSTRAP CONFIDENCE INTERVALS FOR MIA METHOD\n')
  cat("=============================================\n\n")
  cat("Setting:\n")
  cat(sprintf("  %-24s %s\n", "Confidence level:", x$conf))
  cat(sprintf("  %-24s %s\n", "Interval type:", x$type))
  cat(sprintf("  %-24s %s\n", "Number of replicates:", x$n_boot))

  # ci_res_name <- x$type
  # if (ci_res_name == 'perc'){
  #   ci_res_name <- 'percent'
  # } else if (ci_res_name == 'norm'){
  #   ci_res_name <- 'normal'
  # }

  cat("\nResults:\n")
  X_temp <- paste(paste0(x$mia_res$X_names, "=", x$mia_res$X_values_1), collapse = ", ")
  cat(sprintf("  %-24s %s\n", "Predictor values:", X_temp))
  # cat(sprintf("  %-24s %s\n", "Mean estimate:",
  #             formatC(x$mia_res$mean_est_1, digits = digits, format = "f")))
  ci_vals_1 <- x$ci_1[[4]][4:5]
  ci_string_1 <- paste0("(", formatC(ci_vals_1[1], digits = digits, format = "f"),
                        ', ', formatC(ci_vals_1[2], digits = digits, format = "f"),
                        ')')
  cat(sprintf("  %-24s %s\n", "CI for mean: ", ci_string_1))

  if (!is.null(x$mia_res$X_values_2)){
    X_temp_2 <- paste(paste0(x$mia_res$X_names, "=", x$mia_res$X_values_2), collapse = ", ")
    cat(sprintf("\n  %-24s %s\n", "Predictor values:", X_temp_2))
    # cat(sprintf("  %-24s %s\n", "Mean estimate:",
    #             formatC(x$mia_res$mean_est_2, digits = digits, format = "f")))
    ci_vals_2 <- x$ci_2[[4]][4:5]
    ci_string_2 <- paste0("(", formatC(ci_vals_2[1], digits = digits, format = "f"),
                          ', ', formatC(ci_vals_2[2], digits = digits, format = "f"),
                          ')')
    cat(sprintf("  %-24s %s\n", "CI for mean: ", ci_string_2))

    if (!is.na(x$mia_res$contrast_est)){
      # cat(sprintf("\n  %-24s %s\n", paste0('Mean ', x$mia_res$contrast_type, ' estimate:'),
      #             formatC(x$mia_res$contrast_est, digits = digits, format = "f")))
      ci_vals_contrast <- x$ci_contrast[[4]][4:5]
      ci_string_contrast <- paste0("(", formatC(ci_vals_contrast[1], digits = digits, format = "f"),
                            ', ', formatC(ci_vals_contrast[2], digits = digits, format = "f"),
                            ')')
      cat(sprintf("\n  %-24s %s\n", paste0("CI for ", x$mia_res$contrast_type, ':'), ci_string_contrast))
    }
  }
}
