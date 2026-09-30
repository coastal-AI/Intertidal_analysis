## granadeiro_fit.R -- the R side of pyintertidal.granadeiro2021
##
## Granadeiro, Belo, Henriques, Catalao & Catry (2021) "Using Sentinel-2
## Images to Estimate Topography, Tidal-Stage Lags and Exposure Periods over
## Large Intertidal Areas", Remote Sensing 13(2):320.  All statistical steps
## that the paper names an R package for run here, with that package:
##
##   fit       the 4-parameter logistic of Eq. 4 (p.6) exactly as
##             nplr::nplr(x, y, useLog = FALSE, npars = 4) fits it, "lean":
##             NA drop and sort as nplr, inits <- nplr:::.initPars(x, y, 4),
##             best <- nlm(f = nplr:::.sce, p = inits, x = x, yobs = y,
##                         nplr:::.wsqRes, 0.25, nplr:::.nPL4),
##             fail if best$iterations == 0 or the fit is constant
##             (length(unique(signif(yFit, 5))) == 1), inflection
##             x = xmid + (1/scal) * log10(s) (nplr:::.inflPoint; s = 1).
##   lag       the same lean fit on the rising and on the ebbing scenes at
##             every lag of the grid (p.7, Figs. 4-5).
##   nplr      REFERENCE path for the equality proof: try(nplr(...)) and
##             getInflexion(fit)$x (optionally y <- convertToProp(y) first).
##   nplr_lag  REFERENCE path of `lag`.
##   lmodel2   lmodel2::lmodel2(y ~ x) regression.results (OLS, MA, SMA) for
##             the inter-calibration proof (p.4: major axis, lmodel2).
##   gam       mgcv::gam(lag ~ s(lon, lat)) with the defaults (bs = "tp",
##             k = 30 for two covariates, method = "GCV.Cp") and
##             predict() on new points (p.7-8).
##   eq1       the co-author's Eq. 1 line (Map_InterSedim_Bijagos,
##             DEM_based_intertidalmask_creation.R) for the Eq. 1 test.
##   versions  R and package versions.
##
## Usage:  Rscript granadeiro_fit.R <spec.txt>
## The spec is key=value lines.  Every array is raw float64 little-endian
## (numpy tofile / writeBin), numpy C order: a (rows, cols) numpy array is
## read here as a cols x rows column-major matrix.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript granadeiro_fit.R <spec.txt>")

read_spec <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(lines)]
  pos <- regexpr("=", lines, fixed = TRUE)
  setNames(as.list(substring(lines, pos + 1)), substring(lines, 1, pos - 1))
}
spec <- read_spec(args[1])
num <- function(key) {
  if (is.null(spec[[key]])) stop(sprintf("spec key '%s' missing", key))
  as.numeric(spec[[key]])
}

rd <- function(con, n) {
  v <- readBin(con, what = "double", n = n, size = 8, endian = "little")
  if (length(v) != n) stop(sprintf("short read: %d of %d doubles", length(v), n))
  v
}
write_f64 <- function(path, v) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.double(v), con, size = 8, endian = "little")
}

## ---------------------------------------------------------------------------
## the lean nplr path (fit / lag) and the reference nplr() path
## ---------------------------------------------------------------------------
## output of one fit, 11 doubles:
##   1 status (0 ok; 1 nlm iterations == 0; 2 constant fitted values;
##             3 any other error, e.g. in .initPars or nlm)
##   2 inflection x (NaN unless status 0)
##   3..7 bottom, top, xmid, scal, s (nlm estimate; NaN on error)
##   8 nlm iterations  9 nlm code  10 points used (after the NA drop)
##   11 nlm minimum (the weighted objective; NaN for the reference path)
N_OUT <- 11

lean_nplr4 <- function(x, y) {
  res <- rep(NA_real_, N_OUT)
  res[1] <- 3
  ## nplr(): NAs <- union(which(is.na(x)), which(is.na(y))); x <- x[-NAs] ...
  bad <- is.na(x) | is.na(y)
  if (any(bad)) {
    x <- x[!bad]
    y <- y[!bad]
  }
  ## nplr(): y <- y[order(x)]; x <- sort(x)
  y <- y[order(x)]
  x <- sort(x)
  res[10] <- length(x)
  best <- tryCatch(suppressWarnings({
    inits <- .initPars(x, y, 4)
    nlm(f = .sce, p = inits, x = x, yobs = y, .wsqRes, 0.25, .nPL4)
  }), error = function(e) NULL)
  if (is.null(best)) return(res)
  res[8] <- best$iterations
  res[9] <- best$code
  res[11] <- best$minimum
  res[3:7] <- best$estimate[1:5]
  if (best$iterations == 0) {
    res[1] <- 1
    return(res)
  }
  e <- best$estimate
  yFit <- .nPL4(e[1], e[2], e[3], e[4], e[5], x)
  if (length(unique(signif(yFit, 5))) == 1) {
    res[1] <- 2
    return(res)
  }
  res[1] <- 0
  ## nplr:::.inflPoint: x = xmid + (1/scal) * log10(s)
  res[2] <- e[3] + (1 / e[4]) * log10(e[5])
  res
}

ref_nplr4 <- function(x, y, prop) {
  res <- rep(NA_real_, N_OUT)
  res[1] <- 3
  fit <- tryCatch(suppressWarnings(suppressMessages({
    if (prop) y <- nplr::convertToProp(y)
    nplr::nplr(x = x, y = y, useLog = FALSE, npars = 4, silent = TRUE)
  })), error = function(e) e)
  if (inherits(fit, "error")) {
    msg <- conditionMessage(fit)
    if (grepl("'nlm' failed to estimate parameters", msg, fixed = TRUE)) res[1] <- 1
    else if (grepl("nplr failed and returned constant fitted values", msg, fixed = TRUE)) res[1] <- 2
    return(res)
  }
  p <- nplr::getPar(fit)$params
  res[1] <- 0
  res[2] <- nplr::getInflexion(fit)$x
  res[3:7] <- c(p$bottom, p$top, p$xmid, p$scal, p$s)
  res[10] <- length(nplr::getX(fit))
  res
}

mode <- spec$mode

if (mode %in% c("fit", "nplr", "lag", "nplr_lag")) {
  suppressPackageStartupMessages(library(nplr))
  .initPars <- nplr:::.initPars
  .sce <- nplr:::.sce
  .wsqRes <- nplr:::.wsqRes
  .nPL4 <- nplr:::.nPL4
}

if (mode %in% c("fit", "nplr")) {
  n_px <- num("n_px"); n_obs <- num("n_obs")
  x_shared <- num("x_shared") == 1
  prop <- if (is.null(spec$prop)) FALSE else num("prop") == 1
  con <- file(spec$in_file, "rb")
  if (x_shared) X <- rd(con, n_obs) else X <- matrix(rd(con, n_px * n_obs), nrow = n_obs)
  Y <- matrix(rd(con, n_px * n_obs), nrow = n_obs)
  close(con)
  out <- matrix(NA_real_, nrow = N_OUT, ncol = n_px)
  for (j in seq_len(n_px)) {
    x <- if (x_shared) X else X[, j]
    out[, j] <- if (mode == "fit") lean_nplr4(x, Y[, j]) else ref_nplr4(x, Y[, j], prop)
  }
  write_f64(spec$out_file, out)

} else if (mode %in% c("lag", "nplr_lag")) {
  n_px <- num("n_px"); n_obs <- num("n_obs"); n_lag <- num("n_lag")
  con <- file(spec$in_file, "rb")
  H <- matrix(rd(con, n_lag * n_obs), nrow = n_obs)    # column l = heights at lag l
  rising <- rd(con, n_obs) == 1
  Y <- matrix(rd(con, n_px * n_obs), nrow = n_obs)
  close(con)
  ## out[value, limb, lag, pixel]; value 1 status, 2 inflection; limb 1 rising, 2 ebbing
  out <- array(NA_real_, dim = c(2, 2, n_lag, n_px))
  for (j in seq_len(n_px)) {
    yr <- Y[rising, j]
    ye <- Y[!rising, j]
    for (l in seq_len(n_lag)) {
      if (mode == "lag") {
        r <- lean_nplr4(H[rising, l], yr)
        e <- lean_nplr4(H[!rising, l], ye)
      } else {
        r <- ref_nplr4(H[rising, l], yr, FALSE)
        e <- ref_nplr4(H[!rising, l], ye, FALSE)
      }
      out[, 1, l, j] <- r[1:2]
      out[, 2, l, j] <- e[1:2]
    }
  }
  write_f64(spec$out_file, out)

} else if (mode == "lmodel2") {
  suppressPackageStartupMessages(library(lmodel2))
  n_obs <- num("n_obs"); n_jobs <- num("n_jobs")
  con <- file(spec$in_x, "rb"); x <- rd(con, n_obs); close(con)
  con <- file(spec$in_y, "rb")
  out <- matrix(NA_real_, nrow = 8, ncol = n_jobs)
  for (k in seq_len(n_jobs)) {
    y <- rd(con, n_obs)
    m <- tryCatch(suppressMessages(suppressWarnings(
      lmodel2(y ~ x, data = data.frame(y = y, x = x)))), error = function(e) NULL)
    if (is.null(m)) next
    rr <- m$regression.results
    get <- function(met) as.numeric(rr[rr$Method == met, c("Intercept", "Slope")])
    out[, k] <- c(m$n, get("OLS"), get("MA"), get("SMA"), m$r)
  }
  close(con)
  write_f64(spec$out_file, out)

} else if (mode == "gam") {
  suppressPackageStartupMessages(library(mgcv))
  n <- num("n"); m <- num("m"); seed <- num("seed")
  con <- file(spec$in_sample, "rb"); S <- matrix(rd(con, n * 3), nrow = 3); close(con)
  d <- data.frame(lon = S[1, ], lat = S[2, ], lag = S[3, ])
  set.seed(seed)
  fit <- gam(lag ~ s(lon, lat), data = d)
  con <- file(spec$in_pred, "rb"); P <- matrix(rd(con, m * 2), nrow = 2); close(con)
  pred <- as.numeric(predict(fit, newdata = data.frame(lon = P[1, ], lat = P[2, ])))
  write_f64(spec$out_file, pred)
  write_f64(spec$out_fitted, as.numeric(fitted(fit)))
  s <- summary(fit)
  sm <- fit$smooth[[1]]
  f17 <- function(v) format(v, digits = 17)
  info <- c(
    paste0("r_version=", R.version.string),
    paste0("mgcv_version=", as.character(packageVersion("mgcv"))),
    paste0("formula=", deparse(formula(fit))),
    paste0("family=", fit$family$family, "/", fit$family$link),
    paste0("method=", fit$method),
    paste0("optimizer=", paste(fit$optimizer, collapse = "/")),
    paste0("basis=", class(sm)[1]),
    paste0("k=", sm$bs.dim),
    paste0("n=", s$n),
    paste0("n_unique_xy=", nrow(unique(d[, c("lon", "lat")]))),
    paste0("knots_subsampled=", as.integer(s$n > 2000)),
    paste0("seed_set=", seed),
    paste0("tprs_xt_seed=", if (is.null(sm$xt$seed)) "default(1)" else sm$xt$seed),
    paste0("edf_smooth=", f17(s$s.table[1, "edf"])),
    paste0("ref_df_smooth=", f17(s$s.table[1, "Ref.df"])),
    paste0("F_smooth=", f17(s$s.table[1, "F"])),
    paste0("p_smooth=", f17(s$s.table[1, "p-value"])),
    paste0("edf_total=", f17(sum(fit$edf))),
    paste0("gcv_score=", f17(fit$gcv.ubre)),
    paste0("scale_sig2=", f17(fit$sig2)),
    paste0("sp=", f17(fit$sp[1])),
    paste0("r_sq_adj=", f17(s$r.sq)),
    paste0("dev_expl=", f17(s$dev.expl)),
    paste0("intercept=", f17(coef(fit)[1])),
    paste0("converged=", fit$converged),
    paste0("mgcv_conv_fully_converged=", if (is.null(fit$mgcv.conv$fully.converged)) NA else fit$mgcv.conv$fully.converged),
    paste0("pred_min=", f17(min(pred))),
    paste0("pred_max=", f17(max(pred)))
  )
  writeLines(info, spec$out_summary)

} else if (mode == "eq1") {
  n <- num("n")
  con <- file(spec$in_file, "rb"); M <- matrix(rd(con, n * 5), nrow = 5); close(con)
  Tsat <- M[1, ]; TLW <- M[2, ]; THW <- M[3, ]; hLW <- M[4, ]; hHW <- M[5, ]
  ## verbatim from the co-author's DEM_based_intertidalmask_creation.R
  hsat <- hHW-((hHW-hLW)*(cos((pi*(Tsat-TLW)/(THW-TLW)))+1))/2
  write_f64(spec$out_file, hsat)

} else if (mode == "versions") {
  v <- c(paste0("r_version=", R.version.string))
  for (p in c("nplr", "lmodel2", "mgcv")) {
    v <- c(v, paste0(p, "=", tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING")))
  }
  writeLines(v, spec$out_file)

} else {
  stop(sprintf("unknown mode '%s'", mode))
}
