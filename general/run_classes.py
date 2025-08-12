def invert_log(logged_df, columns_to_log):
  logged_df = logged_df.copy()
  
  for i in columns_to_log:
    logged_df[i] = logged_df[i].apply(np.exp)

  return logged_df


def invert_diff(original_df, differenced_df, columns_to_diff):
    original_df = original_df.copy()
    differenced_df = differenced_df.copy()
  
    for key in columns_to_diff:
      print("Column: " + key)
      first_invert = original_df[key].shift(columns_to_diff[key]) + differenced_df[key]
      differenced_df[key] = first_invert

    return differenced_df


class OneStepTimeSeriesValidationARDL:
    def __init__(self, df_y_log, df_x_future, to_log_y, fit, y, lags):
        self.df_y_log = df_y_log
        self.df_x_future = df_x_future
        self.to_log_y = to_log_y
        self.fit = fit
        self.y = y
        self.lags = lags

    def run_validation(self):
        """Runs the one-step time series validation and returns predictions, residuals, and RMSE."""
        #code to split into train/test sets
        os_y_train, os_y_test = train_test_split(self.df_y_log, test_size=0.3, shuffle=False)
        os_x_train = self.df_x_future.copy(deep=True).loc[os_y_train.index[0]:os_y_train.index[-1] + DateOffset(months=self.lags * 3)]
        os_x_test = self.df_x_future.copy(deep=True).loc[os_y_test.index[0]:os_y_test.index[-1] + DateOffset(months=self.lags * 3)]
        
        #code to forecast and compute predictions
        os_predict = invert_log(self.fit.forecast(steps=len(os_y_test), exog=os_x_test).to_frame(name=self.y).set_index(os_y_test.index), self.to_log_y)[self.y]
        os_fitted = invert_log(self.fit.forecast(steps=len(os_y_train), exog=os_x_train).to_frame(name=self.y).set_index(os_y_train.index), self.to_log_y)[self.y]

        #code to calculate residuals and RMSE
        os_residual = os_fitted - invert_log(os_y_train, self.to_log_y)[self.y]
        os_rmse = abs(os_predict - invert_log(os_y_test, self.to_log_y)[self.y])
        os_mean_rmse = rmse(os_predict, invert_log(os_y_test, self.to_log_y)[self.y])

        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse

class OneStepTimeSeriesValidationARIMA:
    def __init__(self, model, y, to_log, df_log, fit):
        self.model = model
        self.y = y
        self.to_log = to_log
        self.df_log = df_log
        self.fit = fit

    def run_validation(self):
        # Split data into train/test sets
        os_train, os_test = train_test_split(self.df_log, test_size=0.3, shuffle=False)
        osv = self.fit.apply(os_train)

        # Forecast for the test set
        os_predict = invert_log(osv.forecast(steps=len(os_test)).to_frame(name=self.y), self.to_log)[self.y]

        # Get fitted values for the training set
        os_fitted = invert_log(osv.fittedvalues.to_frame(name=self.y), self.to_log)[self.y]

        # Calculate residuals and RMSE
        os_residual = os_fitted - invert_log(os_train, self.to_log)[self.y]
        os_rmse = abs(os_predict - invert_log(os_test, self.to_log)[self.y])
        os_mean_rmse = rmse(os_predict, invert_log(os_test, self.to_log)[self.y])

        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse

class OneStepTimeSeriesValidationOLS:
    def __init__(self, df_x, df_y_log, fit, to_log_y, y ):
        self.df_x = df_x
        self.df_y_log = df_y_log
        self.fit = fit
        self.to_log_y = to_log_y
        self.y = y

    def run_validation(self):
        """Runs the one-step time series validation and returns predictions, residuals, and RMSE."""
        #code to split into train/test sets
        os_x_train, os_x_test, os_y_train, os_y_test = train_test_split(self.df_x, self.df_y_log, test_size=0.3, shuffle=False)
        
        #code to forecast and compute predictions
        os_predict = invert_log(self.fit.predict(os_x_test).to_frame(name=self.y),self.to_log_y)[self.y]
        os_fitted = invert_log(self.fit.predict(os_x_train).to_frame(name=self.y),self.to_log_y)[self.y]

        #code to calculate residuals and RMSE
        os_residual =  os_fitted - invert_log(os_y_train, self.to_log_y)[self.y]
        os_rmse = abs(os_predict-invert_log(os_y_test, self.to_log_y)[self.y])
        os_mean_rmse = rmse(invert_log(os_y_test, self.to_log_y)[self.y], os_predict)

        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse


class OneStepTimeSeriesValidationVAR:
    def __init__(self, df_all_logdiff, model_lag, df_all_log, y, to_diff_y, to_log_y):
        self.df_all_logdiff = df_all_logdiff
        self.model_lag = model_lag
        self.df_all_log = df_all_log
        self.y = y
        self.to_diff_y = to_diff_y
        self.to_log_y = to_log_y

    def run_validation(self):
        """Runs the one-step time series validation and returns predictions, residuals, and RMSE."""
        #code to split into train/test sets
        os_train, os_test = train_test_split(self.df_all_logdiff, test_size=0.3, shuffle=False)
        os_model = VAR(endog=os_train)
        os_fit = os_model.fit(self.model_lag)
        
        #code to forecast and compute predictions
        os_predict = invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), pd.DataFrame(os_fit.forecast(steps=len(os_test), y=os_model.endog), columns=os_model.endog_names, index=os_test.index)[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]
        os_fitted = invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), os_fit.fittedvalues[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]

        #code to calculate residuals and RMSE
        os_residual =  os_fitted - invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), os_train[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]
        os_rmse = abs(os_predict-invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), os_test[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y])
        os_mean_rmse = rmse(os_predict,invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), os_test[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y])

        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse

class OneStepTimeSeriesValidationVECM:
    def __init__(self, df_all_log, terms, optimal_lags, coint, to_log_y, y):
        self.df_all_log = df_all_log
        self.terms = terms
        self.optimal_lags = optimal_lags
        self.coint = coint
        self.to_log_y = to_log_y
        self.y = y
      

    def run_validation(self):
        """Runs the one-step time series validation and returns predictions, residuals, and RMSE."""
        #code to split into train/test sets
        os_train, os_test = train_test_split(self.df_all_log, test_size=0.3, shuffle=False)
        os_model = VECM(endog=os_train, deterministic=self.terms, k_ar_diff=self.optimal_lags, coint_rank=self.coint)
        os_fit = os_model.fit()
        
        #code to forecast and compute predictions
        os_predict = invert_log(pd.DataFrame(os_fit.predict(steps=len(os_test)), columns=os_model.endog_names, index=os_test.index)[self.y].to_frame(name=self.y), self.to_log_y)[self.y]
        os_fitted = invert_log(pd.DataFrame(os_fit.fittedvalues, columns=os_model.endog_names, index=os_train.iloc[self.optimal_lags+1:,:].index)[self.y].to_frame(name=self.y), self.to_log_y)[self.y]

        #code to calculate residuals and RMSE
        os_residual =  os_fitted - invert_log(os_train[self.y].to_frame(name=self.y), self.to_log_y)[self.y]
        os_rmse = abs(os_predict-invert_log(os_test[self.y].to_frame(name=self.y), self.to_log_y)[self.y])
        os_mean_rmse = rmse(os_predict,invert_log(os_test[self.y].to_frame(name=self.y), self.to_log_y)[self.y])
        
        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse

class OneStepTimeSeriesValidationRW:
    def __init__(self, df, fit, y):
        self.df = df
        self.fit = fit
        self.y = y
      

    def run_validation(self):
        """Runs the one-step time series validation and returns predictions, residuals, and RMSE."""
        #code to split into train/test sets
        os_train, os_test = train_test_split(self.df, test_size=0.3, shuffle=False)
        osv = self.fit.apply(os_train)
        
        #code to forecast and compute predictions
        os_predict = osv.forecast(len(os_test)).to_frame(name=self.y)[self.y]
        os_fitted = osv.fittedvalues.to_frame(name=self.y)[self.y]

        #code to calculate residuals and RMSE
        os_residual =  os_fitted - os_train[self.y]
        os_rmse = abs(os_predict - os_test[self.y])
        os_mean_rmse = rmse(os_predict, os_test[self.y])
        
        return os_predict, os_fitted, os_residual, os_rmse, os_mean_rmse


class TimeSeriesCrossValidationARDL:
    def __init__(self, df_y_log, df_x_future, fit, y, to_log_y, lags, horizon, df_x,splits):
        self.df_y_log = df_y_log
        self.df_x_future = df_x_future
        self.fit = fit
        self.y = y
        self.to_log_y = to_log_y
        self.lags = lags
        self.horizon = horizon
        self.df_x = df_x
        self.splits = splits

    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_rmse = []
        tscv_error = []

        #code to get the maximum split possible for time series cross validation
        self.splits = int(np.floor(((len(self.df_x)-self.lags)/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv' + str(i))
            residual_list.append('resid' + str(i))

        for train_index, test_index in tscv.split(self.df_y_log):
            #code to split into train/test sets
            cv_y_train, cv_y_test = self.df_y_log.iloc[train_index], self.df_y_log.iloc[test_index]
            cv_x_train = self.df_x_future.copy(deep=True).loc[cv_y_train.index[0]:cv_y_train.index[-1] + DateOffset(months=self.lags * 3)]
            cv_x_test = self.df_x_future.copy(deep=True).loc[cv_y_test.index[0]:cv_y_test.index[-1] + DateOffset(months=self.lags * 3)]

            #code to forecast and compute predictions
            cv_predict = invert_log(self.fit.forecast(steps=len(cv_y_test), exog=cv_x_test).to_frame(name=self.y).set_index(cv_y_test.index), self.to_log_y)[self.y]
            cv_fitted = invert_log(self.fit.forecast(steps=len(cv_y_train), exog=cv_x_train).to_frame(name=self.y).set_index(cv_y_train.index), self.to_log_y)[self.y]

            #code to calculate residuals and errors
            cv_residual = cv_fitted - invert_log(cv_y_train, self.to_log_y)[self.y]
            cv_error = abs(cv_predict - invert_log(cv_y_test, self.to_log_y)[self.y]).array
            cv_rmse = rmse(cv_predict, invert_log(cv_y_test, self.to_log_y)[self.y])

            #code to append results to lists
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_rmse.append(cv_rmse)
            tscv_error.append(cv_error)

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list

class TimeSeriesCrossValidationARIMA:
    def __init__(self, model, y, to_log, df_log, fit, splits, horizon, df):
        self.model = model
        self.y = y
        self.to_log = to_log
        self.df_log = df_log
        self.fit = fit
        self.splits = splits
        self.horizon = horizon
        self.df = df

    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_error = []
        tscv_rmse = []

        #code to get the maximum split possible for time series cross validation
        self.splits = int(np.floor((len(self.df)/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv' + str(i))
            residual_list.append('resid' + str(i))

        for train_index, test_index in tscv.split(self.df_log):
            cv_train, cv_test = self.df_log.iloc[train_index], self.df_log.iloc[test_index]
            
            cv = self.fit.apply(cv_train)
            
            cv_predict = invert_log(cv.forecast(self.horizon).to_frame(name=self.y), self.to_log)[self.y]
            cv_fitted = invert_log(cv.fittedvalues.to_frame(name=self.y), self.to_log)[self.y]
            cv_residual = cv_fitted - invert_log(cv_train, self.to_log)[self.y]
            cv_error = abs(cv_predict-invert_log(cv_test, self.to_log)[self.y]).array
            cv_rmse = rmse(cv_predict, invert_log(cv_test, self.to_log)[self.y])

            #code to append results to lists
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_error.append(cv_error)
            tscv_rmse.append(cv_rmse)

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list

class TimeSeriesCrossValidationOLS:
    def __init__(self, splits, horizon, df_x, df_y_log, fit, to_log_y, y):
      self.splits = splits
      self.horizon = horizon
      self.df_x = df_x
      self.df_y_log = df_y_log
      self.fit = fit
      self.to_log_y = to_log_y
      self.y = y


    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_rmse = []
        tscv_error = []

        #code to get the maximum split possible for time series cross validation
        self.splits = int(np.floor((len(self.df_x)/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv' + str(i))
            residual_list.append('resid' + str(i))

        for train_index, test_index in tscv.split(self.df_y_log):
            cv_x_train, cv_x_test, cv_y_train, cv_y_test = self.df_x.iloc[train_index], self.df_x.iloc[test_index], self.df_y_log.iloc[train_index], self.df_y_log.iloc[test_index]

            cv_predict = invert_log(self.fit.predict(cv_x_test).to_frame(name=self.y),self.to_log_y)[self.y]
            cv_fitted = invert_log(self.fit.fittedvalues.to_frame(name=self.y),self.to_log_y)[self.y]
            
            cv_residual = cv_fitted - invert_log(cv_y_train, self.to_log_y)[self.y]
            cv_error = abs(cv_predict-invert_log(cv_y_test,self.to_log_y)[self.y]).array
            cv_rmse = rmse(cv_predict, invert_log(cv_y_test,self.to_log_y)[self.y])

            tscv_error.append(cv_error)
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_rmse.append(cv_rmse)
        

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list

class TimeSeriesCrossValidationVAR:
    def __init__(self, splits, horizon, df_all_logdiff, model_lag, df_all_log, to_diff_y, y, lags, to_log_y):
      self.splits = splits
      self.horizon = horizon
      self.df_all_logdiff = df_all_logdiff
      self.model_lag = model_lag
      self.df_all_log = df_all_log
      self.to_diff_y = to_diff_y
      self.y = y
      self.lags = lags
      self.to_log_y = to_log_y


    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_rmse = []
        tscv_error = []

        #code to get the maximum split possible for time series cross validation
        self.splits = int(np.floor(((len(self.df_all_logdiff)-(self.lags))/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv'+str(i))
            residual_list.append('resid'+str(i))

        for train_index, test_index in tscv.split(self.df_all_logdiff):
            cv_train, cv_test = self.df_all_logdiff.iloc[train_index], self.df_all_logdiff.iloc[test_index]

            cv_model = VAR(endog=cv_train)
            cv_fit = cv_model.fit(self.model_lag)
            cv_predict = invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), pd.DataFrame(cv_fit.forecast(steps=len(cv_test), y=cv_model.endog), columns=cv_model.endog_names, index=cv_test.index)[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]
            cv_fitted = invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), cv_fit.fittedvalues[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]
            cv_residual =  cv_fitted - invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), cv_train[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]
            cv_error = abs(cv_predict-invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), cv_test[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y]).array
            cv_rmse = rmse(cv_predict, invert_log(invert_diff(self.df_all_log[self.y].to_frame(name=self.y), cv_test[self.y].to_frame(name=self.y), self.to_diff_y), self.to_log_y)[self.y])
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_rmse.append(cv_rmse)
            tscv_error.append(cv_error)

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list

class TimeSeriesCrossValidationVECM:
    def __init__(self, splits, horizon, df_all_log, lags, terms, optimal_lags, coint, to_log_y, y):
        self.splits = splits
        self.horizon = horizon
        self.df_all_log = df_all_log
        self.lags = lags
        self.terms = terms
        self.optimal_lags = optimal_lags
        self.coint = coint
        self.to_log_y = to_log_y
        self.y = y

    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_rmse = []
        tscv_error = []

        #code to get the maximum split possible for time series cross validation
        self.splits = int(np.floor(((len(self.df_all_log)-(self.lags+1))/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv'+str(i))
            residual_list.append('resid'+str(i))

        for train_index, test_index in tscv.split(self.df_all_log):
            cv_train, cv_test = self.df_all_log.iloc[train_index], self.df_all_log.iloc[test_index]

            cv_model = VECM(endog=cv_train, deterministic=self.terms, k_ar_diff=self.optimal_lags, coint_rank=self.coint)
            cv_fit = cv_model.fit()
            cv_predict = invert_log(pd.DataFrame(cv_fit.predict(steps=len(cv_test)), columns=cv_model.endog_names, index=cv_test.index)[self.y].to_frame(name=self.y), self.to_log_y)[self.y]
            cv_fitted = invert_log(pd.DataFrame(cv_fit.fittedvalues, columns=cv_model.endog_names, index=cv_train.iloc[self.optimal_lags+1:,:].index)[self.y].to_frame(name=self.y), self.to_log_y)[self.y]
            cv_residual =  cv_fitted - invert_log(cv_train[self.y].to_frame(name=self.y), self.to_log_y)[self.y]
            cv_error = abs(cv_predict-invert_log(cv_test[self.y].to_frame(name=self.y), self.to_log_y)[self.y]).array
            cv_rmse = rmse(cv_predict, invert_log(cv_test[self.y].to_frame(name=self.y), self.to_log_y)[self.y], axis=0)
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_rmse.append(cv_rmse)
            tscv_error.append(cv_error)

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list


class TimeSeriesCrossValidationRW:
    def __init__(self, splits, horizon, df, fit, y):
        self.splits = splits
        self.horizon = horizon
        self.df = df
        self.fit = fit
        self.y = y

    def run_cross_validation(self):
        """Runs time series cross-validation and returns predictions, residuals, RMSE, and error."""
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=self.splits, test_size=self.horizon)
        tscv_list = []
        residual_list = []
        tscv_residual = []
        tscv_predict = []
        tscv_error = []
        tscv_rmse = []

        self.splits = int(np.floor((len(self.df)/self.horizon) - 1))

        for i in range(self.splits):
            tscv_list.append('tscv'+str(i))
            residual_list.append('resid'+str(i))

        for train_index, test_index in tscv.split(self.df):
            cv_train, cv_test = self.df.iloc[train_index], self.df.iloc[test_index]

            cv = self.fit.apply(cv_train)

            cv_predict = cv.forecast(self.horizon).to_frame(name=self.y)[self.y]
            cv_fitted = cv.fittedvalues.to_frame(name=self.y)[self.y]
            cv_residual = cv_fitted - cv_train[self.y]
            cv_error = abs(cv_predict - cv_test[self.y]).array
            cv_rmse = rmse(cv_predict, cv_test[self.y])
            tscv_residual.append(cv_residual)
            tscv_predict.append(cv_predict)
            tscv_error.append(cv_error)
            tscv_rmse.append(cv_rmse)

        return tscv_predict, tscv_residual, tscv_rmse, tscv_error, tscv_list, residual_list
