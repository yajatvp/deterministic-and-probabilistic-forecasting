# deterministic-and-probabilistic-forecasting
1-hr ahead deterministic and probabilistic forecasting of Global Horizontal Irradiance (GHI) index using Random Forest and Support Vector Regression in Python.

Summary:
A historical dataset for two years (2014-15) of solar energy parameters like the hourly GHI, Wind speed, Clear-sky GHI, Temperature, Pressure etc. of a particular location is provided. Using Python’s scikitlearn package, a time series forecasting of the hourly GHI value for the year 2015 is conducted. This has been done for Deterministic and Probabilistic based forecasting methods. A Random Forest based algorithm has been used for the deterministic forecasting which has been compared with the Persistence of Cloudiness (POC) method of forecast. Support Vector Regression (SVR) model is used to predict the standard deviation values in case of probabilistic forecasts. Corresponding error metrics including MAE, RMSE and mean pinball error are calculated.
