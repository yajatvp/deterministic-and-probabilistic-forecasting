# deterministic-and-probabilistic-forecasting
1-hr ahead deterministic and probabilistic forecasting of Global Horizontal Irradiance (GHI) index using the Random Forest and Support Vector Regression in Python.

## Summary
A historical dataset for two years (2014-15) of solar energy parameters like the hourly GHI, Wind speed, Clear-sky GHI, Temperature, Pressure etc. of a particular location is provided. Using Python’s scikitlearn package, a time series forecasting of the 1-hr ahead GHI value for the year 2015 is conducted. A Random Forest based algorithm has been used for the deterministic forecasting which has been compared with the Persistence of Cloudiness (POC) method of forecast. Support Vector Regression (SVR) model is used to predict the standard deviation values at each timestamp in case of probabilistic forecasts. Corresponding error metrics including MAE, RMSE and mean pinball error are calculated. It is found that the pinball error of the probabilistic forecast is the lesser for night-time conditions due to less variability of GHI values. A normalized average of 3.01 units (W/m^2) was calculated for the complete 2015 forecast.

The data is available for both years in the CSV format. `forecast_main.py` script loads the data for both the years and computes the required operations using scikitlearn library in a section-wise method. `post_process.m` contains MATLAB based post-processing code to genereate the plots used in the final project report. Details on the project can be found in the report **Pandya_Project2_Report.pdf**.

