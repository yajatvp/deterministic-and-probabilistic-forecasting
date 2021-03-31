#%% Yajat Pandya - Project 2: MECH 6342
# Load .csv data file

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.cd('C:\Users\yajat\Documents\courses\renewable energy\Project 2')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.io import savemat
import scipy.io

dta2014 = pd.read_csv('2014.csv')
dta2015 = pd.read_csv('2015.csv')
dta = pd.read_csv('Project_2_dataset.csv')

dta.columns = dta.columns.str.replace('.', '_')

#%% DETERMINISTIC FORECASTING: Persistence of Cloudiness forecast

dta['SPI'] = np.divide(dta.GHI,dta.Clearsky_GHI) 
poc = np.zeros(np.size(dta.GHI)-1)
poc = np.array(dta.GHI[0:-1]) + np.multiply(np.array(dta.SPI[0:-1]), 
                                        np.array(dta.Clearsky_GHI[1:]) - np.array(dta.Clearsky_GHI[0:-1]))
poc = np.nan_to_num(poc)
#poc = dta.GHI[0:-1] + np.multiply(dta.SPI[0:-1], dta.Clearsky_GHI[1:] - dta.GHI[0:-1])

dta['GHI_POC'] = np.zeros(np.size(dta.GHI))
dta.GHI_POC[1:] = poc

RSquared_poc = r2_score(dta.GHI[8760:], dta.GHI_POC[8760:])
rmse_poc = np.sqrt(mean_squared_error(dta.GHI[8760:], dta.GHI_POC[8760:]))
mae_poc = mean_absolute_error(dta.GHI[8760:], dta.GHI_POC[8760:])
#%% Random Forest based forecast
# build our RF model
RF_Model = RandomForestRegressor(n_estimators=100,
                                 max_features="auto", oob_score=True)

labels = dta.GHI[0:8760, None]#[:, None]
features = np.array([dta.Clearsky_GHI[0:8760],dta.Temperature[0:8760],dta.Relative_Humidity[0:8760]])#dta.Clearsky_GHI[0:8760, None]#[:, None]

# Fit the RF model with features and labels.
rgr=RF_Model.fit(np.transpose(features), labels)

X_test_predict=pd.DataFrame(
    rgr.predict(np.transpose(np.array([dta.Clearsky_GHI[8760:],dta.Temperature[8760:],dta.Relative_Humidity[8760:]]))))
    
X_train_predict=pd.DataFrame(
    rgr.predict(np.transpose(np.array([dta.Clearsky_GHI[0:8760],dta.Temperature[0:8760],dta.Relative_Humidity[0:8760]]))))
#pd.DataFrame(rgr.predict(dta.Clearsky_GHI[0:8760, None]))#.set_index('GHI')

# combine the training and testing dataframes to visualize
# and compare.
RF_predict = X_train_predict.append(X_test_predict,ignore_index=True).rename(
    columns={0:'GHI'})
dta['GHI_RF'] = np.array(RF_predict.GHI)

dta['diffRF'] = dta.GHI - dta.GHI_RF

dta[['GHI', 'GHI_RF']].plot()
dta['diffRF'].plot()

RSquared_RF = r2_score(dta.GHI[8760:], X_test_predict)
rmse_RF = np.sqrt(mean_squared_error(dta.GHI[8760:], X_test_predict))
mae_RF = mean_absolute_error(dta.GHI[8760:], X_test_predict)

#%% Random Forest (RF1) - Only timestamp, prev. GHI, Solar Zenith Angle feature
# build our RF model
RF_Model = RandomForestRegressor(n_estimators=100,
                                 max_features="auto", oob_score=True)

dta['day'] = pd.DatetimeIndex(pd.to_datetime(dta.Timestamp)).dayofyear
dta['hour'] = pd.DatetimeIndex(pd.to_datetime(dta.Timestamp)).hour

labels = dta.GHI[1:8760]#[:, None]
features = np.array([dta.GHI[0:8759],dta.day[0:8759],dta.hour[0:8759],dta.Solar_Zenith_Angle[0:8759]])#dta.Clearsky_GHI[0:8760, None]#[:, None]

# Fit the RF model with features and labels.
rgr=RF_Model.fit(np.transpose(features), labels)

X_test_predict=pd.DataFrame(
    rgr.predict(np.transpose(np.array([dta.GHI[8759:-1],dta.day[8759:-1],dta.hour[8759:-1],dta.Solar_Zenith_Angle[8759:-1]]))))
    
X_train_predict=pd.DataFrame(
    rgr.predict(np.transpose(np.array([dta.GHI[0:8759],dta.day[0:8759],dta.hour[0:8759],dta.Solar_Zenith_Angle[0:8759]]))))
#pd.DataFrame(rgr.predict(dta.Clearsky_GHI[0:8760, None]))#.set_index('GHI')

# combine the training and testing dataframes to visualize
# and compare.
RF_predict = X_train_predict.append(X_test_predict,ignore_index=True).rename(
    columns={0:'GHI'})
dta['GHI_RF'] = np.append(0, np.array(RF_predict.GHI))


dta['diffRF'] = dta.GHI - dta.GHI_RF

dta[['GHI', 'GHI_RF']].plot()
dta['diffRF'].plot()

RSquared_RF = r2_score(dta.GHI[8760:], X_test_predict)
rmse_RF = np.sqrt(mean_squared_error(dta.GHI[8760:], X_test_predict))
mae_RF = mean_absolute_error(dta.GHI[8760:], X_test_predict)
#%% Random Forest (RF2) - like previous, but includes all training data before the target hour
# Takes terribly long time on a 16 GiB memory PC
X_test_predict_hr = np.zeros(np.size(dta2014.GHI))
for i in range(8760,17520):    
    #let's get the labels and features in order to run our model fitting
    labels = dta.GHI[1:i]#[:, None]
    features = np.array([dta.GHI[0:i-1],dta.day[0:i-1],dta.hour[0:i-1],dta.Solar_Zenith_Angle[0:i-1]])#dta.Clearsky_GHI[0:8760, None]#[:, None]
    
    # Fit the RF model with features and labels.
    rgr=RF_Model.fit(np.transpose(features), labels)
    
    # Now that we've run our models and fit it, let's create
    # dataframes to look at the results
    X_test_predict_hr[i-8760]=(
        rgr.predict(np.transpose(np.array([dta.GHI[i-1],dta.day[i-1],dta.hour[i-1],dta.Solar_Zenith_Angle[i-1]])).reshape(1, -1)))
        
    print(i)
    #X_train_predict=pd.DataFrame(
        #rgr.predict(np.transpose(np.array([dta.Clearsky_GHI[0:8760],dta.Temperature[0:8760],dta.Relative_Humidity[0:8760]]))))
    #pd.DataFrame(rgr.predict(dta.Clearsky_GHI[0:8760, None]))#.set_index('GHI')
    
    # combine the training and testing dataframes to visualize
    # and compare.
#%% Error metrics
RSquared_RF_hr = r2_score(dta.GHI[8760:], X_test_predict_hr)
rmse_RF_hr = np.sqrt(mean_squared_error(dta.GHI[8760:], X_test_predict_hr))
mae_RF_hr = mean_absolute_error(dta.GHI[8760:], X_test_predict_hr)

X_test_hr=pd.DataFrame(X_test_predict_hr)
X_test_hr=X_test_hr.rename(columns={0:'GHI'})

RF_predict_hr = dta.GHI[0:8760].append(X_test_hr.GHI,ignore_index=True)
dta['GHI_RF_hr'] = np.array(RF_predict_hr)

# Save to post-process in MATLAB
savemat("run1_det.mat", {name: col.values for name, col in dta.items()})


#%% PROBABILISTIC FORECASTING: Sigma surrogate modeling using SVR and sigma prediction on 2015 time stamps
sg = scipy.io.loadmat('sigma_surrogate.mat')
sigma = sg['sigma']
sig=np.transpose(sigma[0][0:-1])

features = np.array([dta.GHI[0:8759],dta.day[0:8759],dta.hour[0:8759],dta.Solar_Zenith_Angle[0:8759]])

SVR_model = SVR()
svr = SVR_model.fit(np.transpose(features), sig)

sig1 = pd.DataFrame(
    svr.predict(np.transpose(np.array([dta.GHI[8759:-1],dta.day[8759:-1],dta.hour[8759:-1],dta.Solar_Zenith_Angle[8759:-1]]))))

# Save .mat to post-process in MATLAB
savemat("run1_sig.mat", {'sig1': np.array(sig1)})
