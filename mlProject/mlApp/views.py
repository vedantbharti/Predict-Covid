from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import Holt
import joblib
from covid import Covid


covid = Covid()
India_cases = covid.get_status_by_country_name("India")
Confirmed_India = India_cases['confirmed']
Active_India = India_cases['active']
Recovered_India = India_cases['recovered']
Deaths_India = India_cases['deaths']
mortality_rate_India = (Deaths_India/Confirmed_India)*100
recovery_rate_India = (Recovered_India/Confirmed_India)*100

Active_World = covid.get_total_active_cases()
Confirmed_World = covid.get_total_confirmed_cases()
Recovered_World = covid.get_total_recovered()
Deaths_World = covid.get_total_deaths()
mortality_rate_World = (Deaths_World/Confirmed_World)*100
recovery_rate_World = (Recovered_World/Confirmed_World)*100

def index(request):
    context = {'Confirmed_India':Confirmed_India,
                'Active_India':Active_India,
                'Recovered_India':Recovered_India,
                'Deaths_India':Deaths_India,
                'mortality_rate_India':mortality_rate_India,
                'recovery_rate_India':recovery_rate_India,
                'Active_World':Active_World,
                'Confirmed_World':Confirmed_World,
                'Recovered_World':Recovered_World,
                'Deaths_World':Deaths_World,
                'mortality_rate_World':mortality_rate_World,
                'recovery_rate_World':recovery_rate_World
                }
    return render(request,'index.html',context)

def predict(request):
    if request.method=='POST':
        Region = request.POST.get('Region')
        year = int(request.POST.get('Year'))
        month = int(request.POST.get('Month'))
        date = int(request.POST.get('Date'))

        covid19 = pd.read_csv('mlApp/static/covid_19_clean_complete.csv')
        covid19.drop(['Province/State','WHO Region'], axis=1, inplace=True)
        covid19['Date'] = pd.to_datetime(covid19['Date'])
        byDate = covid19.groupby(['Date']).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'})
        covid19_India = covid19[covid19['Country/Region']=='India']
        byDate_India = covid19_India.groupby(['Date']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})

        if Region == 'World':
            byDate['Days Since'] = byDate.index-byDate.index[0]
            byDate['Days Since'] = byDate['Days Since'].dt.days
            train_data = byDate.iloc[:int(byDate.shape[0]*0.95)]
            test_data = byDate.iloc[int(byDate.shape[0]*0.95):]
            SVM = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)
            SVM.fit(np.array(train_data['Days Since']).reshape(-1,1),np.array(train_data['Confirmed']).reshape(-1,1))
            prediction_SVM = SVM.predict(np.array(test_data['Days Since']).reshape(-1,1))
            rmse_SVM = np.sqrt(mean_squared_error(np.array(test_data['Confirmed']).reshape(-1,1),prediction_SVM))

            accuracy_SVM = (1-np.mean(np.abs((np.array(test_data['Confirmed']).reshape(-1,1)-prediction_SVM)/np.array(test_data['Confirmed']).reshape(-1,1))))*100

            d = byDate.index[-1]
            d1 = dt.date(year,month,date)
            d2 = d.to_pydatetime().date()
            n = (d1-d2).days + 1
            if(n<=1):
                return HttpResponse('Wrong input value for the prediction date')
            new_date = []
            new_prediction_SVM = []
            for i in range(1,n):
              new_date.append(byDate.index[-1]+timedelta(days=i))
              new_prediction_SVM.append(SVM.predict(np.array(byDate['Days Since'].max()+i).reshape(-1,1))[0])
            pd.set_option('display.float_format',lambda x: '%.f'%x)
            model_predictions = pd.DataFrame(zip(new_date,new_prediction_SVM),columns=['Dates','SVR'])

            ans_SVM = int(round(model_predictions['SVR'].iloc[-1]))

            train_data_ts = byDate.iloc[:int(byDate.shape[0]*0.85)]
            test_data_ts = byDate.iloc[int(byDate.shape[0]*0.85):]
            holt = Holt(np.asarray(train_data_ts['Confirmed']),exponential=True).fit(smoothing_level=0.5,smoothing_slope=0.5)
            y_prediction = test_data_ts.copy()
            y_prediction['Holt'] = holt.forecast(len(test_data_ts))
            Holt_accuracy = (1-np.mean(np.abs((np.array(test_data_ts['Confirmed']).reshape(-1,1)-np.array(y_prediction['Holt']).reshape(-1,1))/np.array(test_data_ts['Confirmed']).reshape(-1,1))))*100

            holt_new_date = []
            holt_new_prediction = []
            for i in range(1,n):
              holt_new_date.append(byDate.index[-1]+timedelta(days=i))
              holt_new_prediction.append(holt.forecast((len(test_data_ts)+i))[-1])

            model_predictions['Holt\'s linear model prediction'] = holt_new_prediction
            ans_Holt = int(round(model_predictions['Holt\'s linear model prediction'].iloc[-1]))

            context = { 'Confirmed_India':Confirmed_India,
                        'Active_India':Active_India,
                        'Recovered_India':Recovered_India,
                        'Deaths_India':Deaths_India,
                        'mortality_rate_India':mortality_rate_India,
                        'recovery_rate_India':recovery_rate_India,
                        'Active_World':Active_World,
                        'Confirmed_World':Confirmed_World,
                        'Recovered_World':Recovered_World,
                        'Deaths_World':Deaths_World,
                        'mortality_rate_World':mortality_rate_World,
                        'recovery_rate_World':recovery_rate_World,
                        'accuracy_SVM':accuracy_SVM,
                        'ans_SVM':ans_SVM,
                        'Holt_accuracy':Holt_accuracy,
                        'ans_Holt':ans_Holt
                        }

        else:
            byDate_India = byDate_India[byDate_India['Confirmed']!=0]
            byDate_India['Days Since'] = byDate_India.index-byDate_India.index[0]
            byDate_India['Days Since'] = byDate_India['Days Since'].dt.days
            train_data_India = byDate_India.iloc[:int(byDate_India.shape[0]*0.95)]
            test_data_India = byDate_India.iloc[int(byDate_India.shape[0]*0.95):]
            SVM_India = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)
            SVM_India.fit(np.array(train_data_India['Days Since']).reshape(-1,1),np.array(train_data_India['Confirmed']).reshape(-1,1))
            prediction_SVM_India = SVM_India.predict(np.array(test_data_India['Days Since']).reshape(-1,1))
            rmse_SVM_India = np.sqrt(mean_squared_error(np.array(test_data_India['Confirmed']).reshape(-1,1),prediction_SVM_India))

            accuracy_SVM_India = (1-np.mean(np.abs((np.array(test_data_India['Confirmed']).reshape(-1,1)-prediction_SVM_India)/np.array(test_data_India['Confirmed']).reshape(-1,1))))*100

            d = byDate_India.index[-1]
            d1 = dt.date(year,month,date)
            d2 = d.to_pydatetime().date()
            n = (d1-d2).days + 1
            if(n<=1):
                return HttpResponse('Wrong input value for the prediction date')
            new_date_India = []
            new_prediction_SVM_India = []
            for i in range(1,n):
              new_date_India.append(byDate_India.index[-1]+timedelta(days=i))
              new_prediction_SVM_India.append(SVM_India.predict(np.array(byDate_India['Days Since'].max()+i).reshape(-1,1))[0])
            pd.set_option('display.float_format',lambda x: '%.f'%x)
            model_predictions_India = pd.DataFrame(zip(new_date_India,new_prediction_SVM_India),columns=['Dates','SVR'])

            ans_SVM_India = int(round(model_predictions_India['SVR'].iloc[-1]))

            train_data_ts_India = byDate_India.iloc[:int(byDate_India.shape[0]*0.85)]
            test_data_ts_India = byDate_India.iloc[int(byDate_India.shape[0]*0.85):]
            holt = Holt(np.asarray(train_data_ts_India['Confirmed']),exponential=True).fit(smoothing_level=0.5,smoothing_slope=0.5)
            y_prediction_India = test_data_ts_India.copy()
            y_prediction_India['Holt'] = holt.forecast(len(test_data_ts_India))
            Holt_accuracy_India = (1-np.mean(np.abs((np.array(test_data_ts_India['Confirmed']).reshape(-1,1)-np.array(y_prediction_India['Holt']).reshape(-1,1))/np.array(test_data_ts_India['Confirmed']).reshape(-1,1))))*100

            holt_new_date_India = []
            holt_new_prediction_India = []
            for i in range(1,n):
              holt_new_date_India.append(byDate_India.index[-1]+timedelta(days=i))
              holt_new_prediction_India.append(holt.forecast((len(test_data_ts_India)+i))[-1])

            model_predictions_India['Holt\'s linear model prediction'] = holt_new_prediction_India
            ans_Holt_India = int(round(model_predictions_India['Holt\'s linear model prediction'].iloc[-1]))

            context = { 'Confirmed_India':Confirmed_India,
                        'Active_India':Active_India,
                        'Recovered_India':Recovered_India,
                        'Deaths_India':Deaths_India,
                        'mortality_rate_India':mortality_rate_India,
                        'recovery_rate_India':recovery_rate_India,
                        'Active_World':Active_World,
                        'Confirmed_World':Confirmed_World,
                        'Recovered_World':Recovered_World,
                        'Deaths_World':Deaths_World,
                        'mortality_rate_World':mortality_rate_World,
                        'recovery_rate_World':recovery_rate_World,
                        'accuracy_SVM':accuracy_SVM_India,
                        'ans_SVM':ans_SVM_India,
                        'Holt_accuracy':Holt_accuracy_India,
                        'ans_Holt':ans_Holt_India
                        }

    else:
        context = {'Confirmed_India':Confirmed_India,
                    'Active_India':Active_India,
                    'Recovered_India':Recovered_India,
                    'Deaths_India':Deaths_India,
                    'mortality_rate_India':mortality_rate_India,
                    'recovery_rate_India':recovery_rate_India,
                    'Active_World':Active_World,
                    'Confirmed_World':Confirmed_World,
                    'Recovered_World':Recovered_World,
                    'Deaths_World':Deaths_World,
                    'mortality_rate_World':mortality_rate_World,
                    'recovery_rate_World':recovery_rate_World
                    }

        return render(request,'index.html',context)

    return render(request,'index.html',context)
