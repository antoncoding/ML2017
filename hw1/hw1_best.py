import numpy as np
import pandas as pd
import random
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]


np.set_printoptions(suppress=True)
data = pd.read_csv(train_file,encoding='big5')

def Date_update_month(string):
    return int(string.split('/')[1])
def Date_update_day(string):
    return int(string.split('/')[2])

data['Month']=data['日期'].apply(Date_update_month)
data['Day']= data['日期'].apply(Date_update_day)

data = data[['Month', 'Day','測項', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']]

data = data.set_index(['Month','Day','測項']).stack().unstack(2).reset_index()
data = data.rename(columns={'level_2':'Hours'})
data.columns.name=None

data = data[['Month', 'Day', 'Hours','PM2.5', 'SO2','AMB_TEMP','O3','WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']]

df_pm25 = data['PM2.5'].apply(lambda x: int(x))
df_so2 = data['SO2'].apply(lambda x: float(x))
df_O3 = data['O3'].apply(lambda x: float(x))
df_wind = data['WIND_SPEED'].apply(lambda x: float(x))
df_dir = data['WIND_DIREC'].apply(lambda x: float(x))
df_windhr = data['WS_HR'].apply(lambda x: float(x))
df_dirhr = data['WD_HR'].apply(lambda x: float(x))
df_temp = data['AMB_TEMP'].apply(lambda x: float(x))

row = 12
col = 480

numpy_pm25 = np.matrix(data=df_pm25).reshape(row,col)
for i in range(row):
	for j in range(col):
		if numpy_pm25[i,j]==-1:
			numpy_pm25[i,j]=numpy_pm25[i,j-1]
            
numpy_pm25_square = np.power(numpy_pm25,2)
numpy_so2 = np.matrix(data=df_so2).reshape(row,col)
numpy_O3 = np.matrix(data=df_O3).reshape(row,col)
numpy_wind = np.matrix(data=df_wind).reshape(row,col)
numpy_dir = np.matrix(data=df_dir).reshape(row,col)
numpy_windhr = np.matrix(data= df_windhr).reshape(row,col)
numpy_dirhr = np.matrix(data=df_dirhr).reshape(row,col)

final_data = pd.read_csv(test_file,names=['id','Observation','1','2','3','4','5','6','7','8','9'])
final_data = final_data.set_index(['id','Observation']).stack().unstack(1).reset_index()
final_data.columns=['id', 'order', 'AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
final_data['id'] = final_data['id'].apply(lambda x: int(x.split('_')[1]))
final_data = final_data.sort_values(['id','order'])

final_pm25 = final_data['PM2.5'].apply(lambda x: int(x))
final_so2 = final_data['SO2'].apply(lambda x: float(x))
final_O3 = final_data['O3'].apply(lambda x: float(x))
final_wind = final_data['WIND_SPEED'].apply(lambda x: float(x))
final_dir = final_data['WIND_DIREC'].apply(lambda x: float(x))
final_windhr = final_data['WS_HR'].apply(lambda x: float(x))
final_dirhr = final_data['WD_HR'].apply(lambda x: float(x))

feature_count = 8
period = 9
train_data_length = (480-period)*12

final_numpy_pm25 = np.matrix(data=final_pm25).reshape(240,9)
final_numpy_pm25_square = np.matrix(data=final_pm25**2).reshape(240,9)
final_numpy_so2 = np.matrix(data=final_so2).reshape(240,9)
final_numpy_O3 = np.matrix(data=final_O3).reshape(240,9)
final_numpy_wind = np.matrix(data=final_wind).reshape(240,9)
final_numpy_dir = np.matrix(data=final_dir).reshape(240,9)
final_numpy_windhr = np.matrix(data=final_windhr).reshape(240,9)
final_numpy_dirhr = np.matrix(data=final_dirhr).reshape(240,9)

test_x = []
for i in range(0, 240):
    test_x.append(
        np.asarray(final_numpy_pm25[i,9-period:9]).tolist()[0] +
        np.asarray(final_numpy_pm25_square[i,9-period:9]).tolist()[0] +
        np.asarray(final_numpy_so2[i,9-period:9]).tolist()[0] +
        np.asarray(final_numpy_O3[i,9-period:9]).tolist()[0] +
        np.asarray(final_numpy_wind[i,9-period:9]).tolist()[0] +
        np.asarray(final_numpy_dir[i,9-period:9]).tolist()[0]+
        np.asarray(final_numpy_windhr[i,9-period:9]).tolist()[0]+
        np.asarray(final_numpy_dirhr[i,9-period:9]).tolist()[0]
    )
    
test_x = np.array(test_x)

train_x = []
train_y = np.zeros(train_data_length)
random.seed(777)
count_train_data = 0
j=0
for m in range(0,row):
    for h in range(0,col - period):
    	# j = j+1
        j = random.randint(1,3)
        if j ==3 :
            train_x.append( 
	            np.asarray(numpy_pm25[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_pm25_square[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_so2[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_O3[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_wind[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_dir[m,h:h+period]).tolist()[0]+
	            np.asarray(numpy_windhr[m,h:h+period]).tolist()[0] +
	            np.asarray(numpy_dirhr[m,h:h+period]).tolist()[0]
	                                         )
            train_y[count_train_data] = numpy_pm25[m,h+period]
            count_train_data +=1

train_x = np.array(train_x)

b = 0.0
w = np.zeros(feature_count*period)
w[period-1]=1
lr = 0.01
lr_all = np.full(feature_count*period,0.01)
r = 2.5
iteration = 10000

# Sum gradiant
sigma_b = 0.0
sigma = np.zeros(feature_count*period)

for i in range(iteration):

    grad_b = 0
    grad = np.zeros(feature_count*period)

    for n in range(count_train_data):    
        y_minus_f = train_y[n] - b - train_x[n].dot(w)
        
        # gradient update
        grad_b = - 2.0*(y_minus_f) + 2*r*b
        grad = -2*(y_minus_f)* train_x[n] +2*r*w
           
        # gradiant summary update
        sigma_b = sigma_b + grad_b**2
        sigma = sigma + grad**2

        # weight update
        b = b - lr/np.sqrt(sigma_b)*grad_b
        w = w - lr_all/np.sqrt(sigma)*grad

answer_list = []
error =0
for n in range(0,240):    
    answer = b + test_x[n].dot(w)
    answer_list.append(answer)

# make output.csv
output = pd.DataFrame(data=answer_list)
output.columns = ['value']
output = output.rename_axis('id')
output.reset_index(level=0,inplace=True)
output['id'] = output['id'].apply(lambda x: "id_" + str(x))
output.to_csv(output_file,index=False)