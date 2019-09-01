from sklearn import preprocessing
input_dat = [[90.3,44.85,41,32,19.85], 
                [60.3,24.25,81,52,79.85],
                [30.3,14.25,51,32,49.95]]

m = preprocessing.MinMaxScaler(feature_range=(0,10))
p = m.fit_transform(input_dat)
print(p)