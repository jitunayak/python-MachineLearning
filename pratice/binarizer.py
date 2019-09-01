from sklearn import preprocessing

input_dat = [[2.3,4.5,1,2,1.5]]

binarized = preprocessing.Binarizer(threshold=2).transform(input_dat)

print(binarized)