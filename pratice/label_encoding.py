from sklearn import preprocessing

input_labels =['red','blue','orange','black','pink']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

for i,item in enumerate(encoder.classes_):
    print(item,"-->",i)

test_labels =['pink','red']

encoded_values = encoder.transform(test_labels)

print(list(encoded_values))

#inverse transform

values = [1,2]
d = encoder.inverse_transform(values)
print(list(d))