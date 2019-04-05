import os
import pandas as pd
import json

train = json.load(open('/home/lucas/Documentos/kaggle/cooking/train.json'))
test = json.load(open('/home/lucas/Documentos/kaggle/cooking/test.json'))

train = [(item['cuisine'], ' '.join(item['ingredients']).lower()) for item in train]
test = [' '.join(item['ingredients']).lower() for item in test]

train[2:5]

test = train[:int(len(train)*0.2)]
train = train[int(len(train)*0.2):]
len(test) + len(train)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(binary=True)


x = tfidf.fit_transform([item[1] for item in train])
x = x.astype('float16')

x_test = tfidf.transform([item[1] for item in test])
x_test = x_test.astype('float16')

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform([item[0] for item in train])

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x, y)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x, y)

y_test = classifier.predict(x_test)
y_pred = lb.inverse_transform(y_test)

rfc_y_test = rfc.predict(x_test)
rfc_y_pred = lb.inverse_transform(rfc_y_test)

len(y_pred)
print(y_pred[i], test[i][0])

rst = [(a, b) for a, b in zip(y_pred, [item[0] for item in test]) if a != b]
1-(len(rst)/len(y_pred))


rfc_rst = [(a, b) for a, b in zip(rfc_y_pred, [item[0] for item in test]) if a != b]
1-(len(rfc_rst)/len(rfc_y_pred))





print(len([item for item in y_pred]))


df = pd.DataFrame()
df = pd.read_json('/home/lucas/Documentos/kaggle/cooking/train.json')

test = pd.get_dummies(df.iloc[:, 0], drop_first=True)
# test.drop(test.columns[1], axis=1, inplace=True)
# test = test.iloc[:, 1:]

aux = pd.concat([df, pd.get_dummies(df.iloc[:, 0], drop_first=True)], axis=1)

aux.drop(columns=['cuisine'], axis=1, inplace=True)

[item.loc['ingredients'] for item in aux.iloc[:, :]]

aux.shape