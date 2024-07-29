import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor

#データの読み込み
titanic_train_file = "../input/titanic/train.csv"
titanic_train_data = pd.read_csv(titanic_train_file)
print(titanic_train_data.columns)


#データの数を確認
column_num = titanic_train_data.shape[0]
row_num = titanic_train_data.shape[1]
print(column_num)
print(row_num)



#解析項目
features = ["Sex", "Age", "Fare"]

#Nanを含む行を削除
titanic_train_data = titanic_train_data.dropna(subset=features)

#データの数を確認
column_num = titanic_train_data.shape[0]
row_num = titanic_train_data.shape[1]
print(column_num)
print(row_num)

#訓練データの処理
y = titanic_train_data.Survived
X = titanic_train_data[features]
dummy_X = pd.get_dummies(X, drop_first=True, dtype=int)
mean = dummy_X.mean()

dummy_X.describe()



#モデルの作成
model = DecisionTreeRegressor(random_state=1)
model.fit(dummy_X, y)



#テストデータの読み込み
titanic_test_file = "../input/titanic/test.csv"
titanic_test_data = pd.read_csv(titanic_test_file)

#テストデータの処理
x = titanic_test_data[features]
dummy_x = pd.get_dummies(x, drop_first=True, dtype=int)
validated_x = dummy_x.fillna({"Sex_male": mean.Sex_male, "Age": mean.Age, "Fare": mean.Fare})

validated_x.describe()



#結果の予測
predict_y = model.predict(validated_x)

#ファイル出力
output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predict_y})
output.to_csv('submission.csv', index=False)