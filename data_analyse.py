from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv(r"C:/Users/WANGYONG/Desktop/internet+/data/tmp_data.csv")
df = pd.DataFrame(df)
df = shuffle(df)
df_x = pd.concat((df['soil humidity(0.85-0.9)'],df['air temperature(15-26)'],df["air humidity(0.6-0.8)"],df["illumination intensity( 2-6 klx)"]),axis=1)
df_y = pd.DataFrame(df['label'])
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=0)
model = LR()
reg = model.fit(x_train,y_train)
score = reg.score(x_test,y_test)
print("score:{}".format(score))

# print("x_train:{}".format(x_train.head()))
# print("y_train:{}".format(y_train.head()))
# print("x_test:{}".format(x_test.head()))
# print("y_test:{}".format(y_test.head()))

# print("x:{}".format(df_x.columns))
# reg = LR.fit()