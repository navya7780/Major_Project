from xgboost import XGBClassifier
import xgboost as xgbc
import time
t1 = time.time()
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_train_preds = xgbc.predict(X_train)
pickle.dump(xgbc,open('xgbcmodel.pkl','wb'))
