from sklearn.ensemble import ExtraTreesClassifier
t1=time.time()
etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy",max_features=1.0,min_samples_leaf=3,min_samples_split=20,n_estimators=100)
etc.fit(X_train, y_train)
y_train_preds = etc.predict(X_train)
pickle.dump(etc,open('etcmodel.pkl','wb'))