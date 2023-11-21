from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(min_samples_split=10,class_weight={1:100},random_state=0)
rf.fit(X_train, y_train)
y_train_preds = rf.predict(X_train)
pickle.dump(rf,open('randomforestmodel.pkl','wb'))