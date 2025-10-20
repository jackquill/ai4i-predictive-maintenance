rf = RandomForestClassifier(
#     n_estimators=300, # number of trees 
#     class_weight="balanced",  # handle class imbalance
#     n_jobs=-1 # use all available cores
# )

# rf.fit(X_train, y_train)
# preds = rf.predict(X_test)
# print("Sample predictions:", preds[:20])
# print("Actual values:", y_test.values[:20])

# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# preds = rf.predict(X_test)

# print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
# print("\nClassification Report:\n", classification_report(y_test, preds))
# print("\nROC-AUC Score:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
