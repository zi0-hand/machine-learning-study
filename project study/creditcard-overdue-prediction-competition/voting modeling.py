from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

param={
    'num_leaves':[32,64],
    'min_data_in_leaf':[1,5,10],
    'colsample_bytree':[0.8,1],
    'n_estimators':[50, 100, 150]
}

def gridcv_process(model, param, cv_num, n_thread):
    model_cv = GridSearchCV(model, param, cv=cv_num, n_jobs=n_thread, refit=True)
    model_cv.fit(X_train, y_train)
    print('GridSearchCV 최적 파라미터:', GS_LGB.best_params_)
    print('GridSearchCV 최고 정확도:{0:.4f}'.format(GS_LGB.best_score_)

rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)

models_clf = [rf_clf, gb_clf, xgb_clf]

def voting_process(models):
    lst = []
    for i_num, model in enumerate(models):
        ele = (model.__class__.__name__, model)
        lst.append(ele)
    vo_clf = VotingClassifier(estimators= lst, voting='soft')
    vo_clf.fit(X_train, y_train)
    y_pred = vo_clf.predict_proba(X_test)
    print(f"log_loss: {log_loss(y_test['credit'], y_pred)}")


