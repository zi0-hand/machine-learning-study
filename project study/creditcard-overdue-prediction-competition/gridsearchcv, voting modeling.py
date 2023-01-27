from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def gridcv_process(model, param, cv_num, n_thread):
    model_cv = GridSearchCV(model, param, cv=cv_num, n_jobs=n_thread, refit=True)
    model_cv.fit(X_train, y_train)
    print('GridSearchCV 최적 파라미터:', model_cv.best_params_)
    print('GridSearchCV 최고 정확도:{0:.4f}'.format(model_cv.best_score_))

def voting_process(models):
    lst = []
    for i_num, model in enumerate(models):
        ele = (model.__class__.__name__, model)
        lst.append(ele)
    vo_clf = VotingClassifier(estimators= lst, voting='soft')
    vo_clf.fit(X_train, y_train)
    y_pred = vo_clf.predict_proba(X_test)
    print(f"log_loss: {log_loss(y_test['credit'], y_pred)}")


