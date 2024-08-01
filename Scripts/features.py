def XT(X_train_full, X_test_full):
    categorical_cols = [cname for cname in X_train_full.columns if
                        X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if
                      X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_test = X_test_full[my_cols].copy()

    return X_test