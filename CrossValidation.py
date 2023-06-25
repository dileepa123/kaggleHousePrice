from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from Preprocess import X_full, y, numerical_cols, categorical_cols
from Validate import numerical_transformer, categorical_transformer


def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])),
        ('model', GradientBoostingRegressor(n_estimators=n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X_full, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

results = {50*i:get_score(50*i) for i in range(1,15) } # Your code here
print (results)