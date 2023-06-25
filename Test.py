import pandas as pd
from Preprocess import X_test
from Validate import my_pipeline
preds_test = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
