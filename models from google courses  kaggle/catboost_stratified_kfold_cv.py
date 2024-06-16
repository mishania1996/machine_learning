pip install catboost

import numpy as np
import pandas as pd 

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import warnings
warnings.filterwarnings("ignore")

RAND_VAL=42
num_folds=5 ## Number of folds (the meaning is explained later)
n_est=6000 ## Number of estimators

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#then modified the last one above
feat_cols=df_train.columns.drop(['label'])
X = df_train[feat_cols]
y = df_train['label']
cat_features = np.where(X.dtypes != np.float64)[0] #category features

#actual model defining and training

#(1)The dataset is split into kk equally sized folds (subsets).
#(2)For each of the k iterations, one fold is used as the validation set, and the remaining k−1 folds are combined to form the training set.
#(3)Process is repeated k times
folds = StratifiedKFold(n_splits=num_folds,random_state=RAND_VAL,shuffle=True)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    # Split data into training and validation sets for the current fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # Create CatBoost Pool objects for training and validation data
    train_pool = Pool(X_train, y_train,cat_features=cat_features)
    val_pool = Pool(X_val, y_val,cat_features=cat_features)
    
    # Initialize and train the CatBoostClassifier
    model = CatBoostClassifier(
    eval_metric='AUC',
    task_type='GPU',
    learning_rate=0.02,
    iterations=n_est)
    
    # Evaluate the model on the validation set
    model.fit(train_pool, eval_set=val_pool,verbose=300)
    
    y_pred_val = model.predict_proba(X_val[feat_cols])[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print("AUC for fold ",n_fold,": ",auc_val)
    
    # Generate predictions for the test set
    y_pred_test = model.predict_proba(df_test[feat_cols])[:,1]
    print("----------------")
  
  
  
prediction = pd.DataFrame(y_pred_test)

prediction.to_csv("predictions_final",index = False)

'''
-----additional info on the model----------
CatBoostClassifier is a powerful machine learning algorithm for classification tasks. It is a type of gradient boosting on decision trees that is designed to handle categorical features more effectively and efficiently than many other algorithms.

***Key Features***:

Handles categorical features natively without the need for extensive preprocessing like one-hot encoding. It uses an efficient method to convert categorical features into numerical values during training.
CatBoost is optimized for both CPU and GPU, making it suitable for large datasets and complex models.
The algorithm includes various regularization techniques and overfitting detectors to prevent overfitting and improve generalization.
CatBoost can handle missing values directly, which simplifies the data preprocessing pipeline.

***Important parameters***

iterations: Number of boosting iterations (trees). More iterations can lead to better performance but can also increase the risk of overfitting.
learning_rate: The learning rate determines the step size at each iteration while moving toward a minimum of the loss function.
depth: The maximum depth of the tree. Deeper trees can capture more complex patterns but can also overfit.
eval_metric: Metric used for evaluation (e.g., 'AUC', 'Accuracy', 'Logloss').
cat_features: List of categorical features. These features will be treated as categorical by CatBoost.
task_type: 'CPU' or 'GPU'. Specifies whether to use the CPU or GPU for training.
random_seed: Seed for random number generation to ensure reproducibility.
l2_leaf_reg: Coefficient at the L2 regularization term of the cost function.
bagging_temperature: Controls the intensity of Bayesian bagging. Lower values make the algorithm more conservative.
'''
