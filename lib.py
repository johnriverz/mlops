import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer # final step of the pipeline

# brings in KNNImputer
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce

# for find_random_state function
from sklearn.neighbors import KNeighborsClassifier  #the KNN model
from sklearn.metrics import f1_score  #typical metric used to measure goodness of a model

sklearn.set_config(
    transform_output="pandas"
)  # says pass pandas tables through pipeline instead of numpy matrices

# random state variables
titanic_variance_based_split = 107
customer_variance_based_split = 113

# # bring in tatanic data in trimmed form
# lib_url = 'https://raw.githubusercontent.com/fickas/asynch_models/main/datasets/titanic_trimmed.csv'
# titanic_trimmed = pd.read_csv(lib_url)
# titanic_features = titanic_trimmed.drop(columns='Survived') # drop the label column

# two transformers
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

# transformed_df = titanic_transformer.fit_transform(titanic_features) # our transformed dataframe

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)

# save fitted transformer
fitted_pipeline = titanic_transformer.fit(X_train, y_train)  #notice just fit method called
import joblib
joblib.dump(fitted_pipeline, 'fitted_pipeline.pkl')

# This class will rename one or more columns.
class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
    # your __init__ method below
    def __init__(self, mapping_dict: dict):
        assert isinstance(
            mapping_dict, dict
        ), f"{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead."
        self.mapping_dict = mapping_dict

    # define fit to do nothing but give warning
    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    # write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
    def transform(self, X):
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."
        # assert self.mapping_dict in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_dict}"'  #column legit?

        missing_columns = set(self.mapping_dict.keys()) - set(X.columns)
        if missing_columns:
            missing_columns_str = ", ".join(missing_columns)
            raise AssertionError(
                f"{self.__class__.__name__}.transform: Columns not found in DataFrame: {missing_columns_str}"
            )

        # do actual transforming
        X_ = X.copy()
        X_.rename(columns=self.mapping_dict, inplace=True)
        return X_

    # write fit_transform that skips fit
    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


# this class will hot encode columnds
class CustomOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=False):
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first

    # fill in the rest below
    # assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'

    # define fit to do nothing but give warning
    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    # write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
    def transform(self, X):
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."

        if self.target_column not in X.columns:
            raise AssertionError(
                f"{self.__class__.__name__}.transform: Target column '{self.target_column}' not found in DataFrame."
            )

        # do actual OHE
        X_ = X.copy()
        X_ = pd.get_dummies(
            X_,
            columns=[self.target_column],
            dummy_na=self.dummy_na,
            drop_first=self.drop_first,
        )
        return X_

    # write fit_transform that skips fit
    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


# this class will map ordinal column
class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_column, mapping_dict: dict):
        assert isinstance(
            mapping_dict, dict
        ), f"{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead."
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column  # column to focus on

    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X):
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."
        assert (
            self.mapping_column in X.columns.to_list()
        ), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  # column legit?

        # Set up for producing warnings. First have to rework nan values to allow set operations to work.
        # In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
        # Strategy is to convert empty values to a string then the string back to np.nan
        placeholder = "NaN"
        column_values = (
            X[self.mapping_column].fillna(placeholder).tolist()
        )  # convert all nan values to the string "NaN" in new list
        column_values = [
            np.nan if v == placeholder else v for v in column_values
        ]  # now convert back to np.nan
        keys_values = self.mapping_dict.keys()

        column_set = set(
            column_values
        )  # without the conversion above, the set will fail to have np.nan values where they should be.
        keys_set = set(
            keys_values
        )  # this will have np.nan values where they should be so no conversion necessary.

        # now check to see if all keys are contained in column.
        keys_not_found = keys_set - column_set
        if keys_not_found:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n"
            )

        # now check to see if some keys are absent
        keys_absent = column_set - keys_set
        if keys_absent:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n"
            )

        # do actual mapping
        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


# a transformer for the pearson correlation coefficient
class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None
        self.is_fit = False  # check this is True before using transform

    # computes the correlated columns and stores it for the transform method later
    # returns self
    def fit(self, X, y=None):
        self.is_fit = True

        df_corr = X.corr(method="pearson")
        masked_df = np.abs(df_corr) > self.threshold
        upper_mask = np.triu(np.abs(df_corr) > self.threshold, k=1)
        self.correlated_columns = [
            (df_corr.columns[index])
            for index, col in enumerate(upper_mask.T)
            if np.any(col)
        ]
        return self

    # drops the correlated columns from fit method
    # should always be run after fit
    def transform(self, X):
        # make sure transformer is fitted
        assert (
            self.is_fit
        ), f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        # make sure we have a dataframe
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."

        # drop columns in self.correlated_columns
        X_ = X.copy()
        X_ = X_.drop(columns=self.correlated_columns)

        return X_

    # write fit_transform that does not skip fit
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result


# transformer for 3Sigma
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.low_bound = None
        self.high_bound = None
        self.is_fit = False  # check this is True before using transform

    # computes the 3Sigma percentages and stores them for the transform method later
    # returns a tuple of the lower and upper boundary
    def fit(self, X, y=None):
        self.is_fit = True

        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"expected Dataframe but got {type(X)} instead."
        assert self.target_column in X.columns, f"unknown column {self.target_column}"
        assert all(
            [isinstance(v, (int, float)) for v in X[self.target_column].to_list()]
        )

        # your code below
        col_mean = X[self.target_column].mean()  # average
        col_sigma = X[self.target_column].std()  # standard deviation

        # find boundaries for outliers
        self.low_bound = col_mean - 3 * col_sigma
        self.high_bound = col_mean + 3 * col_sigma

        return self.low_bound, self.high_bound

    # clips the outlier rows and resets the index
    # should always be run after fit
    def transform(self, X):
        # make sure transformer is fitted
        assert (
            self.is_fit
        ), f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        # make sure we have a dataframe
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."

        # clip the columns in self.target_column and reset index
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(
            lower=self.low_bound, upper=self.high_bound
        )

        X_ = X_.reset_index(drop=True)

        return X_

    # write fit_transform that does not skip fit
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result


# transformer for tukey calculation
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, fence="outer"):
        assert fence in ["inner", "outer"]
        self.target_column = target_column
        self.fence = fence
        self.is_fit = False  # check this is True before using transform
        self.low = None
        self.high = None

    # computes the tukey calculation and stores the result for the transform method later
    # returns tuple of lower and upper bound
    def fit(self, X, y=None):
        self.is_fit = True

        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"expected Dataframe but got {type(X)} instead."
        assert self.target_column in X.columns, f"unknown column {self.target_column}"
        assert all(
            [isinstance(v, (int, float)) for v in X[self.target_column].to_list()]
        )

        # your code below
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)

        if self.fence == "outer":
            iqr = q3 - q1  # inter-quartile range, where q1 is 25% and q3 is 75%
            outer_low = q1 - 3 * iqr  # factor of 2 larger
            outer_high = q3 + 3 * iqr

            self.low, self.high = outer_low, outer_high
        else:
            iqr = q3 - q1  # inter-quartile range, where q1 is 25% and q3 is 75%
            inner_low = q1 - 1.5 * iqr
            inner_high = q3 + 1.5 * iqr

            self.low, self.high = inner_low, inner_high

        return self.low, self.high

    # clips the outlier rows and resets the index
    # should always be run after fit
    def transform(self, X):
        # make sure transformer is fitted
        assert (
            self.is_fit
        ), f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        # make sure we have a dataframe
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."

        # clip the columns in self.target_column and reset index
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(
            lower=self.low, upper=self.high
        )

        X_ = X_.reset_index(drop=True)

        return X_

    # write fit_transform that does not skip fit
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result


# transformer for robust scaler
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        # fill in rest below
        self.column = column
        self.is_fit = False
        self.iqr = 0
        self.med = 0

    # computes the tukey calculation and stores the result for the transform method later
    # returns tuple of lower and upper bound
    def fit(self, X, y=None):
        self.is_fit = True

        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"expected Dataframe but got {type(X)} instead."
        assert self.column in X.columns, f"unknown column {self.column}"
        assert all([isinstance(v, (int, float)) for v in X[self.column].to_list()])

        # your code below
        self.iqr = X[self.column].quantile(0.75) - X[self.column].quantile(0.25)
        self.med = X[self.column].median()

    # clips the outlier rows and resets the index
    # should always be run after fit
    def transform(self, X):
        # make sure transformer is fitted
        assert (
            self.is_fit
        ), f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        # make sure we have a dataframe
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."

        # clip the columns in self.target_column and reset index
        X_ = X.copy()
        X_[self.column] -= self.med
        X_[self.column] /= self.iqr

        return X_

    # write fit_transform that does not skip fit
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result

# takes a dataframe and runs the variance code on it. 
# returns the value to use for the random state in the split method
def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)  #instantiate with k=5.
  vars = []  #collect test_error/train_error where error based on F1 score
  rs_val = 0

  # loop thru each random state (i)
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      vars.append(f1_ratio)

  rs_val = sum(vars)/len(vars)  #get average ratio value

  idx = np.array(abs(vars - rs_val)).argmin()  #find the index of the smallest value

  return idx 