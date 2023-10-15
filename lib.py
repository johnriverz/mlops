import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

sklearn.set_config(
    transform_output="pandas"
)  # says pass pandas tables through pipeline instead of numpy matrices

# # bring in tatanic data in trimmed form
# url = 'https://raw.githubusercontent.com/fickas/asynch_models/main/datasets/titanic_trimmed.csv'
# titanic_trimmed = pd.read_csv(url)
# titanic_features = titanic_trimmed.drop(columns='Survived') # drop the label column

# # an example pipeline
# titanic_transformer = Pipeline(steps=[
#     ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
#     ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
#     ('ohe_joined', CustomOHETransformer(target_column='Joined')),
#     ], verbose=True)

# transformed_df = titanic_transformer.fit_transform(titanic_features) # our transformed dataframe


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

        df_corr = transformed_df.corr(method="pearson")
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
        X_ = transformed_df.drop(columns=self.correlated_columns)

        return X_

    # write fit_transform that does not skip fit
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result
