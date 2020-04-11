import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pickle

def get_column_indicies(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]

def get_categorical_columns(df, subset_cols=None, ignore_cols=None, threshold=None):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if threshold: cat_cols += [x for x in df.select_dtypes(include=[np.number]).columns if df[x].nunique() < threshold]
    if subset_cols is not None: cat_cols = [x for x in cat_cols if x in subset_cols]
    if ignore_cols is not None: cat_cols = [x for x in cat_cols if x not in ignore_cols]
    return cat_cols

def get_categorical_column_indexes(df, subset_cols=None, ignore_cols=None, threshold=None):
    return get_column_indicies(df, get_categorical_columns(df, subset_cols=subset_cols, ignore_cols=ignore_cols, threshold=threshold))

def get_numerical_columns(df, subset_cols=None, ignore_cols=None, threshold=None):
    num_cols = [x for x in df.select_dtypes(include=[np.number]).columns if not threshold or f[x].nunique() >= threshold]
    if subset_cols is not None and type(subset_cols) == list: num_cols = [x for x in num_cols if x in subset_cols]
    if ignore_cols is not None and type(ignore_cols) == list: num_cols = [x for x in num_cols if x not in ignore_cols]
    return num_cols

def get_numerical_column_indexes(df, subset_cols=None, ignore_cols=None, threshold=None):
    return get_column_indicies(df, get_numerical_columns(df, subset_cols=subset_cols, ignore_cols=ignore_cols, threshold=threshold))


class Encoding(BaseEstimator):

    def __init__(self,  categorical_columns=None, return_df=False, new_value_map=0, **kwargs):
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.new_value_map = new_value_map

    def convert_input(self, X): return pd.DataFrame(X).copy(deep=True)

    def create_encoding_dict(self, X, y): return {}

    def apply_encoding(self, X, encoding_dict):
        for col in self.categorical_columns:
            if col in encoding_dict: X[col] = X[col].apply(lambda x: encoding_dict[col].get(x, 0))
        return X

    def fit(self, X, y=None):
        assert X is not None, "Input array is required to call fit method!"
        self.encoding_dict = self.create_encoding_dict(self.convert_input(X), y)
        return self

    def transform(self, X, return_df=True):
        assert hasattr(self, "encoding_dict"), "Fit call is required before transform"
        df = self.apply_encoding(self.convert_input(X), self.encoding_dict)
        return df if self.return_df else df.values

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def save_encoding(self, file_name):
        assert hasattr(self, "encoding_dict"), "Fit call is required before this"
        assert file_name is not None, "file_name cannot be None"
        with open(file_name, "wb") as out_file:
            pickle.dump(self.encoding_dict, out_file)

    def load_encoding(self, file_name):
        assert file_name is not None, "file_name cannot be None"
        self.encoding_dict = pickle.load(open(file_name, "rb"))

    def get_categorical_columns(self, df, categorical_columns=None):
        return categorical_columns if categorical_columns else get_categorical_columns(df)



class LabelEncoding(Encoding):

    def __init__(self, categorical_columns=None, return_df=False, new_value_map=0, **kwargs):
        super(LabelEncoding, self).__init__(categorical_columns=categorical_columns, return_df=return_df, new_value_map=new_value_map)

    def create_encoding_dict(self, X, y):
        return {col: dict(zip(X[col].unique(), range(1, X[col].nunique()+1))) for col in self.get_categorical_columns(X, self.categorical_columns)}



class FreqeuncyEncoding(Encoding):

    def __init__(self, normalize=1, categorical_columns=None, return_df=False, new_value_map=0, **kwargs):
        super(FreqeuncyEncoding, self).__init__(categorical_columns=categorical_columns, return_df=return_df, new_value_map=new_value_map)
        self.normalize = normalize

    def create_encoding_dict(self, X, y):
        return {col: X[col].value_counts(self.normalize).to_dict() for col in self.get_categorical_columns(X, self.categorical_columns)}
