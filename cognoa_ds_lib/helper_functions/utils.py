import pandas as pd
import numpy as np


# handy function to return list of DataFrame columns that have a keyword (like ADIR or ADOS) somewhere in their title
def columns_about(df, keyword):
    return [x for x in list(df.columns) if keyword.lower() in x.lower()]

# handy function to replace certain values in certain columns of a dataframe. Useful for feature value mapping before training
def replace_values_in_dataframe_columns(df, columns, values, replacement, replace_if_equal=True):
    for column in columns:

        if (replace_if_equal):
            mask = df[column].isin(values)
        else:
            mask = np.logical_not(df[column].isin(values))

        df[column][mask] = replacement

# handy function to subsample dataframe by choosing x% of the samples of each 'class' as defined by a column in the df
def subsample_per_class(df, class_column_name, dict_ratio_per_class):
    output_df = pd.DataFrame()
    for class_name in dict_ratio_per_class.keys():
        df_this_class = df[df[class_column_name] == class_name]
        ratio = dict_ratio_per_class[class_name]
        total = len(df_this_class)
        sample_size = int(float(total) * ratio)
        subset = df_this_class.loc[np.random.choice(df_this_class.index, sample_size, replace=False)]
        subset = subset.reset_index()
        output_df = pd.concat([output_df, subset])
    return output_df.reset_index(drop=True)


