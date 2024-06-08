import pandas as pd
-----Creating_Reading_Writing
pd.DataFrame(dict('column_name': list_of_columns),index = list_of_row_names <- default is range(n)) #creates a table
pd.DataFrame(data = matrix_of_cells, columns = list_of_column_names) #creates a table
pd.Series(list_column,index = [optional], name = 'optional') # single column of a DataFrame
dataset = pd.read_csv("file") #csv to DataFrame 
dataset.to_csv('file') #DataFrame to csv
dataset.shape # (num of rows, num of columns)
dataset.head() # grabs top 5 rows
---Indexing-Selecting-Assigning------
dataset.column_name or
dataset['column_name'] #same obvious thing (including indices so 2 columns)
dataset['column_name'][i] # like in a matrix
dataset.iloc[0] #first row (with indices again)
dataset.iloc[:,0] # first column (indexed) or
dataset.iloc[range_of_rows, n]
dataset.loc[:, 'column_name'] or
dataset.loc[:, ['columns']] #refers to elts by the content (: is inclusive from the right!)
dataset.set_index('column') # assign index column(leftmost one)
dataset.column_name == 'name' #indexed column of True/False
dataset.loc[dataset.column_name == 'name'] #returns columns with column_name being 'name'
dataset.loc[dataset.column_name.isin(some_list)] #also  column_name.isnull() <- empty or .notnull()
dataset['column_name'] = 'constant' # makes a const column
dataset['column_name'] = iterable
-----Summary-Functions-Maps----------
dataset.column.describe() # prints ?????
dataset.column.mean() #takes mean
dataset.column.unique() #unique values
dataset.column.value_counts() #above with number of occurences
dataset.column.map(lambda p: f(p)) #changes value p to f(p); return no modify
dataset.apply(lambda, axis = 'columns') #applies lambda to each row; or axis = 'index' to each column; returns
dataset.column_1 + 'str' + dataset.column_2 #combines
dataset.column.idxmax() #returns index with max value
----Grouping-Sorting----------
dataset.groupby('column_name1').column_name2.min() #returns Series -> name1 and min value of name2 in each subgroup of name1
dataset.groupby('column1').apply(lambda f: f(x)) # returns Series with index comes from column1. The input of f is a dataframe of all rows having a given column1 value
dataset.groupby.(['column1','column2']).apply(lambda f: f(x)) #creates multindexed table, i.e. (n+1)-dim matrix
dataset.groupby.(['column1']).column2.agg([f1,f2]) #returns dataframe f1(column2),f2(column2) indexed by column1
dataset.reset_index() #makes index sequential. If we have multiindexed df it makes it singleindexed.
dataset.sort_values(by = 'column1', (default) ascending = True) # sorts rows wrt column1
dataset.sort_values(by = ['col1', 'col2']) #sorts wrt 2 columns as above
dataset.sort_index() # sorts the index column
--------------
dataset[pd.isnull(dataset.column1)] # returns subdataframe with rows not containing column1 value (NaN)
dataset[pd.notnull(dataset.column1)] # the complement of the above
dataset.column1.fillna("string") # fills Nan in column1 with string
dataset.column1.replace('string1', 'string2') #replaces string1 with string2 in the given column
------Renaming-Combining-------
dataset.rename(columns = {'oldname':'newname'}) #names of columns
dataset.rename(index = {0:'new_zero', 1 : 'new_one'}) #names of indices
dataset.rename_axis('name1', axis = 'rows' or 'columns')
pd.concat([dataset1, dataset2]) # concatenation of rows (number of columns are fixed)
dataset1.join(dataset2, lsuffix='x', rsuffix = 'y') # concatenates columns (number of indices are fixed)
