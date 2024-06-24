EX_ORIG_CODE = """
# a6_group_on_title_and_id
df = df.groupby(['Title', 'Id']).agg(
    comments_now=('Id', 'count'),
    prev_contributions_mean=('PreviousContributions', 'mean'),
    prev_threads_mean=('PreviousThreads', 'mean'),
    cont_this_year=('ContributionsThisYear', 'mean'),
    threads_this_year=('ThreadsThisYear', 'mean')
).reset_index()

# a6_get_totalpreviousthreads
df["total_previous_threads"] = df["prev_threads_mean"]  + df["threads_this_year"]
# a6_get_totalpreviouscontributions
df["total_previous_contributions"] = df["prev_contributions_mean"]  + df["cont_this_year"]
# a6_get_contperthread
df["cont_per_thread"] = df["total_previous_contributions"] / df["total_previous_threads"]
"""

EX_CONVERTED_CODE = """
def t1(df: pd.DataFrame):
    df = df.groupby(['Title', 'Id']).agg(
        comments_now=('Id', 'count'),
        prev_contributions_mean=('PreviousContributions', 'mean'),
        prev_threads_mean=('PreviousThreads', 'mean'),
        cont_this_year=('ContributionsThisYear', 'mean'),
        threads_this_year=('ThreadsThisYear', 'mean')
    ).reset_index()
    return TransformDataReturn(
            df=df, 
            groupby_cols=frozenset(['Title', 'Id']),
            column_mapping={
                    frozenset(['PreviousContributions']: 'prev_contributions_mean',
                    frozenset(['PreviousThreads']): 'prev_threads_mean',
                    frozenset(['ContributionsThisYear']): 'cont_this_year',
                    frozenset(['ThreadsThisYear']): 'threads_this_year'},
                    },
            transform_verb='derive'
        )
def t2(df: pd.DataFrame):
    df["total_previous_threads"] = df["prev_threads_mean"]  + df["threads_this_year"]
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['prev_threads_mean', 'threads_this_year']): 'total_previous_threads'},
            transform_verb='derive'
        )
def t3(df: pd.DataFrame):
    df["total_previous_contributions"] = df["prev_contributions_mean"]  + df["cont_this_year"]
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['prev_contributions_mean', 'cont_this_year']): 'total_previous_contributions'},
            transform_verb='derive'
        )
def t4(df: pd.DataFrame):
    df["cont_per_thread"] = df["total_previous_contributions"] / df["total_previous_threads"]
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['total_previous_contributions', 'total_previous_threads']): 'cont_per_thread'},
            transform_verb='derive'
        )
        
transform_funcs = [t1, t2, t3, t4]
"""


EX_ORIG_CODE_2 = """
df = df[(df['Type'] == 2) & (df['DebateSize'] > 1)].sort_values(by=['ThreadId', 'Order'])
df['live_and_not_live'] = df.groupby('ThreadId')['Live'].transform(lambda x: len(x) > 1)
df = df.astype({'live_and_not_live': int})
"""

EX_CONVERTED_CODE_2 = """
def t1(df: pd.DataFrame):
    df = df[(df['Type'] == 2) & (df['DebateSize'] > 1)]
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['Type', 'DebateSize']): 'ALL'},
            transform_verb='filter'
        )
def t2(df: pd.DataFrame):
    df = df.sort_values(by=['ThreadId', 'Order'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['ThreadId', 'Order']): 'ALL'},
            transform_verb='orderby'
        )

def t3(df: pd.DataFrame):
    df['live_and_not_live'] = df.groupby('ThreadId')['Live'].transform(lambda x: len(x) > 1)
    return TransformDataReturn(
            df=df, 
            groupby_cols=frozenset(['ThreadId']),
            column_mapping={frozenset(['Live']): 'live_and_not_live'},
            transform_verb='groupby'
        )

def t4(df: pd.DataFrame):
    df = df.astype({'live_and_not_live': int})
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['live_and_not_live']): 'live_and_not_live'},
            transform_verb='derive'
        )
        
transform_funcs = [t1, t2, t3, t4]
"""

EX_ORIG_CODE_3 = """
df = df.dropna(subset=['redCards', 'rater1', 'rater2', 'meanIAT', 'meanExp'])

# create skin tone variable
# average skin rating from two raters
ratings = (df.rater1 + df.rater2) / 2
df['skin_tone'] = pd.cut(ratings, bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=['Very Light', 'Light', 'Neutral', 'Dark', 'Very Dark'])
"""

EX_CONVERTED_CODE_3 = """
def t1(df: pd.DataFrame):
    df = df.dropna(subset=['redCards', 'rater1', 'rater2', 'meanIAT', 'meanExp'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['redCards', 'rater1', 'rater2', 'meanIAT', 'meanExp']): 'ALL'},
            transform_verb='filter'
        )
def t2(df: pd.DataFrame):
    # we include ratings in the dataframe as we want to capture this unit calculation/function 
    df['ratings'] = (df.rater1 + df.rater2) / 2
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['rater1', 'rater2']): 'ratings'},
            transform_verb='derive'
        )
def t3(df: pd.DataFrame):
    df['skin_tone'] = pd.cut(df['ratings'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=['Very Light', 'Light', 'Neutral', 'Dark', 'Very Dark'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['ratings']): 'skin_tone'},
            transform_verb='derive'
        )
transform_funcs = [t1, t2, t3]
"""

EX_ORIG_CODE_4 = """
df['skin_tone'] = (df.rater1 + df.rater2) / 2
    
df['skin_tone'] = df['skin_tone'].replace({
    0.0: 'Very Light Skin',
    0.25: 'Light Skin',
    0.5: 'Neutral Skin',
    0.75: 'Dark Skin',
    1.0: 'Very Dark Skin'
})

model_df = df[['skin_tone', 'redCards', 'meanIAT', 'meanExp']]
model_df = model_df.dropna()
"""

EX_CONVERTED_CODE_4 = """
def t1(df: pd.DataFrame):
    df['skin_tone'] = (df.rater1 + df.rater2) / 2
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['rater1', 'rater2']): 'skin_tone'},
            transform_verb='derive'
        )

def t2(df: pd.DataFrame):
    df['skin_tone'] = df['skin_tone'].replace({
        0.0: 'Very Light Skin',
        0.25: 'Light Skin',
        0.5: 'Neutral Skin',
        0.75: 'Dark Skin',
        1.0: 'Very Dark Skin'
    })
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['skin_tone']): 'skin_tone'},
            transform_verb='derive'
        )

def t3(df: pd.DataFrame):
    model_df = df[['skin_tone', 'redCards', 'meanIAT', 'meanExp']] # this is not really a unique transform, just selecting columns
    model_df = model_df.dropna()
    return TransformDataReturn(
            df=model_df, 
            column_mapping={frozenset(['skin_tone', 'redCards', 'meanIAT', 'meanExp']): 'ALL'},
            transform_verb='filter'
        )
transform_funcs = [t1, t2, t3]
"""

EX_ORIG_CODE_5 = """
df = pd.get_dummies(df, columns=['rater1_cat', 'rater2_cat', 'refCountry_cat'])

# drop unused columns
df = df.drop(['rater1', 'rater2', 'refCountry'], axis=1)
"""

EX_CONVERTED_CODE_5 = """
def t1(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['rater1_cat'])
    new_cols = [col for col in df.columns if 'rater1_cat' in col]
    column_mapping = {for col in new_cols: frozenset(['rater1_cat']): col}
    return TransformDataReturn(
            df=df, 
            column_mapping=column_mapping,
            transform_verb='derive'
        )
        
def t2(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['rater2_cat'])
    new_cols = [col for col in df.columns if 'rater2_cat' in col]
    column_mapping = {for col in new_cols: frozenset(['rater2_cat']): col}
    return TransformDataReturn(
            df=df, 
            column_mapping=column_mapping,
            transform_verb='derive'
        )
        
def t3(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['refCountry_cat'])
    new_cols = [col for col in df.columns if 'refCountry_cat' in col]
    column_mapping = {for col in new_cols: frozenset(['refCountry_cat']): col}
    # this code selects a subset of columns but does not impact any columns that we are outputting so we do not need to put it in it's own function
    df = df.drop(['rater1', 'rater2', 'refCountry'], axis=1) 
    return TransformDataReturn(
            df=df, 
            column_mapping=column_mapping,
            transform_verb='derive'
        )
"""

EX_ORIG_CODE_6 = """
# Total amount spent by each customer
total_spent = (
    transactions_df.groupby("customerID")["transactionAmount"].sum().reset_index()
)
total_spent.columns = ["customerID", "total_spent"]

# Merging with the transactions dataframe to include additional customer details
customer_summary = pd.merge(
    total_spent,
    transactions_df[
        ["customerID", "customerCategory", "averageOrderValue", "loyaltyPoints"]
    ],
    on="customerID",
    how="inner",
)

# Display the result
print(customer_summary)
"""

EX_CONVERTED_CODE_6 = """
def t1(df: pd.DataFrame):
    # this is the same as creating a calculating total_spent and merging with the transactions dataframe
    df["transactionAmount"] = df.groupby("customerID")[ "transactionAmount"].transform("sum")
    customer_summary = df.sort_values("customerID") 
    return TransformDataReturn(
            df=customer_summary, 
            groupby_cols=frozenset(['customerID']),
            column_mapping={frozenset(['transactionAmount']): 'transactionAmount'},
            transform_verb='groupby'
        )
"""

EX_ORIG_CODE_7 = """
# Filtering and preprocessing data
relevant_cols = ['studentID', 'testScores', 'meanGPA', 'rating1', 'rating2']
df = df[relevant_cols].copy()
# Remove rows with missing values in 'rating1' and 'rating2'
df = df.dropna(subset=['rating1', 'rating2'])
"""

EX_CONVERTED_CODE_7 = """
def t1(df: pd.DataFrame):
    # This step selects a subset of columns but does not impact any columns that we are outputting, so we do not need to put it in its own function.
    # We combine it with the next filtering step.
    relevant_cols = ['studentID', 'testScores', 'meanGPA', 'rating1', 'rating2']
    df = df[relevant_cols].copy()
    df = df.dropna(subset=['rating1', 'rating2'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['rating1', 'rating2']): 'ALL'},
            transform_verb='filter'
        )
"""
