HURRICANE_ANALYSIS = {
    "cvars": {
        "ivs": [
            {"description": "Femininity of the hurricane name", "columns": ["masfem"]}
        ],
        "dv": {
            "description": "Number of deaths caused by the hurricane",
            "columns": ["alldeaths"],
        },
        "controls": [
            {
                "description": "Category of the hurricane on the Saffir-Simpson scale",
                "is_moderator": False,
                "moderator_on": None,
                "columns": ["category"],
            },
            {
                "description": "Maximum wind speed of the hurricane at the time of landfall",
                "is_moderator": False,
                "moderator_on": None,
                "columns": ["wind"],
            },
            {
                "description": "Minimum pressure of the hurricane at the time of landfall",
                "is_moderator": False,
                "moderator_on": None,
                "columns": ["min"],
            },
        ],
    },
    "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min'])\n    \n    # Ensure relevant columns are in the correct type\n    df['masfem'] = df['masfem'].astype(float)\n    df['alldeaths'] = df['alldeaths'].astype(int)\n    df['category'] = df['category'].astype(int)\n    df['wind'] = df['wind'].astype(float)\n    df['min'] = df['min'].astype(float)\n    \n    return df",
    "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Fit the model\n    X = df[['masfem', 'category', 'wind', 'min']]\n    y = df['alldeaths']\n    X = sm.add_constant(X)  # Adds a constant term for the intercept\n    \n    model = sm.OLS(y, X).fit()\n    \n    return model.summary()",
}


HURRICANE_ANALYSES_SUBMISSION = {
    "dataset_name": "hurricane",
    "analyses": [
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of hurricane names",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall in the United States",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                    {
                        "description": "Maximum wind speed at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in columns needed\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'min', 'wind'])\n    \n    # Normalize femininity score to a 0-1 range\n    df['masfem'] = (df['masfem'] - df['masfem'].min()) / (df['masfem'].max() - df['masfem'].min())\n    \n    return df",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Define the independent variables (IVs) and control variables\n    ivs = ['masfem']\n    controls = ['category', 'min', 'wind']\n    \n    # Construct the formula for the OLS regression model\n    formula = 'alldeaths ~ ' + ' + '.join(ivs + controls)\n    \n    # Fit the OLS regression model\n    model = sm.OLS.from_formula(formula, data=df).fit()\n    \n    # Display the regression results\n    return model.summary()",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of the hurricane name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min'])\n    \n    # Ensure relevant columns are in the correct type\n    df['masfem'] = df['masfem'].astype(float)\n    df['alldeaths'] = df['alldeaths'].astype(int)\n    df['category'] = df['category'].astype(int)\n    df['wind'] = df['wind'].astype(float)\n    df['min'] = df['min'].astype(float)\n    \n    return df",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Fit the model\n    X = df[['masfem', 'category', 'wind', 'min']]\n    y = df['alldeaths']\n    X = sm.add_constant(X)  # Adds a constant term for the intercept\n    \n    model = sm.OLS(y, X).fit()\n    \n    return model.summary()",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "The femininity of the hurricane's name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min'])\n    return df\n",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Defining the independent variables (including controls) and the dependent variable\n    X = df[['masfem', 'category', 'wind', 'min']]\n    y = df['alldeaths']\n    \n    # Adding a constant term to the independent variables\n    X = sm.add_constant(X)\n    \n    # Fitting the OLS regression model\n    model = sm.OLS(y, X).fit()\n    \n    # Returning the model summary\n    return model.summary()\n",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Masculinity-Femininity index of the hurricane name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Total number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                    {
                        "description": "Number of years elapsed since the hurricane",
                        "is_moderator": False,
                        "columns": ["elapsedyrs"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Filtering out necessary columns\n    df = df[['masfem', 'alldeaths', 'category', 'min', 'wind', 'elapsedyrs']]\n    \n    # Handling missing values by dropping rows with any missing value\n    df = df.dropna()\n    \n    return df",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Define the independent variables and add a constant term for the intercept\n    X = df[['masfem', 'category', 'min', 'wind', 'elapsedyrs']]\n    X = sm.add_constant(X)\n    \n    # Define the dependent variable\n    y = df['alldeaths']\n    \n    # Fit the OLS regression model\n    model = sm.OLS(y, X).fit()\n    \n    # Return the summary of the model\n    return model.summary()",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Masculinity-Femininity rating of hurricane names",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                    {
                        "description": "Normalized property damage caused by hurricanes, adjusted to 2015 monetary values",
                        "is_moderator": False,
                        "columns": ["ndam15"],
                    },
                    {
                        "description": "Elapsed years since the hurricane",
                        "is_moderator": False,
                        "columns": ["elapsedyrs"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in key columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min', 'ndam15', 'elapsedyrs'])\n    return df\n",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Define the independent variables and the dependent variable\n    X = df[['masfem', 'category', 'wind', 'min', 'ndam15', 'elapsedyrs']]\n    y = df['alldeaths']\n    \n    # Add a constant to the independent variables matrix\n    X = sm.add_constant(X)\n    \n    # Fit an OLS regression model\n    model = sm.OLS(y, X).fit()\n    \n    # Display the regression results\n    return model.summary()\n",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of hurricane names",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["wind"],
                    },
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["category"],
                    },
                    {
                        "description": "Normalized property damage caused by hurricanes to 2015 monetary values",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["ndam15"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in the columns masfem, alldeaths, wind, category, and ndam15\n    df = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'category', 'ndam15'])\n    \n    return df\n",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    import statsmodels.api as sm\n    X = df[['masfem', 'wind', 'category', 'ndam15']]\n    X = sm.add_constant(X)  # Adds a constant term to the predictor\n    y = df['alldeaths']\n    \n    model = sm.OLS(y, X).fit()\n    results = model.summary()\n    return results\n",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "The femininity of the hurricane's name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Total number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at landfall",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["min"],
                    },
                    {
                        "description": "Number of years elapsed since the hurricane",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["elapsedyrs"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Ensure no missing values in the relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min', 'elapsedyrs'])\n    \n    # Normalize the 'masfem' column to a range of 0 to 1 (optional, depending on interpretability preference)\n    df['masfem_norm'] = (df['masfem'] - df['masfem'].min()) / (df['masfem'].max() - df['masfem'].min())\n    \n    # Return the transformed dataframe\n    return df",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Define the dependent variable (y) and the independent variables (X)\n    y = df['alldeaths']\n    X = df[['masfem_norm', 'category', 'wind', 'min', 'elapsedyrs']]\n    \n    # Add a constant to the model (intercept)\n    X = sm.add_constant(X)\n    \n    # Fit the regression model\n    model = sm.OLS(y, X).fit()\n    \n    # Return the model results\n    return model.summary()",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of hurricane name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["min"],
                    },
                    {
                        "description": "Elapsed years since the hurricane",
                        "is_moderator": False,
                        "columns": ["elapsedyrs"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop any rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min', 'elapsedyrs'])\n\n    # Ensure that the data types are correct\n    df['year'] = df['year'].astype(int)\n    df['category'] = df['category'].astype(int)\n    df['wind'] = df['wind'].astype(float)\n    df['min'] = df['min'].astype(float)\n    df['elapsedyrs'] = df['elapsedyrs'].astype(float)\n    df['masfem'] = df['masfem'].astype(float)\n    df['alldeaths'] = df['alldeaths'].astype(int)\n\n    return df",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Fit the statistical model\n    X = df[['masfem', 'category', 'wind', 'min', 'elapsedyrs']]\n    y = df['alldeaths']\n    X = sm.add_constant(X)  # Adds constant term for the intercept\n    model = sm.OLS(y, X).fit()\n    # Display the regression results\n    print(model.summary())\n    return model",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of the hurricane name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["wind"],
                    },
                    {
                        "description": "Minimum pressure of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "moderator_on": "",
                        "columns": ["min"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind', 'min'])\n\n    # Ensure column types\n    df['masfem'] = df['masfem'].astype(float)\n    df['alldeaths'] = df['alldeaths'].astype(int)\n    df['category'] = df['category'].astype(int)\n    df['wind'] = df['wind'].astype(float)\n    df['min'] = df['min'].astype(float)\n\n    return df\n",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    X = df[['masfem', 'category', 'wind', 'min']]\n    y = df['alldeaths']\n    \n    # Add constant to the model (intercept)\n    X = sm.add_constant(X)\n    \n    # Fit the model\n    model = sm.OLS(y, X).fit()\n    \n    # Return the summary of the model\n    return model.summary()\n",
        },
        {
            "cvars": {
                "ivs": [
                    {
                        "description": "Femininity of the hurricane's name",
                        "columns": ["masfem"],
                    }
                ],
                "dv": {
                    "description": "Number of deaths caused by the hurricane",
                    "columns": ["alldeaths"],
                },
                "controls": [
                    {
                        "description": "Category of the hurricane on the Saffir-Simpson scale",
                        "is_moderator": False,
                        "columns": ["category"],
                    },
                    {
                        "description": "Maximum wind speed of the hurricane at the time of landfall",
                        "is_moderator": False,
                        "columns": ["wind"],
                    },
                ],
            },
            "transform_code": "def transform(df: pd.DataFrame) -> pd.DataFrame:\n    # Drop rows with missing values in relevant columns\n    df = df.dropna(subset=['masfem', 'alldeaths', 'category', 'wind'])\n\n    # Ensure correct data types\n    df['masfem'] = df['masfem'].astype(float)\n    df['alldeaths'] = df['alldeaths'].astype(int)\n    df['category'] = df['category'].astype(int)\n    df['wind'] = df['wind'].astype(float)\n\n    return df\n",
            "m_code": "def model(df: pd.DataFrame) -> Any:\n    # Define the independent variables (with controls) and the dependent variable\n    X = df[['masfem', 'category', 'wind']]\n    y = df['alldeaths']\n\n    # Add a constant to the independent variables matrix\n    X = sm.add_constant(X)\n\n    # Fit the regression model\n    model = sm.OLS(y, X).fit()\n\n    return model.summary()\n",
        },
    ],
}
