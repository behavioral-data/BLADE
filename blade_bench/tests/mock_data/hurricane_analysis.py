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
