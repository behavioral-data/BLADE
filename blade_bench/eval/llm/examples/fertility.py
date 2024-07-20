from blade_bench.data.dataset import load_dataset_info
from blade_bench.eval.datamodel import (
    AgentCVarsWithCol,
    ControlVarWithCol,
    DVarWithCol,
    EntireAnalysis,
    IVarWithCol,
)

FERTILITY_DINFO = load_dataset_info("fertility")


FERTILITY_TRANSFORM_CODE = """# drop the rows with missing values in the ReportedCycleLength, Rel1, Rel2, and Rel3 columns
df = df.dropna(subset=['ReportedCycleLength'])
df = df.dropna(subset=['Rel1', 'Rel2', 'Rel3'])

df['AvgReligiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)


# Convert date columns to datetime format
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate the cycle length based on provided dates
df['ReportedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days
# Calculate the expected start date of the next menstrual period

df['ExpectedNextPeriod'] = df['StartDateofLastPeriod'] + pd.to_timedelta(df['ReportedCycleLength'], unit='d')

# Calculate the day of ovulation by subtracting 14 days from the expected start date of the next period
# since ovulation typically occurs around 14 days before the start of the next period
df['OvulationDate'] = df['ExpectedNextPeriod'] - pd.to_timedelta(14, unit='d')
# Calculate the cycle day on the date of testing
df['CycleDay'] = (df['DateTesting'] - df['OvulationDate']).dt.days + 14
# Define high-fertility (cycle days 6-14) and low-fertility (cycle days 17-27) groups
df['FertilityGroup'] = df['CycleDay'].apply(lambda x: 'High-Fertility' if 6 <= x <= 14 else ('Low-Fertility' if 17 <= x <= 27 else 'Other'))
# Filter out the 'Other' group to focus on the high and low fertility groups
df = df[df['FertilityGroup'].isin(['High-Fertility', 'Low-Fertility'])]


df['IsIncommittedRelationship'] = df['Relationship'].apply(lambda x: 0 if x in [1,2] else 1)
df['InRelationship'] = df['Relationship'].apply(lambda x: 0 if x == 1 else 1)
df['IsInRelationship'] = df['Relationship'].apply(lambda x: 0 if x == 1 else 1)
"""


FERTILITY_MODEL_CODE = """model = smf.ols('AvgReligiosity ~ InRelationship * FertilityGroup', data=df).fit()
# Display the regression results
print(model.summary())
"""

FERTILITY_CVARS = AgentCVarsWithCol(
    ivs=[
        IVarWithCol(
            description="Women's fertility",
            columns=["FertilityGroup"],
        )
    ],
    controls=[
        ControlVarWithCol(
            description="Relationship status of the individual",
            is_moderator=True,
            moderator_on="Women's fertility",
            columns=["InRelationship"],
        )
    ],
    dv=DVarWithCol(
        description="Women's religiosity",
        columns=["AvgReligiosity"],
    ),
)

FERTILITY_ANALYSIS = EntireAnalysis(
    cvars=FERTILITY_CVARS,
    transform_code=FERTILITY_TRANSFORM_CODE,
    m_code=FERTILITY_MODEL_CODE,
)


FERTILITY_VARIABLES_A = {
    "1": "IV: fertility",
    "2": "IV: Answer to 'How many days long are your menstrual cycles?",
    "3": "IV: Reported Cycle Length",
    "4": "DV: Women's religiosity",
    "5": "DV: Combination of three religious scores",
    "6": "Control: Relationship status of the individual",
    "7": "Control: Is in committed Relationship.",
    "8": "Control: In reference to `StartDateofPeriodBeforeLast`, 'How sure are you about that date?'.",
}

FERTILITY_VARIABLES_B = {
    "1": "Women's Reported Cycle Length",
    "2": "Relationship status of the individual",
    "3": "Women's religiosity",
    "4": "Women's fertility",
}

FERTILITY_CVAR_SIMILARITY_RESULT = {
    "A1-B4": {
        "rationale": "Both variables are speaking about women's fertility although the first one is a bit more ambiguous.",
        "similarity": 9,
    },
    "A2-B1": {
        "rationale": "Both variables are speaking about the length of menstrual cycles.",
        "similarity": 10,
    },
    "A3-B1": {
        "rationale": "Both variables are speaking about the length of menstrual cycles.",
        "similarity": 10,
    },
    "A4-B3": {
        "rationale": "Both variables are speaking about how religious a woman in the study is.",
        "similarity": 10,
    },
    "A5-B3": {
        "rationale": "Both variables are speaking about how religious a women is except the first one goes into more about how to calculate it.",
        "similarity": 8,
    },
    "A6-B2": {
        "rationale": "Both variables are speaking about the relationship status of the individual.",
        "similarity": 10,
    },
    "A7-B2": {
        "rationale": "Both variables are speaking about the relationship status of the individual.",
        "similarity": 10,
    },
}


FERTILITY_MODELS_A = {
    "1": "Linear regression",
    "2": "logistic regression",
    "3": "Two sample t-test without considering relationship",
    "4": "Linear regression with interaction effect",
    "5": "Linear regression with interaction effect -- different binary feature for relationship",
    "6": "Ordinary Least Squares (OLS) Regression",
    "7": "Linear Mixed Model",
    "8": "Basic Linear Regression Model",
    "9": "Linear Regression with Interaction Term",
    "10": "Mixed Effects Model with Polynomial Terms",
}


FERTILITY_MODELS_B = {
    "1": "Linear regression",
    "2": "Logistic regression",
    "3": "Two sample t-test",
    "4": "Two-way ANOVA",
    "5": "Linar Mixed-effects model",
    "6": "Polynomial Mixed-effects model",
    "7": "Polynomial Regression Model",
}


FERTILITY_MODELS_SIMILARITY_RESULT = {
    "A1-B1": {
        "rationale": "Both models are linear regression models.",
    },
    "A2-B2": {
        "rationale": "Both models are logistic regression models.",
    },
    "A3-B3": {
        "rationale": "Both models are t-tests.",
    },
    "A4-B1": {
        "rationale": "Both models are linear regression models",
    },
    "A5-B1": {
        "rationale": "Both models are linear regression models",
    },
    "A6-B1": {
        "rationale": "Both models are linear regression models. OLS is the same as a linear regression model.",
    },
    "A7-B5": {
        "rationale": "Both models are linear mixed effects models",
    },
    "A8-B1": {
        "rationale": "Both models are linear regression models",
    },
    "A9-B1": {
        "rationale": "Both models are linear regression models",
    },
    "A10-B6": {
        "rationale": "Both models are polynomial mixed effects models",
    },
}


if __name__ == "__main__":
    print(FERTILITY_ANALYSIS.model_dump(indent=2))
    print("Here")
