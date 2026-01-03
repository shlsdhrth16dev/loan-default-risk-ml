# feature_config.py

NUMERIC_FEATURES = [
    "age",
    "income",
    "loanamount",
    "creditscore",
    "monthsemployed",
    "numcreditlines",
    "interestrate",
    "loanterm",
    "dtiratio",
]

CATEGORICAL_FEATURES = [
    "education",
    "employmenttype",
    "maritalstatus",
    "hasmortgage",
    "hasdependents",
    "loanpurpose",
    "hascosigner",
]

TARGET_COL = "default"
