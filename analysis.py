import os

import bambi
import numpy as np
import pandas as pd

# Import data.
cntry = pd.read_stata(os.path.join("data", "bradyfinnigan2014countrydata.dta"), convert_categoricals=False)
print(cntry.columns.tolist())

ZA2900 = pd.read_stata(os.path.join("data", "ZA2900.dta"), convert_categoricals=False)
print(ZA2900.shape)

ZA4700 = pd.read_stata(os.path.join("data", "ZA4700.dta"), convert_categoricals=False)
print(ZA4700.shape)

ZA2900_country_codes = {
    "Australia": 1,
    "Canada": 20,
    # "Denmark": 0,      # Not present in 1996 data.
    # "Finland": 0,      # Not present in 1996 data.
    "France": 27,
    "Germany": [2, 3],
    "Ireland": 10,
    "Japan": 24,
    # "Netherlands": 0,  # Not present in 1996 data.
    "New Zealand": 19,
    "Norway": 12,
    # "Portugal": 0,     # Not present in 1996 data.
    "Spain": 25,
    "Sweden": 13,
    "Switzerland": 30,
    "United Kingdom": [4, 5],
    "United States": 6,
}

ZA4700_country_codes = {
    "Australia": 36,
    "Canada": 124,
    "Denmark": 208,
    "Finland": 246,
    "France": 250,
    "Germany": [276.1, 276.2],
    "Ireland": 372,
    "Japan": 392,
    "Netherlands": 528,
    "New Zealand": 554,
    "Norway": 578,
    "Portugal": 620,
    "Spain": 724,
    "Sweden": 752,
    "Switzerland": 756,
    "United Kingdom": [826.1],
    "United States": 840,
}

country2country = {
    1: 36,
    20: 124,
    27: 250,
    2: 276.1,
    3: 276.1,
    10: 372,
    24: 392,
    19: 554,
    12: 578,
    25: 724,
    13: 752,
    30: 756,
    4: 826.1,
    6: 840,
}

y2y = {
    2900: 1996,
    4700: 2006,
}

ZA2900_DEPS = {
    "jobs": "V36",
    "unemployment": "V41",
    "income": "V42",
    "retirement": "V39",
    "housing": "V44",
    "healthcare": "V38",
}

ZA4700_DEPS = {
    "jobs": "V25",
    "unemployment": "V30",
    "income": "V31",
    "retirement": "V28",
    "housing": "V33",
    "healthcare": "V27",
}

INDS = [
    "foreignpct",
    "netmigpct",
]

CONTROLS = {
    "socx"
}

ZA4700_CONTROLS = {

}

# Preprocess ZA2900.
# print(ZA2900['v3'])

# Listwise deletion.
# ZA4700 = ZA4700.dropna(how="any")
# ZA2900 = ZA2900.dropna(how="any")

# Preprocess ZA4700

# Combine the two datasets.
data = pd.DataFrame()
data['country'] = ZA2900['v3']
data['year'] = ZA2900['v1']

data["deps_jobs"] = ZA2900['v36']
data["deps_unemployment"] = ZA2900['v41']
data["deps_income"] = ZA2900['v42']
data["deps_retirement"] = ZA2900['v39']
data["deps_housing"] = ZA2900['v44']
data["deps_healthcare"] = ZA2900['v38']

c2fb = dict(zip(zip(cntry.cntry, cntry.year), cntry.foreignpct))
print(c2fb)

c2nm = dict(zip(zip(cntry.cntry, cntry.year), cntry.netmig))
print(c2nm)

c2ex = dict(zip(zip(cntry.cntry, cntry.year), cntry.socx))
print(c2ex)

c2emp = dict(zip(zip(cntry.cntry, cntry.year), cntry.emprate))
print(c2emp)

data['foreign_born'] = np.nan
data['net_migration'] = np.nan
data['expenditures'] = np.nan
data['employment_rate'] = np.nan

print(data)

# for idx, row in data.iterrows():
#     try:
#         c = country2country[int(data.iloc[idx][0])]
#         y = y2y[int(data.iloc[idx][1])]
#         print((c, y))
#         # data.at[data.columns.get_loc('foreign_born'), idx] = c2fb[(c, y)]
#         # # data.at['foreign_born', idx] = c2fb[(c, y)]
#         # # data.loc[:, ('net_migration', idx)] = c2nm[(c, y)]
#         # # data.loc[:, ('expenditures', idx)] = c2ex[(c, y)]
#         # # data.loc[:, ('employment_rate', idx)] = c2emp[(c, y)]
#
#         data['foreign_born'][idx] = c2fb[(c, y)]
#         data['net_migration'][idx] = c2nm[(c, y)]
#         data['expenditures'][idx] = c2ex[(c, y)]
#         data['employment_rate'][idx] = c2emp[(c, y)]
#     except KeyError:
#         pass

# data.to_pickle("data_dump")
data = pd.read_pickle("data_dump")
print(data)
print(data.columns.tolist())
# Table 4. Fit a two-way fixed effects model. Percent foreign born on welfare
# state attitudes, controlling for social welfare expenditures and the
# employment rate.
model = bambi.Model(data, dropna=True)
model.add('deps_jobs ~ 0')
# print(model)
model.add('foreign_born')
model.add('expenditures')
model.add('employment_rate')
results = model.fit(link="logit")
print(results)
print(results.summary())
# results.plot()

# Table 5. Fix a two-way fixed effects model. Net migration on welfare state
# attitudes, controlling for social welfare expenditures, exployment rate,
# and percent foreign born.
