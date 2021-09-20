###############################################################
                   # ASSOCIATION RULE #
###############################################################

# The aim is to suggest products to users in the product purchasing process by applying
# association analysis to the online retail II dataset.

# 1. Data Preprocessing
# 2. Preparing the Data Structure (Invoice-Product Matrix)
# 3. Extracting Association Rules
# 4. Preparing the Script of the Study
# 5. Recommending a Product to Users at the Basket Stage

###############################################################
                 # Data Preprocessing #
###############################################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# !pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel(r"C:\Users\hp\PycharmProjects\VBO\WEEK_03\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.head(10)

###############################################################
   # Preparing the Data Structure (Invoice-Product Matrix) #
###############################################################

# The situation where we want the data to come:

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

df_fr = df[df['Country'] == "France"]
# type(df_fr) # DataFrame

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df.head()

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
fr_inv_pro_df.head()


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10002)

###############################################################
               # Extracting Association Rules #
###############################################################

# Possibilities of all possible product combinations:
frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

# Extracting association rules:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(500)

# support: probability of seeing both
# confidence: probability of getting Y when X is taken.
# lift: When X is taken, the probability of getting Y increases by this much.

###############################################################
           # Preparing the Script of the Study #
###############################################################

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df_ = pd.read_excel(r"C:\Users\hp\PycharmProjects\VBO\WEEK_03\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

# let's do it according to another country as an example.
rules_grm = create_rules(df, country="Germany")
rules_grm.sort_values("lift", ascending=False).head(50)


###############################################################
   # Recommending a Product to Users at the Basket Stage #
###############################################################

# Sample:
# User sample product id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:2]

check_id(df, 22556)

check_id(df, recommendation_list[0])


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)


