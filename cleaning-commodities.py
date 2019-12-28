
import numpy as np
import pandas as pd


df = pd.read_csv("/kaggle/input/international-energy-statistics/all_energy_statistics.csv")
df.shape
display(df.head())


pd.set_option("display.max_rows", 100)
df.category.value_counts()


df.commodity_transaction.value_counts().tail(50)


df.commodity_transaction.str.count(" - ").value_counts()


split_commodities = df.commodity_transaction.str.split(" -|– ", expand=True)
split_commodities.head()
split_commodities.columns = ["commodity", "transaction_type", "additional_transaction_info"]


split_commodities.head()


pd.set_option('display.max_rows', 100)
split_commodities.commodity.str.lower().str.strip().value_counts()


pd.set_option('display.max_rows', 250)
split_commodities.transaction_type = split_commodities.transaction_type.str.lower().str.strip()


split_commodities.transaction_type = split_commodities.transaction_type.str.replace("transformatin", "transformation")
split_commodities.transaction_type = split_commodities.transaction_type.str.replace("non energy use", "non-energy use")
split_commodities.transaction_type = split_commodities.transaction_type.str.replace(" /", "/")
split_commodities.transaction_type = split_commodities.transaction_type.str.replace("/ ", "/")
split_commodities.transaction_type.value_counts()


split_commodities.additional_transaction_info.str.lower().value_counts()


df = pd.concat([df,
                split_commodities.transaction_type.str.lower(),
                split_commodities.additional_transaction_info.str.lower(),
                split_commodities.commodity.str.lower()], axis=1)


df.head()


with open("cleaned_energy_data.csv", "w+") as file:
    file.write(df.to_csv())


cat cleaned_energy_data.csv 10
