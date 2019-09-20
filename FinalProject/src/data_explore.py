import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
data: pd.DataFrame = pd.read_csv('./BlackFriday.csv')
describe = data.describe()
describe.loc['#unique'] = data.nunique()

describe.to_csv("data_describe.csv")

#null
null_percent = (data.isnull().sum() / len(data))*100
df_null = (pd.DataFrame(null_percent[null_percent > 0].apply(lambda x: "{:.2f}%".format(x)),columns=['Null %']))
df_null.to_csv("data_null.csv")
# category description
cat_describe = data[['Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status', 'Product_Category_1']].astype('object').describe()
cat_describe.loc['percent'] = 100*cat_describe.loc['freq'] / cat_describe.loc['count']
cat_describe.to_csv("cat_describe.csv")

explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(data['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',
        startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(data['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.1f%%',
        startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
df = data.groupby('Product_Category_1').count()
df = 1/(df['User_ID']/data.shape[0])
print(df.to_dict())