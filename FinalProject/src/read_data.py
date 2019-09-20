import pandas as pd 
def read_data():
    data = pd.read_csv("./BlackFriday.csv") 
    return data

def prepend(list, str): 
      
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

def create_transactions_productCat(data):
    gender = prepend(sorted(data.Gender.unique()),"Gender = ")
    age = prepend(sorted(data.Age.unique()),"Age = ")
    occupation = prepend(sorted(data.Occupation.unique()),"Occupation = ")
    city = prepend(sorted(data.City_Category.unique()),"City_Category = ")
    stay_year = prepend(sorted(data.Stay_In_Current_City_Years.unique()),"Stay_In_Current_City_Years = ")
    marital = prepend(sorted(data.Marital_Status.unique()),"Marital_Status = ")
    category1 = prepend(sorted(data.Product_Category_1.unique()),"Product_Category_1 = ")
    colnames = gender + age + occupation + city + stay_year + marital + category1
   
    genders = pd.crosstab(data.User_ID,data.Gender).astype('bool').astype('int')
    ages = pd.crosstab(data.User_ID,data.Age).astype('bool').astype('int')
    occupations = pd.crosstab(data.User_ID,data.Occupation).astype('bool').astype('int')
    cities = pd.crosstab(data.User_ID,data.City_Category).astype('bool').astype('int')
    stay_years = pd.crosstab(data.User_ID,data.Stay_In_Current_City_Years).astype('bool').astype('int')
    marital_status = pd.crosstab(data.User_ID,data.Marital_Status).astype('bool').astype('int')
    categories = pd.crosstab(data.User_ID,data.Product_Category_1).astype('bool').astype('int')

    transactions = pd.concat([genders,ages,occupations,cities,stay_years,marital_status,categories], axis=1, sort=False)
    transactions.columns = colnames
    # use transactions.iloc[:,0:5] to access entries
    transactions.to_csv("transactions_attributes.csv")
data = read_data()
create_transactions_productCat(data)
