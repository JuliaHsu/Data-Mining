import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

def read_transactions():
    transactions_attributes = pd.read_csv("./transactions_attributes.csv") 
    return transactions_attributes
def read_transactions_products():
    data = pd.read_csv("./BlackFriday.csv") 
    # transactions_cat = pd.read_csv("./transactions_productCat.csv")
    product = prepend(sorted(data.Product_ID.unique()),"Product = ")
    category = prepend(sorted(data.Product_Category_1.unique()),"Product_Category_1 = ")
    transactions_product = pd.crosstab(data.User_ID,data.Product_ID).astype('bool').astype('int')
    transactions_cat = pd.crosstab(data.User_ID,data.Product_Category_1).astype('bool').astype('int')
    colnames = product
    colnames_c = category
    transactions_product.columns = colnames
    transactions_cat.columns = colnames_c
    # print(transactions_product)
    # print(transactions_product.sum(axis=0).median(axis = 0))
    print(transactions_product.sum(axis=0).quantile([0.25,0.5,0.75,0.9]))
    return transactions_product,transactions_cat, data

def prepend(list, str): 
      
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 
def product_cat():
    transactions_cat = pd.read_csv("./transactions_productCat.csv") 
    transactions_att = pd.read_csv("./transactions_attributes.csv")
    transactions_att = transactions_att.drop(['User_ID'],axis =1)

    print(transactions_cat.sum(axis=0).quantile([0.1,0.25,0.5,0.75,0.9]))
    # freq_itemsets = apriori(transactions_cat, min_support=0.13, use_colnames=True)
    freq_itemsets = apriori(transactions_att, min_support=0.13, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_p.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules.to_csv("rules_freqitems_3_att.csv")
    # rules.to_csv("rules_freqitems_3_cat.csv")
def products_products(transactions_product):
    freq_itemsets = apriori(transactions_product, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_p.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    # rules.to_csv("rules_freqitems_3_p.csv")
    return rules

def products_gender(transactions_product,data):
    # obtain the unique values of gender attribute. Can be used for the column later
    gender = prepend(sorted(data.Gender.unique()),"Gender = ")
    # binarize the gender attribute: F & M
    genders = pd.crosstab(data.User_ID,data.Gender).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,genders], axis=1, sort=False)
    colnames = list(transactions_product.columns) + gender
    transactions.columns = colnames
    # obtain frequent item sets via apriori algotithm; including the gender attribute
    # set min support as 0.03 because the third quantile number of transactions of the product is 192
    # 0.25: 19.0; 0.50: 70.0; 0.75: 192.0; 0.90: 396.8
    # meaning that we only care about the top 25% of popular products; so the min support is 192 / 5892 = 0.03
    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_g.csv")
    # generate the association rules by filtering the metric lift greater than 2.0
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Gender" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Gender")]
    # eliminate rules that contain attribute "Gender" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Gender")]
    # rules.to_csv("rules_freqitems_3_g.csv")
    return rules, transactions

def products_age(transactions_product,data):
    age = prepend(sorted(data.Age.unique()),"Age = ")
    ages = pd.crosstab(data.User_ID,data.Age).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,ages], axis=1, sort=False)
    colnames = list(transactions_product.columns) + age
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_age.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Age" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Age")]
    # eliminate rules that contain attribute "Age" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Age")]

    # rules.to_csv("rules_freqitems_3_age.csv")
    return rules,transactions

def products_occupation(transactions_product,data):
    occupation = prepend(sorted(data.Occupation.unique()),"Occupation = ")
    occupations = pd.crosstab(data.User_ID,data.Occupation).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,occupations], axis=1, sort=False)
    colnames = list(transactions_product.columns) + occupation
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_occ.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)


    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

    
    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Occupation" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Occupation")]
    # eliminate rules that contain attribute "Occupation" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Occupation")]
    # rules.to_csv("rules_freqitems_3_occ.csv")
    return rules

def products_city(transactions_product,data):
    city = prepend(sorted(data.City_Category.unique()),"City_Category = ")
    cities = pd.crosstab(data.User_ID,data.City_Category).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,cities], axis=1, sort=False)
    colnames = list(transactions_product.columns) + city
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_city.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "City" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("City_Category")]
    # eliminate rules that contain attribute "City" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("City_Category")]
    # rules.to_csv("rules_freqitems_3_city.csv")
    return rules


def products_city_stay(transactions_product,data):
    city_s = prepend(sorted(data.Stay_In_Current_City_Years.unique()),"Stay_In_Current_City_Years = ")
    city_stay = pd.crosstab(data.User_ID,data.Stay_In_Current_City_Years).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,city_stay], axis=1, sort=False)
    colnames = list(transactions_product.columns) + city_s
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_city_stay.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Stay_In_Current_City_Years" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Stay_In_Current_City_Years")]
    # eliminate rules that contain attribute "Stay_In_Current_City_Years" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Stay_In_Current_City_Years")]
    # rules.to_csv("rules_freqitems_3_city_stay.csv")
    return rules

def products_marital(transactions_product,data):
    marital_s = prepend(sorted(data.Marital_Status.unique()),"Marital_Status = ")
    marital = pd.crosstab(data.User_ID,data.Marital_Status).astype('bool').astype('int')
    transactions = pd.concat([transactions_product,marital], axis=1, sort=False)
    colnames = list(transactions_product.columns) + marital_s
    transactions.columns = colnames

    freq_itemsets = apriori(transactions, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_marital.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

   
    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Marital_Status" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Marital_Status")]
    # eliminate rules that contain attribute "Marital_Status" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Marital_Status")]

    # rules.to_csv("rules_freqitems_3_marital.csv")
    return rules

def find_same_freq_itemsets(rules1,transactions1, rules2,transactions2):
    same_count=0
    same_rules =[]
    for i in range(rules1.shape[0]):
        for j in range(rules2.shape[0]):
            # print(type(rules1.iloc[i]['antecedents']))
            set1_a = rules1.iloc[i]['antecedents']
            set2_a = rules2.iloc[j]['antecedents']
            inter_a = set(set1_a) & set(set2_a) 
            union_a = set(set1_a) | set(set2_a) 
            # print("intersaction: ",inter_a)
            diff_a =set(set1_a) ^ set(set2_a) 
            # print("difference: ", diff_a)
            if len(diff_a) ==2:
                set1_c = rules1.iloc[i]['consequents']
                set2_c = rules2.iloc[j]['consequents']
                if (set(set1_c ) == set(set2_c)):
                    same_count+=1
                    # print(diff_a)
                    # print("rule1 ", rules1.iloc[i]['antecedents'])
                    # print("-->", rules1.iloc[i]['consequents'])
                    # print("rule2 ", rules2.iloc[j]['antecedents'])
                    # print("-->", rules2.iloc[j]['consequents'])
                    # print( set(set1_a) & set(set2_a) )
                    same_rules.append([union_a , set1_c, rules1.iloc[i]['lift'], rules2.iloc[i]['lift']])

    same_df = pd.DataFrame(same_rules,columns = ['antecedents', 'consequents','lift_1','lift_2'])
    lift_new = get_lift(same_df,transactions1,transactions2)
    same_df['lift_new'] = lift_new
    print(same_df)
    print(same_count)
    print("rules1 row: "+ str(rules1.shape[0]))
    print("rules2 row: "+ str(rules2.shape[0]))
    return same_df



def get_lift(same_df,transactions1,transactions2):
    lift = []
    data = pd.concat([transactions1,transactions2], axis=1, sort=False)
    prob_a = 0.0
    prob_c = 0.0
    prob_ac = 0.0
    for rule in range(same_df.shape[0]):
        antecedents = list(same_df.iloc[rule]['antecedents'])
        consequents = list(same_df.iloc[rule]['consequents'])
        a_and_c = antecedents + consequents
        df = data
        # check probability of antecedents
        for a in range(len(antecedents)):
            df = df[(df[antecedents[a]] == 1)]
        prob_a = df.shape[0]
        print(prob_a)

        df = data
        for c in range(len(consequents)):
            df = df[(df[consequents[c]] == 1)]
        prob_c = df.shape[0]
        print(prob_c)

        df = data
        for ac in range(len(a_and_c)):
            df = df[(df[a_and_c[ac]] == 1)]
        prob_ac = df.shape[0]
        print(prob_ac)

        lift.append(prob_ac/5892) /((prob_a/5892) *  (prob_c/5892))
    return lift
        

        
            
    
def generate_rules(attribute1,transactions1,attribute2,transactions2):
    transactions_cat = pd.concat([transactions1,transactions2], axis=1, sort=False)
    freq_itemsets = apriori(transactions_cat, min_support=0.03, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_city_stay.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Stay_In_Current_City_Years" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains(attribute1)]
    # eliminate rules that contain attribute "Stay_In_Current_City_Years" in consequents
    rules = rules[rules['antecedents'].astype(str).str.contains(attribute2)]
    # rules = rules[~rules["consequents"].astype(str).str.contains("Stay_In_Current_City_Years")]
    rules.to_csv("rules_freqitems_3_ga.csv")

def category_age(transactions_cat, data):
    age = prepend(sorted(data.Age.unique()),"Age = ")
    ages = pd.crosstab(data.User_ID,data.Age).astype('bool').astype('int')
    transactions = pd.concat([transactions_cat,ages], axis=1, sort=False)
    colnames = list(transactions_cat.columns) + age
    transactions.columns = colnames
    # print(transactions[transactions[:].isnull()])

    freq_itemsets = apriori(transactions, min_support=0.13, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_age.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Age" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Age")]
    # eliminate rules that contain attribute "Age" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Age")]

    rules.to_csv("rules_freqitems_3_age_c.csv")
    return rules,transactions

def category_occ(transactions_cat, data):
    city = prepend(sorted(data.City_Category.unique()),"City_Category = ")
    cities = pd.crosstab(data.User_ID,data.City_Category).astype('bool').astype('int')
    transactions = pd.concat([transactions_cat,cities], axis=1, sort=False)
    colnames = list(transactions_cat.columns) + city
    transactions.columns = colnames
    # print(transactions[transactions[:].isnull()])

    freq_itemsets = apriori(transactions, min_support=0.13, use_colnames=True)
    # freq_itemsets.to_csv("frequent_items_3_age.csv")
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=2.0)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    # eliminate rules only contain one antecedent
    rules = rules[rules['antecedent_len']>=2]
    # reserve the rules contain attribute "Age" in antecendents
    rules = rules[rules['antecedents'].astype(str).str.contains("Age")]
    # eliminate rules that contain attribute "Age" in consequents
    rules = rules[~rules["consequents"].astype(str).str.contains("Age")]

    rules.to_csv("rules_freqitems_3_city_c.csv")
    return rules,transactions

    

# transactions_attributes = read_transactions()
# tid = transactions_attributes.drop(columns = ['User_ID'])
# frequent_predicate_set = apriori(tid, min_support=0.03, use_colnames=True)
# frequent_predicate_set.to_csv("frequent_predicate_set.csv")
# rules = association_rules(frequent_predicate_set, metric="lift", min_threshold=1.0)
# rules.to_csv("rule_freq_predicate.csv")
product_cat()
transactions_product,transactions_cat, data = read_transactions_products()
rules_p = products_products(transactions_product)
rules_g , transactions_gender= products_gender(transactions_product,data)

rules_age_c, transactions_age_c = category_age(transactions_cat,data)
rules_occ_c, transactions_occ_c = category_occ(transactions_cat,data)
rules_age, transactions_age = products_age(transactions_product,data)
rules_occ = products_occupation(transactions_product,data)
rules_city = products_city(transactions_product,data)
rules_city_s = products_city_stay(transactions_product,data)
rules_marital = products_marital(transactions_product,data)


# same_a_g = find_same_freq_itemsets(rules_age,transactions_age,rules_g,transactions_gender)

# generate_rules("Age",transactions_age,"Gender",transactions_gender)
