import pandas as pd
import random


seed = 1

# Reproducible results
random.seed (seed)


# @var df DataFrame
df = pd.read_csv ('../assets/PoliticES/2022/dataset.csv')


# Keep 50 tweets per user
df = df.groupby ('label').sample (n = 50, random_state = seed)


# @var fields_to_export List
fields_to_export = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass', 'tweet']


# @var fields_without_labels_to_export List
fields_without_labels_to_export = ['tweet']


# @var number_of_users_in_training int
number_of_users_in_training = 100


# @var number_of_users_in_testing int
number_of_users_in_testing = 20


# @var users List Get the users from training or validation
users = list (df.loc[df['__split'].isin (['train', 'val'])]['label'].unique ())
users = random.sample (users, number_of_users_in_training + number_of_users_in_testing)


# @var users_in_testing List Sample from the sample
users_in_testing = random.sample (users, number_of_users_in_testing)


# @var users_in_training List Get the users that are not in testing
users_in_training = list (set (users) ^ set (users_in_testing))


# @var df_train DataFrame
df_train = df.loc[df['label'].isin (users_in_training)][fields_to_export]
df_test = df.loc[df['label'].isin (users_in_testing)][fields_to_export]


# Subsample tweets
df_train.to_csv ('../assets/PoliticES/2022/deploy-dataset/development.csv')
df_test.to_csv ('../assets/PoliticES/2022/deploy-dataset/development_test.csv')
df_test[fields_without_labels_to_export].to_csv ('../assets/PoliticES/2022/deploy-dataset/development_test_without_labels.csv')
