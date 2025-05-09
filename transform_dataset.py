import pandas as pd
from scipy.stats import mode

original_path = "../crowdsourced-datasets/my-dataset/labeled-by-all.csv"
transformed_path = "../crowdsourced-datasets/my-dataset/transformed_dataset.csv"


df = pd.read_csv(original_path)
# Perform majority voting on the three columns
def majority_vote(row):
    labels = row[['verbal_binary_label', 'activpal_binary_label', 'google_binary_label']]
    if len(set(labels)) == 3:
        return 0
    return mode(labels)[0][0]

df['majority_vote'] = df.apply(majority_vote, axis=1)

df.to_csv("../crowdsourced-datasets/my-dataset/mv.csv", index=False)
df = df[df['majority_vote'] != 0] # Remove rows where there is no majority vote

df_melted = pd.melt(df, id_vars=['participant', 'timestamp', 'majority_vote'], value_vars=['verbal_binary_label', 'activpal_binary_label', 'google_binary_label'],
                    var_name='workerID', value_name='response')

# df_melted['goldLabel'] = None
df_melted.rename(columns={'majority_vote': 'goldLabel'}, inplace=True)
df_melted['taskID'] = df_melted['participant'].astype(str) + '_' + df_melted['timestamp'].astype(str)
# df_melted.rename(columns={'participant': 'taskID'}, inplace=True)

new_df = df_melted[['workerID', 'taskID', 'response', 'goldLabel', 'timestamp']]

new_df.to_csv(transformed_path, index=False)
