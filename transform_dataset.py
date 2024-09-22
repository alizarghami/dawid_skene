import pandas as pd

original_path = "../crowdsourced-datasets/my-dataset/binary_labeled_data.csv"
transformed_path = "../crowdsourced-datasets/my-dataset/transformed_dataset.csv"


df = pd.read_csv(original_path)

df_melted = pd.melt(df, id_vars=['participant', 'timestamp', 'consensus_binary_label'], value_vars=['verbal_binary_label', 'activpal_binary_label', 'google_binary_label'],
                    var_name='workerID', value_name='response')

# df_melted['goldLabel'] = None
df_melted.rename(columns={'consensus_binary_label': 'goldLabel'}, inplace=True)
df_melted.rename(columns={'participant': 'taskID'}, inplace=True)

new_df = df_melted[['workerID', 'taskID', 'response', 'goldLabel', 'timestamp']]

new_df.to_csv(transformed_path, index=False)
