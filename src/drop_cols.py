import pandas as pd

file_path = 'data/final/parallel_corpus_v5_labse_gold.csv'
print(f'Loading dataset from {file_path}...')
df = pd.read_csv(file_path)

print(f'Current columns: {df.columns.tolist()}')
columns_to_drop = ['similarity_bn_bn', 'similarity_bridge', 'source_file']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

print(f'Final columns: {df.columns.tolist()}')
df.to_csv(file_path, index=False)
print(f'\n✅ Successfully removed scores and filename!')
print(f'🎯 Total Pairs Generated: {len(df)}')
