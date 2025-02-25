import pandas as pd


spandf = pd.read_parquet('./unlp-2025-shared-task/data/span_detection/train.parquet')
techclass = pd.read_parquet('unlp-2025-shared-task/data/techniques_classification/train.parquet')

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)

csv_file = './unlp-2025-shared-task/data/span_detection/train.csv'
spandf.to_csv(csv_file, index=False)
tech_csv = './unlp-2025-shared-task/data/techniques_classification/train.csv'
techclass.to_csv(tech_csv, index=False)

spandf = pd.read_csv(tech_csv)
techclass = pd.read_csv(tech_csv)

print(f"File saved as {csv_file}")

## print(spandf.head())
## print(techclass.head())

## print(spandf.info())
## print(techclass.info())

manipulative_posts = spandf[spandf['manipulative'] == True]
print(manipulative_posts.head())


