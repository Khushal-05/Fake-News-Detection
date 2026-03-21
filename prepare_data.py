# prepare_data.py
import sys
sys.path.append('.')

from utils.preprocessing import DataPreprocessor
import pandas as pd

# Initialize preprocessor
preprocessor = DataPreprocessor()

print("Loading Zenodo Hindi dataset...")

# Load your converted dataset
df = pd.read_csv('data/raw/assamese_fake_news_dataset.csv')

# If you have multilingual:
# df = pd.read_csv('data/raw/multilingual_zenodo_dataset.csv')

print(f"Loaded {len(df)} samples")
print(f"Fake: {sum(df['label']==0)} | Real: {sum(df['label']==1)}")

# Clean text
print("Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)

# Remove empty texts
df = df[df['cleaned_text'].str.len() > 0]

print(f"After cleaning: {len(df)} samples")

# Split data
print("Splitting data...")
train_df, val_df, test_df = preprocessor.split_data(
    df, 
    test_size=0.2, 
    val_size=0.1,
    stratify_by_language=True
)

# Save
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"\n✅ Data preparation complete!")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nLanguage distribution in training set:")
print(train_df['language'].value_counts())
