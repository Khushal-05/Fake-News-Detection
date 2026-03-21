"""
Data preprocessing for multilingual Indian language fake news detection.

Fixes from original:
    - text_column variable was mutated inside the loop, causing every
      subsequent file to use the renamed column from the first file even
      if it had the original column name — classic loop-variable-leakage bug.
      Fixed by using a local variable per iteration.
    - prepare_dataset now raises a clear FileNotFoundError / KeyError with
      the filename and available columns when a text column cannot be found,
      instead of silently proceeding and crashing inside .apply().
    - Label mapping uses case-insensitive matching and raises a clear warning
      when unknown label values produce NaN (rather than silent dropna loss).
    - clean_text casts input to str before regex operations so non-string
      values from CSV reads (int, float non-NaN) don't crash the pipeline.
    - clean_text uses a simple None / pd.isna check that is safe for both
      plain strings (inference.py) and DataFrame cells (prepare_dataset).
    - split_data: stratify_col is added to a copy of df to avoid mutating
      the caller's DataFrame in place.
    - save_processed_data helper added to centralise output path logic used
      by the __main__ block.
"""

import os
import re
import warnings

import pandas as pd
import numpy as np
from langdetect import detect
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess multilingual Indian language text data for fake news detection."""

    # Unicode ranges for all 10 supported Indian scripts
    _INDIAN_UNICODE = (
        r'\u0900-\u097F'   # Devanagari  (Hindi, Marathi, Sanskrit)
        r'\u0980-\u09FF'   # Bengali     (Bengali, Assamese)
        r'\u0A00-\u0A7F'   # Gurmukhi    (Punjabi)
        r'\u0A80-\u0AFF'   # Gujarati
        r'\u0B00-\u0B7F'   # Oriya
        r'\u0B80-\u0BFF'   # Tamil
        r'\u0C00-\u0C7F'   # Telugu
        r'\u0C80-\u0CFF'   # Kannada
        r'\u0D00-\u0D7F'   # Malayalam
        r'\u0600-\u06FF'   # Arabic / Urdu
    )
    _CLEAN_PATTERN = re.compile(
        r'[^\w\s' + _INDIAN_UNICODE + r']'
    )

    def __init__(self):
        self.supported_languages = [
            'hi', 'en', 'ur', 'bn', 'ta', 'pa', 'mr', 'gu', 'as', 'ml',
            'pn',   # alias kept for backwards compatibility with existing data
        ]

    # ── Text cleaning ─────────────────────────────────────────────────────── #

    def clean_text(self, text) -> str:
        """
        Clean and normalise a single text string.

        Safe to call with:
            - plain Python str  (from inference.py / test.py)
            - pandas Series cell values (from prepare_dataset)
            - None, float NaN, or other non-string types

        Returns empty string for null / empty inputs.
        """
        # Handle None and pandas NA / NaN values
        if text is None:
            return ""
        try:
            if pd.isna(text):
                return ""
        except (TypeError, ValueError):
            pass   # pd.isna raises on some types — treat as non-null

        text = str(text).strip()
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters, keeping word chars, spaces, and
        # characters from all supported Indian scripts
        text = self._CLEAN_PATTERN.sub(' ', text)

        return text.strip()

    # ── Language detection ────────────────────────────────────────────────── #

    def detect_language(self, text) -> str:
        """
        Detect language code from text.
        Returns the detected code if it is in supported_languages, else 'unknown'.
        """
        try:
            lang = detect(str(text))
            return lang if lang in self.supported_languages else 'unknown'
        except Exception:
            return 'unknown'

    # ── Dataset preparation ───────────────────────────────────────────────── #

    def prepare_dataset(
        self,
        file_paths: list,
        label_column: str = 'label',
        text_column: str = 'text',
    ) -> pd.DataFrame:
        """
        Load, clean, and combine CSVs from multiple language sources.

        Args:
            file_paths:   List of (file_path, language_code) tuples.
            label_column: Column name containing the label (default: 'label').
            text_column:  Preferred column name for text (default: 'text').
                          If absent, falls back to any column whose name
                          contains 'text' or 'content'.

        Returns:
            Combined DataFrame with columns: cleaned_text, label, language.

        Raises:
            FileNotFoundError: if a CSV file does not exist.
            KeyError:          if no usable text column is found in a file.
            ValueError:        if the label column is missing from a file.
        """
        all_data = []

        for file_path, language in file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"Data file not found: '{file_path}'"
                )

            df = pd.read_csv(file_path)

            # ── Resolve text column (local variable — does NOT leak) ────── #
            local_text_col = text_column
            if local_text_col not in df.columns:
                candidates = [
                    c for c in df.columns
                    if 'text' in c.lower() or 'content' in c.lower()
                ]
                if not candidates:
                    raise KeyError(
                        f"No text column found in '{file_path}'. "
                        f"Available columns: {list(df.columns)}. "
                        f"Specify the correct column via text_column=."
                    )
                local_text_col = candidates[0]
                print(
                    f"  [info] '{file_path}': using '{local_text_col}' "
                    f"as text column ('{text_column}' not found)."
                )

            # ── Validate label column ─────────────────────────────────── #
            if label_column not in df.columns:
                raise ValueError(
                    f"Label column '{label_column}' not found in '{file_path}'. "
                    f"Available columns: {list(df.columns)}."
                )

            # ── Clean text ────────────────────────────────────────────── #
            df['cleaned_text'] = df[local_text_col].apply(self.clean_text)

            # ── Standardise labels (0 = fake, 1 = real) ──────────────── #
            if df[label_column].dtype == object:
                label_map = {
                    'fake': 0, 'false': 0, '0': 0,
                    'real': 1, 'true':  1, '1': 1,
                }
                df['label'] = df[label_column].str.strip().str.lower().map(label_map)
                unmapped = df['label'].isna().sum()
                if unmapped > 0:
                    unique_vals = df[label_column].unique().tolist()
                    warnings.warn(
                        f"'{file_path}': {unmapped} rows have unrecognised "
                        f"label values {unique_vals} — they will be dropped. "
                        f"Expected: fake/false/0 or real/true/1."
                    )
            else:
                df['label'] = df[label_column].astype(int)

            # ── Language column ───────────────────────────────────────── #
            df['language'] = language

            # ── Keep only required columns and drop nulls ─────────────── #
            df = df[['cleaned_text', 'label', 'language']].dropna()

            # Drop rows with empty cleaned text
            df = df[df['cleaned_text'].str.strip() != '']

            print(
                f"  Loaded '{file_path}' [{language}]: "
                f"{len(df)} rows after cleaning."
            )
            all_data.append(df)

        if not all_data:
            raise ValueError("No data loaded — file_paths list was empty or all files failed.")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} rows total.")
        return combined_df

    # ── Train / val / test split ──────────────────────────────────────────── #

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_by_language: bool = True,
    ) -> tuple:
        """
        Split data into train, validation, and test sets.

        Args:
            df:                   Input DataFrame (not modified in-place).
            test_size:            Fraction for test set (default 0.2).
            val_size:             Fraction for val set from total (default 0.1).
            random_state:         Random seed for reproducibility.
            stratify_by_language: If True, stratify by language×label combination
                                  to preserve class balance per language.

        Returns:
            (train_df, val_df, test_df) — all with reset indices.
        """
        # Work on a copy so we never mutate the caller's DataFrame
        df = df.copy()

        if stratify_by_language:
            # Combined stratify key: e.g. 'hi_0', 'en_1'
            df['_stratify'] = df['language'] + '_' + df['label'].astype(str)

            # Drop strata with fewer than 2 samples (train_test_split requirement)
            counts = df['_stratify'].value_counts()
            rare   = counts[counts < 2].index.tolist()
            if rare:
                warnings.warn(
                    f"Dropping {df['_stratify'].isin(rare).sum()} rows from "
                    f"strata with < 2 samples: {rare}"
                )
                df = df[~df['_stratify'].isin(rare)]

            stratify_col = df['_stratify']
        else:
            stratify_col = df['label']

        # First split: (train + val) vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )

        # Second split: train vs val
        if stratify_by_language:
            stratify_train = train_val_df['_stratify']
        else:
            stratify_train = train_val_df['label']

        # Adjust val fraction relative to the train+val subset
        adjusted_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify_train,
        )

        # Drop the temporary stratify column from all three splits
        for split in (train_df, val_df, test_df):
            if '_stratify' in split.columns:
                split.drop(columns=['_stratify'], inplace=True)

        print(
            f"Split sizes — "
            f"train: {len(train_df)}  "
            f"val: {len(val_df)}  "
            f"test: {len(test_df)}"
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # ── Convenience: save splits ──────────────────────────────────────────── #

    @staticmethod
    def save_processed_data(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = 'data/processed',
    ) -> None:
        """Save the three splits to CSV files in output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'),   index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        print(f"Saved train/val/test CSVs to '{output_dir}/'")


# ════════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                                 #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Add / uncomment sources as you collect datasets
    data_sources = [
        ('data/raw/bengali_fake_news_dataset.csv',  'bn'),
        # ('data/raw/hindi_fake_news_dataset.csv',   'hi'),
        # ('data/raw/tamil_fake_news_dataset.csv',   'ta'),
        # ('data/raw/marathi_fake_news_dataset.csv', 'mr'),
        # ('data/raw/gujarati_fake_news_dataset.csv','gu'),
        # ('data/raw/malayalam_fake_news_dataset.csv','ml'),
        # ('data/raw/punjabi_fake_news_dataset.csv', 'pa'),
        # ('data/raw/urdu_fake_news_dataset.csv',    'ur'),
        # ('data/raw/english_fake_news_dataset.csv', 'en'),
        # ('data/raw/assamese_fake_news_dataset.csv','as'),
    ]

    df = preprocessor.prepare_dataset(data_sources)
    train_df, val_df, test_df = preprocessor.split_data(df)
    DataPreprocessor.save_processed_data(train_df, val_df, test_df)

    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("\nLanguage distribution (train):")
    print(train_df['language'].value_counts())
    print("\nLabel distribution (train):")
    print(train_df['label'].value_counts())