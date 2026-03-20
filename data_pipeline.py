"""
CSCI316 Project 2 — Data Collection & Cleaning Pipeline
Task: Tamil-English Code-Switched Sentiment Analysis
Primary Dataset: community-datasets/tamilmixsentiment (HuggingFace)
"""

import re
import unicodedata
import pandas as pd
from datasets import load_dataset
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load_tamilmix():
    """Load the primary Tamil-English code-switched sentiment dataset."""
    print("[INFO] Loading tamilmixsentiment dataset from HuggingFace...")
    dataset = load_dataset("community-datasets/tamilmixsentiment")
    
    train_df = pd.DataFrame(dataset["train"])
    val_df   = pd.DataFrame(dataset["validation"])
    test_df  = pd.DataFrame(dataset["test"])
    
    print(f"  Train:      {len(train_df):>6} rows")
    print(f"  Validation: {len(val_df):>6} rows")
    print(f"  Test:       {len(test_df):>6} rows")
    print(f"  Total:      {len(train_df) + len(val_df) + len(test_df):>6} rows")
    print(f"  Columns:    {list(train_df.columns)}")
    print(f"  Labels:     {sorted(train_df['label'].unique())}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LABEL INSPECTION & MAPPING
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "Positive":       0,
    "Negative":       1,
    "Mixed_feelings": 2,
    "unknown_state":  3,
    "not-Tamil":      4,
}

# For binary classification experiments (optional), you can merge to 3 classes:
BINARY_MAP = {
    "Positive":       "Positive",
    "Negative":       "Negative",
    "Mixed_feelings": "Neutral",
    "unknown_state":  None,   # Will be dropped
    "not-Tamil":      None,   # Will be dropped
}

def inspect_labels(df, split_name="train"):
    """Print label distribution for a split."""
    counts = Counter(df["label"])
    total = len(df)
    print(f"\n[LABEL DISTRIBUTION — {split_name.upper()}]")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {count:>5} ({count/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

# Tamil Unicode block: U+0B80 to U+0BFF
TAMIL_UNICODE_RANGE = re.compile(r'[\u0B80-\u0BFF]')

# Common Tamil Romanization / Tanglish patterns (heuristic)
TANGLISH_PATTERN = re.compile(r'\b(tha|da|pa|ma|na|va|la|ra|ka|nga|nna|lla|thi|di)\b', re.IGNORECASE)

def is_tamil_script(text):
    """Returns True if the text contains any Tamil Unicode characters."""
    return bool(TAMIL_UNICODE_RANGE.search(text))

def is_code_switched(text):
    """
    Heuristic: a comment is code-switched if it contains BOTH:
    - At least one ASCII word (English/Tanglish)
    - Either Tamil script OR Tanglish-like tokens
    """
    has_ascii_words = bool(re.search(r'[a-zA-Z]', text))
    has_tamil = is_tamil_script(text) or bool(TANGLISH_PATTERN.search(text))
    return has_ascii_words and has_tamil

def clean_text(text):
    """
    Clean a single text string:
    1. Normalize unicode (NFC — important for Tamil script)
    2. Remove URLs
    3. Remove HTML entities
    4. Remove excessive punctuation / repeated characters
    5. Strip leading/trailing whitespace
    6. Collapse multiple spaces
    7. Keep Tamil script, Latin script, digits, and core punctuation
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode normalization (NFC is standard for Indic scripts)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 3. Remove HTML entities like &amp; &lt; &#39; etc.
    text = re.sub(r'&[a-zA-Z]+;|&#\d+;', '', text)
    
    # 4. Remove repeated characters > 3x (e.g. "haaaaa" → "haaa", "!!!!!!" → "!!!")
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    
    # 5. Remove emojis and other non-Tamil/Latin/digit characters
    #    Keep: Tamil script (U+0B80-U+0BFF), Latin, digits, spaces, basic punctuation
    text = re.sub(r'[^\u0B80-\u0BFFa-zA-Z0-9\s.,!?\'"\-]', ' ', text)
    
    # 6. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Strip
    text = text.strip()
    
    return text

def filter_and_clean(df):
    """Apply all cleaning and filtering steps to a DataFrame."""
    original_len = len(df)
    
    # Step A: Drop rows with null text
    df = df.dropna(subset=["text"])
    
    # Step B: Clean text
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    
    # Step C: Drop empty strings after cleaning
    df = df[df["text_clean"].str.len() > 0]
    
    # Step D: Drop very short texts (less than 3 characters — uninformative)
    df = df[df["text_clean"].str.len() >= 3]
    
    # Step E: Drop duplicate text entries
    df = df.drop_duplicates(subset=["text_clean"])
    
    # Step F: Add metadata columns
    df["has_tamil_script"] = df["text_clean"].apply(is_tamil_script)
    df["is_code_switched"]  = df["text_clean"].apply(is_code_switched)
    df["text_length"]       = df["text_clean"].str.len()
    df["word_count"]        = df["text_clean"].str.split().str.len()
    
    # Step G: Map labels to integers
    df["label_int"] = df["label"].map(LABEL_MAP)
    
    print(f"  Rows before cleaning: {original_len}")
    print(f"  Rows after cleaning:  {len(df)}")
    print(f"  Dropped: {original_len - len(df)}")
    print(f"  Has Tamil script:  {df['has_tamil_script'].sum()} ({df['has_tamil_script'].mean()*100:.1f}%)")
    print(f"  Code-switched:     {df['is_code_switched'].sum()} ({df['is_code_switched'].mean()*100:.1f}%)")
    
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: OPTIONAL — 3-CLASS VERSION (for simpler baseline)
# ─────────────────────────────────────────────────────────────────────────────

def make_3class_version(df):
    """
    Collapse 5 labels into 3:
      Positive → Positive
      Negative → Negative
      Mixed_feelings → Neutral
      unknown_state, not-Tamil → DROPPED

    Returns a filtered DataFrame with a new column 'label_3class'.
    """
    df = df.copy()
    df["label_3class"] = df["label"].map(BINARY_MAP)
    df = df[df["label_3class"].notna()].reset_index(drop=True)
    df["label_3class_int"] = df["label_3class"].map({"Positive": 0, "Negative": 1, "Neutral": 2})
    print(f"\n[3-CLASS] Rows after dropping unknown/not-Tamil: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SAVE CLEANED DATA
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(train_df, val_df, test_df, prefix="cleaned"):
    """Save cleaned splits as CSVs."""
    cols = ["text", "text_clean", "label", "label_int",
            "has_tamil_script", "is_code_switched", "text_length", "word_count"]
    
    train_df[cols].to_csv(f"{prefix}_train.csv", index=False, encoding="utf-8")
    val_df[cols].to_csv(f"{prefix}_val.csv", index=False, encoding="utf-8")
    test_df[cols].to_csv(f"{prefix}_test.csv", index=False, encoding="utf-8")
    
    print(f"\n[SAVED] {prefix}_train.csv ({len(train_df)} rows)")
    print(f"[SAVED] {prefix}_val.csv   ({len(val_df)} rows)")
    print(f"[SAVED] {prefix}_test.csv  ({len(test_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BACK-TRANSLATION AUGMENTATION (stub — for low-resource classes)
# ─────────────────────────────────────────────────────────────────────────────

def back_translation_augment(df, target_label="Negative", augment_factor=1.5):
    """
    Stub for back-translation augmentation.
    
    Strategy:
      1. Filter rows of the minority class (e.g., Negative or Mixed_feelings)
      2. Translate text Tamil→English using googletrans or deep_translator
      3. Translate back English→Tamil
      4. Append to dataframe as augmented samples

    NOTE: Run this after confirming your environment has internet access
    and the `deep_translator` library installed:
        pip install deep-translator

    This is a stub — uncomment and run when ready.
    """
    # from deep_translator import GoogleTranslator
    # 
    # minority = df[df["label"] == target_label].copy()
    # n_to_generate = int(len(minority) * (augment_factor - 1))
    # sample = minority.sample(n=min(n_to_generate, len(minority)), random_state=42)
    # 
    # augmented_rows = []
    # for _, row in sample.iterrows():
    #     try:
    #         # Tamil → English
    #         en_text = GoogleTranslator(source="ta", target="en").translate(row["text_clean"])
    #         # English → Tamil
    #         back_text = GoogleTranslator(source="en", target="ta").translate(en_text)
    #         augmented_rows.append({
    #             "text": row["text"],
    #             "text_clean": back_text,
    #             "label": row["label"],
    #             "label_int": row["label_int"],
    #             "has_tamil_script": is_tamil_script(back_text),
    #             "is_code_switched": is_code_switched(back_text),
    #             "text_length": len(back_text),
    #             "word_count": len(back_text.split()),
    #             "augmented": True
    #         })
    #     except Exception as e:
    #         print(f"[WARN] Translation failed: {e}")
    # 
    # aug_df = pd.DataFrame(augmented_rows)
    # print(f"[AUGMENT] Generated {len(aug_df)} new samples for '{target_label}'")
    # return pd.concat([df, aug_df], ignore_index=True)
    
    print("[AUGMENT] Back-translation stub — uncomment code to enable.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DATASET SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(train_df, val_df, test_df):
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"{'Split':<12} {'Rows':>6} {'Avg Words':>10} {'Code-Switched %':>17}")
    print("-"*50)
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        avg_words = df["word_count"].mean()
        cs_pct = df["is_code_switched"].mean() * 100
        print(f"{name:<12} {len(df):>6} {avg_words:>10.1f} {cs_pct:>16.1f}%")
    print("="*60)
    
    print("\nNOTE for report:")
    print("  - Dataset: community-datasets/tamilmixsentiment")
    print("  - Source: YouTube comments (Tamil movie trailers, 2019)")
    print("  - Script mix: Roman script Tanglish + Tamil Unicode")
    print("  - Code-switch types: Inter-sentential, Intra-sentential, Tag-switching")
    print("  - Labels: Positive, Negative, Mixed_feelings, unknown_state, not-Tamil")
    print("  - Known challenge: Class imbalance (Positive >> Negative)")
    print("  - Preprocessing decisions: NFC normalization, URL removal,")
    print("    deduplication, length filtering, emoji removal")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load
    train_df, val_df, test_df = load_tamilmix()
    
    # 2. Inspect raw labels
    inspect_labels(train_df, "train")
    inspect_labels(val_df,   "validation")
    inspect_labels(test_df,  "test")
    
    # 3. Clean
    print("\n[CLEANING — TRAIN]")
    train_clean = filter_and_clean(train_df)
    print("\n[CLEANING — VALIDATION]")
    val_clean = filter_and_clean(val_df)
    print("\n[CLEANING — TEST]")
    test_clean = filter_and_clean(test_df)
    
    # 4. Optional: make 3-class version
    # train_3c = make_3class_version(train_clean)
    
    # 5. Optional: augment minority classes
    # train_clean = back_translation_augment(train_clean, target_label="Mixed_feelings")
    
    # 6. Save
    save_splits(train_clean, val_clean, test_clean, prefix="tamilmix")
    
    # 7. Summary
    print_summary(train_clean, val_clean, test_clean)
