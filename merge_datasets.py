import os
import glob
import re
import pandas as pd
import numpy as np

# Define mappings and configurations
DATA_DIR = r"c:\Users\ASUS\Documents\Data Science\Data Analysis\Machine learning\data"
OUTPUT_FILE = r"c:\Users\ASUS\Documents\Data Science\Data Analysis\Machine learning\final_merged_dataset.csv"

# Pre-defined mapping for expected titles to clean prefixes
PREFIX_MAPPING = {
    "Number of Employed Persons by Class of Worker": "EmployedClass",
    "Mean Hours Worked in One Week": "MeanHours",
    "Levels of Key Employment Indicators": "LevelsKEI",
    "Rates Key Employment Indicators": "RatesKEI",
    "Population 15 Years Old and Over by Sex and by Age Group": "Population15Plus",
    "Persons in the Labor Force by Sex and by Age Group": "LaborForce",
    "Employed Persons by Sex and by Age Group": "Employed",
    "Unemployed Persons by Sex and by Age Group": "Unemployed",
    "Persons Not in the Labor Force by Sex and by Age Group": "NotInLaborForce",
    "Underemployed Persons by Sex and by Age Group": "Underemployed",
    "Visibly Underemployed Persons by Sex and by Age Group": "VisiblyUnderemployed"
}

VALID_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", 
    "august", "september", "october", "november", "december"
}

MONTH_NUM_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

def clean_prefix_name(title: str) -> str:
    """Extracts and shortens a metric name from the file title row."""
    if not isinstance(title, str):
        return "Feature"
    
    # Extract portion before colon (which contains data ranges)
    base_title = title.split(":")[0].strip()
    
    # If in standard mapping, use it
    if base_title in PREFIX_MAPPING:
        return PREFIX_MAPPING[base_title]
    
    # Dynamic fallback: strip trailing standard grouping suffixes
    clean = base_title
    suffixes_to_strip = [
        " by Sex and by Age Group", 
        " by Class of Worker", 
        " in One Week", 
        " by Sex", 
        " by Age Group"
    ]
    for suffix in suffixes_to_strip:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
            
    # Keep alphanumeric characters and capitalize words to form CamelCase
    words = re.findall(r'[a-zA-Z0-9]+', clean)
    if not words:
        return "Feature"
    return "".join(w.capitalize() for w in words)

def process_excel_file(filepath: str) -> pd.DataFrame:
    """Cleans, headers-flattens, and processes a single economic Excel file."""
    print(f"Processing: {os.path.basename(filepath)}...")
    
    # 1. Read sheet without parsing headers to inspect structure
    df = pd.read_excel(filepath, header=None)
    
    # 2. Extract Title and Feature Prefix
    title = df.iloc[0, 0]
    prefix = clean_prefix_name(title)
    print(f"  -> Title: '{title}'")
    print(f"  -> Extracted Prefix: '{prefix}'")
    
    # 3. Locate where calendar months start in column 1 (0-indexed)
    start_row = None
    for idx, val in enumerate(df[1]):
        if isinstance(val, str) and val.strip().lower() in VALID_MONTHS:
            start_row = idx
            break
            
    if start_row is None:
        raise ValueError(f"No valid calendar month found in column 1 of {os.path.basename(filepath)}")
    
    # 4. Hierarchical Header Flattening
    # Keep header rows from index 1 to start_row - 1, dropping rows that are all NaN
    header_df = df.iloc[1:start_row, :].copy()
    header_df = header_df.dropna(how='all')
    
    # Separate key columns (Year, Month) from headers of feature columns (from col 2 onwards)
    header_features = header_df.iloc[:, 2:]
    
    # Horizontally forward-fill merged cell labels
    header_features_filled = header_features.ffill(axis=1)
    
    # Concatenate hierarchical layers with a pipe '|'
    col_names = ["Year", "Month"]
    for col_idx in range(header_features_filled.shape[1]):
        col_vals = header_features_filled.iloc[:, col_idx].tolist()
        cleaned_vals = [str(v).strip() for v in col_vals if pd.notna(v) and str(v).strip() != ""]
        flat_name = " | ".join(cleaned_vals)
        
        # Avoid empty column names
        if not flat_name:
            flat_name = f"Col_{col_idx + 1}"
            
        col_names.append(f"{prefix} - {flat_name}")
        
    # 5. Extract and Clean Data Rows
    df_data = df.iloc[start_row:].copy()
    df_data.columns = col_names
    
    # Forward-fill Year (column 0)
    df_data["Year"] = df_data["Year"].ffill()
    
    # Filter rows: keep only valid month rows (filters out annual averages, metadata & footnotes)
    df_data = df_data[df_data["Month"].astype(str).str.strip().str.lower().isin(VALID_MONTHS)]
    
    # Standardize Month column casing
    df_data["Month"] = df_data["Month"].astype(str).str.strip().str.title()
    
    # Handle numeric columns (column 2 onwards)
    for col in df_data.columns[2:]:
        # Strip commas and whitespace
        s = df_data[col].astype(str).str.replace(",", "", regex=False).str.strip()
        # Coerce '.' placeholders and any other text non-numeric symbols to NaN
        s = s.replace(r'^\s*\.\s*$', np.nan, regex=True)
        df_data[col] = pd.to_numeric(s, errors='coerce')
        
    # Ensure Year is integer
    df_data = df_data.dropna(subset=["Year"])
    df_data["Year"] = df_data["Year"].astype(int)
    
    # Drop rows that are entirely NaN in all feature columns (indicating missing quarterly survey data)
    df_data = df_data.dropna(subset=df_data.columns[2:], how='all')
    
    print(f"  -> Processed Shape: {df_data.shape}")
    return df_data

def run_pipeline():
    """Main pipeline execution to discover, clean, and merge all datasets."""
    print("Starting ML Dataset Merging Pipeline...")
    
    # Find all Excel files
    excel_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
    if not excel_files:
        print(f"Error: No Excel files found in {DATA_DIR}")
        return
        
    print(f"Found {len(excel_files)} Excel files to merge.")
    
    master_df = None
    
    for filepath in excel_files:
        df_clean = process_excel_file(filepath)
        
        if master_df is None:
            master_df = df_clean
        else:
            # Check and resolve any naming collisions in non-key columns
            collisions = set(df_clean.columns).intersection(set(master_df.columns)) - {"Year", "Month"}
            if collisions:
                file_stem = os.path.splitext(os.path.basename(filepath))[0]
                print(f"  [COLLISION] Resolving columns {collisions} with suffix '_{file_stem}'")
                rename_map = {c: f"{c}_{file_stem}" for c in collisions}
                df_clean = df_clean.rename(columns=rename_map)
            
            # Outer merge to synchronize timelines
            master_df = pd.merge(master_df, df_clean, on=["Year", "Month"], how="outer")
            
    if master_df is None or len(master_df) == 0:
        print("Pipeline aborted: Master dataset is empty.")
        return
        
    # 6. Chronological Sorting & Index Mapping
    # Map text months to numeric
    master_df["Month"] = master_df["Month"].map(MONTH_NUM_MAP)
    master_df = master_df.dropna(subset=["Month"])
    master_df["Month"] = master_df["Month"].astype(int)
    
    # Chronologically sort by Year then Month, and drop the temporary sorting index
    master_df = master_df.sort_values(by=["Year", "Month"]).reset_index(drop=True)
    
    # 7. Print Data Quality Report
    print("\n" + "="*40)
    print("         DATA QUALITY METRICS REPORT")
    print("="*40)
    print(f"Master Dataset Shape: {master_df.shape}")
    print(f"Total Observations (Rows): {len(master_df)}")
    print(f"Total Features (Cols): {len(master_df.columns)}")
    
    min_year, max_year = master_df["Year"].min(), master_df["Year"].max()
    min_m = master_df.loc[master_df["Year"] == min_year, "Month"].min()
    max_m = master_df.loc[master_df["Year"] == max_year, "Month"].max()
    print(f"Chronological Range: {min_year}-{min_m:02d} to {max_year}-{max_m:02d}")
    
    # Feature columns (everything except Year and Month)
    feature_cols = master_df.columns[2:]
    
    # Average missingness
    avg_missing = master_df[feature_cols].isna().mean().mean() * 100
    print(f"Average Missingness (NaN rate) Across Features: {avg_missing:.2f}%")
    
    # Top missing columns
    missing_rates = master_df[feature_cols].isna().mean() * 100
    print("\nTop 5 Columns with Highest Missing Rates:")
    print(missing_rates.sort_values(ascending=False).head(5).to_string())
    
    # Validate feature data types
    dtypes = master_df[feature_cols].dtypes
    non_floats = [col for col, dtype in dtypes.items() if not np.issubdtype(dtype, np.floating)]
    if non_floats:
        print(f"\nWARNING: {len(non_floats)} non-float feature columns detected: {non_floats}")
    else:
        print("\nSuccess: All feature columns cast to float64.")
        
    # Zero variance features check
    zero_var_cols = [col for col in feature_cols if master_df[col].nunique(dropna=True) <= 1]
    if zero_var_cols:
        print(f"WARNING: Zero-variance (constant) features found: {zero_var_cols}")
    else:
        print("Success: Zero-variance check passed (no constant feature columns).")
        
    # 8. Export master dataset to CSV
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMaster dataset exported successfully to:\n  '{OUTPUT_FILE}'")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
