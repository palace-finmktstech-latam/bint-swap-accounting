"""
Utility functions and constants for the Accounting Interface Validator
"""

# Product code mappings
PRODUCT_MAPPING = {
    'SWAP_TASA': 'Swap Tasa',
    'SWAP_MONE': 'Swap Moneda',
    'SWAP_ICP': 'Swap Cámara'
}

# Event code mappings
EVENT_MAPPING = {
    'INICIO': 'Curse',
    'VALORIZACION': 'Valorización MTM',
    'VENCIMIENTO': 'Vcto',
    'TERMINO': 'Vcto',
    'REVERSA_VALORIZACION': 'Reversa Valorización MTM'
}

# Subproduct mappings for day trades
SUBPRODUCT_MAPPING = {
    'Moneda': 'Swap Moneda',
    'Tasa': 'Swap Tasa',
    'Cámara': 'Swap ICP'
}

def normalize_deal_number(deal_number):
    """
    Normalize deal numbers to integers for consistent comparison.
    Handles cases where deal numbers might be floats like 4096.0
    """
    try:
        return int(float(str(deal_number)))
    except (ValueError, TypeError):
        return None

def find_column_by_keywords(df, keywords):
    """
    Find a column in a DataFrame by searching for keywords in column names.
    
    Args:
        df: DataFrame to search
        keywords: List of keywords to search for (case insensitive)
    
    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword.lower() in col_str for keyword in keywords):
            return col
    return None

def format_dataframe_for_display(df, numeric_precision=2):
    """
    Format a DataFrame for consistent display in Streamlit.
    
    Args:
        df: DataFrame to format
        numeric_precision: Number of decimal places for numeric columns
    
    Returns:
        Formatted DataFrame
    """
    df_copy = df.copy()
    
    # Convert object columns to strings for Arrow compatibility
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
    
    # Round numeric columns
    numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].round(numeric_precision)
    
    return df_copy

def calculate_match_statistics(validation_df):
    """
    Calculate match statistics from a validation results DataFrame.
    
    Args:
        validation_df: DataFrame with validation results containing 'status' column
    
    Returns:
        Dictionary with match statistics
    """
    if len(validation_df) == 0:
        return {
            'total_count': 0,
            'full_match_count': 0,
            'partial_match_count': 0,
            'no_match_count': 0,
            'match_percentage': 0.0
        }
    
    status_counts = validation_df['status'].value_counts()
    full_match_count = status_counts.get('Full Match', 0)
    partial_match_count = status_counts.get('Partial Match', 0)
    no_match_count = status_counts.get('No Match', 0)
    total_count = len(validation_df)
    
    match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
    
    return {
        'total_count': total_count,
        'full_match_count': full_match_count,
        'partial_match_count': partial_match_count,
        'no_match_count': no_match_count,
        'match_percentage': match_percentage,
        'status_counts': status_counts.to_dict()
    }

def get_trade_number_column(df):
    """
    Find the trade number column in a DataFrame.
    
    Args:
        df: DataFrame to search
    
    Returns:
        Column name if found, None otherwise
    """
    keywords = ['operación', 'operacion', 'nro.', 'número operación']
    return find_column_by_keywords(df, keywords)

def validate_required_columns(df, required_columns, file_description="file"):
    """
    Validate that required columns exist in a DataFrame.
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        file_description: Description of the file for error messages
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns 