import streamlit as st
import pandas as pd
import numpy as np
import re

def parse_interface_file(file):
    """Parse the accounting interface Excel file."""
    
    # Get file name for logging
    file_name = getattr(file, 'name', 'unknown')
    st.write(f"Attempting to parse Interface file: {file_name}")
    
    # Try to read with different engines
    df_preview = None
    df = None
    parsing_engine = None
    
    engines_to_try = ['openpyxl', 'xlrd']
    
    for engine in engines_to_try:
        try:
            st.write(f"Trying to read Interface file with {engine} engine...")
            
            # Read the Excel file with headers on row 11 (index 10) for preview
            df_preview = pd.read_excel(file, header=10, nrows=15, engine=engine)
            
            # Reset file pointer and read full data
            file.seek(0)
            df = pd.read_excel(file, header=10, engine=engine)
            
            parsing_engine = engine
            st.success(f"✅ Successfully parsed Interface file using {engine} engine")
            break
            
        except Exception as e:
            st.warning(f"Failed with {engine} engine: {str(e)}")
            file.seek(0)  # Reset file pointer for next attempt
            continue
    
    # If all engines failed
    if df is None or df_preview is None:
        st.error("❌ Could not parse the Interface file with any Excel engine")
        st.error("Please ensure the file is a valid Excel (.xlsx/.xls) file")
        return pd.DataFrame(), {}
    
    # Convert all columns to string before displaying
    df_preview = df_preview.astype(str)
    
    # Display preview
    st.write("Interface file preview (first 15 rows):")
    st.dataframe(df_preview)
    
    # Remove the last two rows as these are just totals
    df = df.iloc[:-2]
    st.write("Interface file columns:", df.columns.tolist())
    
    # Identify key columns
    key_columns = {}
    for col in df.columns:
        col_str = str(col).lower()
        if 'glosa' in col_str:
            key_columns['glosa'] = col
        elif 'cuenta' in col_str:
            key_columns['account'] = col
        elif 'debe' in col_str:
            key_columns['debit'] = col
        elif 'haber' in col_str:
            key_columns['credit'] = col
        elif 'voucher' in col_str or 'n°' in col_str or 'id' in col_str:
            key_columns['id'] = col
    
    # Display identified columns
    st.write("Identified key columns:", key_columns)
    
    # Extract event types from Glosa
    if 'glosa' in key_columns:
        glosa_col = key_columns['glosa']
        # Convert to string to ensure extraction works
        df[glosa_col] = df[glosa_col].astype(str)
        
        # Extract event type - Updated to handle both VENCIMIENTO and TERMINO
        df['event_type'] = df[glosa_col].apply(lambda x: 
            'Valorización MTM' if 'Valorización' in x or 'MTM' in x 
            else ('Curse' if 'Curse' in x 
            else ('Vcto' if 'Vcto' in x or 'Vencimiento' in x  # Both VENCIMIENTO and TERMINO use Vcto/Vencimiento glosa
            else ('Reversa Valorización MTM' if 'Reversa' in x or 'Reverso' in x 
            else None))))
        
        # Note: TERMINO entries will also be classified as 'Vcto' since they share the same glosa
        # The validation logic will separate them by matching against different rule accounts
        
        # Extract instrument type
        df['instrument_type'] = df[glosa_col].apply(lambda x: 
            'Swap Tasa' if 'Swap Tasa' in x 
            else ('Swap Moneda' if 'Swap Moneda' in x 
            else ('Swap Cámara' if 'Swap Cámara' in x else None)))
        
        # Extract currency pair using regex pattern matching
        def extract_currency_pair(glosa):
            # Look for pattern like "XXX-YYY" where XXX and YYY are typically 2-3 letters
            pattern = r'\b([A-Z]{2,4})-([A-Z]{2,4})\b'
            match = re.search(pattern, glosa)
            if match:
                return match.group(0)  # Return the full match (e.g., "CHF-UF")
            return None
        
        # Apply the function to extract currency pairs
        df['currency_pair'] = df[glosa_col].apply(extract_currency_pair)
        
        # Show counts
        event_counts = df['event_type'].value_counts(dropna=False)
        instrument_counts = df['instrument_type'].value_counts(dropna=False)
        currency_counts = df['currency_pair'].value_counts(dropna=False)
        
        st.write("Event type counts:", event_counts.to_dict())
        st.write("Instrument type counts:", instrument_counts.to_dict())
        st.write("Currency pair counts:", currency_counts.to_dict())
        
        # Show additional info about Vcto entries
        vcto_count = len(df[df['event_type'] == 'Vcto'])
        if vcto_count > 0:
            st.info(f"ℹ️ Found {vcto_count} 'Vcto' entries (includes both VENCIMIENTO and TERMINO entries)")
    
    return df, key_columns

def parse_mtm_file(file):
    """Parse the MTM Excel file and sum MTM values by deal."""
    
    # Get file extension to determine parsing approach
    file_name = getattr(file, 'name', 'unknown')
    file_extension = file_name.lower().split('.')[-1] if '.' in file_name else 'unknown'
    
    st.write(f"Attempting to parse MTM file: {file_name}")
    
    df = None
    parsing_method = None
    
    # Try different parsing methods based on file extension and fallbacks
    if file_extension in ['xlsx', 'xls']:
        # Try Excel parsing with different engines
        engines_to_try = ['openpyxl', 'xlrd'] if file_extension == 'xlsx' else ['xlrd', 'openpyxl']
        
        for engine in engines_to_try:
            try:
                st.write(f"Trying to read as Excel with {engine} engine...")
                df = pd.read_excel(file, engine=engine)
                parsing_method = f"Excel ({engine})"
                st.success(f"✅ Successfully parsed as Excel using {engine} engine")
                break
            except Exception as e:
                st.warning(f"Failed with {engine} engine: {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
                continue
    
    # If Excel parsing failed or file extension suggests CSV, try CSV parsing
    if df is None:
        try:
            st.write("Trying to read as CSV...")
            file.seek(0)  # Reset file pointer
            
            # First, try to detect delimiter by reading a sample
            sample = file.read(4096)
            file.seek(0)
            
            # Try to decode the sample
            try:
                sample_str = sample.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    sample_str = sample.decode('latin1')
                except UnicodeDecodeError:
                    sample_str = sample.decode('utf-8', errors='ignore')
            
            # Detect delimiter
            if ';' in sample_str:
                delimiter = ';'
            elif ',' in sample_str:
                delimiter = ','
            elif '\t' in sample_str:
                delimiter = '\t'
            else:
                delimiter = ','  # Default fallback
            
            st.write(f"Detected delimiter: '{delimiter}'")
            
            # Try different encodings
            encodings_to_try = ['utf-8', 'latin1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding)
                    parsing_method = f"CSV (delimiter='{delimiter}', encoding='{encoding}')"
                    st.success(f"✅ Successfully parsed as CSV with {encoding} encoding")
                    break
                except Exception as e:
                    st.warning(f"Failed with {encoding} encoding: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"CSV parsing also failed: {str(e)}")
    
    # If all parsing methods failed
    if df is None:
        st.error("❌ Could not parse the MTM file with any method")
        st.error("Please ensure the file is a valid Excel (.xlsx/.xls) or CSV file")
        return pd.DataFrame(), pd.DataFrame()
    
    st.write(f"✅ File successfully parsed using: {parsing_method}")
    st.write("MTM file columns:", df.columns.tolist())
    
    # Map product codes to instrument types
    product_mapping = {
        'SWAP_TASA': 'Swap Tasa',
        'SWAP_MONE': 'Swap Moneda',
        'SWAP_ICP': 'Swap Cámara'
    }
    
    # Find product column
    product_col = next((col for col in df.columns if 'product' in str(col).lower()), None)
    if not product_col:
        st.error("Could not find product column in MTM file")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter out unknown product codes and report them
    valid_products = set(product_mapping.keys())
    unknown_products = set(df[product_col].unique()) - valid_products
    
    if unknown_products:
        st.warning(f"Found {len(unknown_products)} unknown product codes: {', '.join(map(str, unknown_products))}")
        st.write("These products will be excluded from the analysis")
    
    # Filter dataframe to only include known products
    df = df[df[product_col].isin(valid_products)].copy()
    
    if len(df) == 0:
        st.error("No valid products found in MTM file")
        return pd.DataFrame(), pd.DataFrame()
    
    # Add instrument type column
    df['instrument_type'] = df[product_col].map(product_mapping)
    
    # Find MTM value column
    mtm_col = next((col for col in df.columns if 'm2m_clp' in str(col).lower()), None)
    if not mtm_col:
        st.error("Could not find MTM value column in the MTM file")
        return df, pd.DataFrame()
    
    # Find deal number column
    deal_col = next((col for col in df.columns if 'deal' in str(col).lower() and 'number' in str(col).lower()), None)
    if not deal_col:
        st.error("Could not find deal number column in the MTM file")
        return df, pd.DataFrame()
    
    # Convert MTM values to numeric
    df[mtm_col] = pd.to_numeric(df[mtm_col], errors='coerce')
    
    # Sum MTM values by deal number
    mtm_sums = df.groupby(deal_col)[mtm_col].sum().reset_index()
    mtm_sums.rename(columns={deal_col: 'deal_number', mtm_col: 'total_mtm'}, inplace=True)
    
    # Round total_mtm to 0 decimal places (integers)
    mtm_sums['total_mtm'] = mtm_sums['total_mtm'].round(0)
    
    # Add direction column
    mtm_sums['direction'] = mtm_sums['total_mtm'].apply(
        lambda x: 'POSITIVO' if x > 0 else 'NEGATIVO')
    
    # Display summary
    st.write(f"Found {len(mtm_sums)} unique deals")
    
    # Convert to string/dict before displaying to avoid Arrow issues
    direction_counts = mtm_sums['direction'].value_counts()
    st.write("MTM direction counts:", direction_counts.to_dict())
    
    # Convert to strings for display
    mtm_sums_display = mtm_sums.head().copy()
    mtm_sums_display['deal_number'] = mtm_sums_display['deal_number'].astype(str)
    st.write("Sample of MTM sums:")
    st.dataframe(mtm_sums_display)
    
    return df, mtm_sums

def parse_day_trades_file(file):
    """Parse the day trades CSV file."""
    # First, try to inspect the file content to determine the format
    file.seek(0)  # Reset file pointer to beginning
    sample = file.read(4096).decode('latin1')  # Read a sample to inspect
    file.seek(0)  # Reset file pointer again
    
    st.write("Sample of file content:")
    st.text(sample[:500])  # Show first 500 characters
    
    # Try to detect the delimiter
    if ';' in sample:
        delimiter = ';'
    elif ',' in sample:
        delimiter = ','
    elif '\t' in sample:
        delimiter = '\t'
    else:
        delimiter = None
    
    st.write(f"Detected delimiter: {delimiter}")
    
    # Try different parsing approaches
    try:
        # Try with detected delimiter
        if delimiter:
            df = pd.read_csv(file, delimiter=delimiter, encoding='latin1')
        else:
            # If no delimiter detected, try with default
            df = pd.read_csv(file, encoding='latin1')
    except Exception as e:
        st.error(f"First parsing attempt failed: {str(e)}")
        
        # Reset file pointer
        file.seek(0)
        
        try:
            # Try with excel-like parsing
            df = pd.read_csv(file, encoding='latin1', sep=None, engine='python')
        except Exception as e:
            st.error(f"Second parsing attempt failed: {str(e)}")
            return pd.DataFrame()
    
    # Check if we got any data
    if len(df.columns) <= 1:
        st.error("Failed to properly parse columns from the CSV file")
        st.write("Found columns:", df.columns.tolist())
        return pd.DataFrame()
    
    st.write("Day trades file columns:", df.columns.tolist())
    
    # Check if the required columns exist
    required_columns = ['Número Operación', 'Producto', 'Subproducto', 'Moneda Activa', 'Monto Activo', 'Moneda Pasiva', 'Monto Pasivo', 'Cobertura']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in day trades file: {', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # Convert numeric columns to appropriate types
    df['Número Operación'] = pd.to_numeric(df['Número Operación'], errors='coerce')
    df['Monto Activo'] = pd.to_numeric(df['Monto Activo'], errors='coerce')
    df['Monto Pasivo'] = pd.to_numeric(df['Monto Pasivo'], errors='coerce')

    # Map subproduct codes to instrument types
    subproduct_mapping = {
        'Moneda': 'Swap Moneda',
        'Tasa': 'Swap Tasa',
        'Cámara': 'Swap Cámara'
    }
    
    # Create instrument_type column based on Subproducto
    df['instrument_type'] = df['Subproducto'].map(subproduct_mapping)
    
    # Display summary information
    st.write(f"Found {len(df)} day trades")
    
    # Show product distribution
    product_counts = df['Producto'].value_counts()
    st.write("Product counts:", product_counts.to_dict())
    
    # Show subproduct distribution
    subproduct_counts = df['Subproducto'].value_counts()
    st.write("Subproduct counts:", subproduct_counts.to_dict())
    
    # Show instrument type mapping results
    instrument_counts = df['instrument_type'].value_counts(dropna=False)
    st.write("Instrument type counts:", instrument_counts.to_dict())
    
    # Display preview of the parsed data
    st.subheader("Day Trades Preview")
    preview_columns = ['Número Operación', 'Producto', 'Subproducto', 'instrument_type', 'Cobertura',
                     'Moneda Activa', 'Monto Activo', 'Moneda Pasiva', 'Monto Pasivo']
    st.dataframe(df[preview_columns], use_container_width=True)
    
    return df

def parse_expiries_file(file):
    """Parse the expiries Excel file."""
    try:
        # Get file name for logging
        file_name = getattr(file, 'name', 'unknown')
        st.write(f"Attempting to parse Expiries file: {file_name}")
        
        # Try different engines
        engines_to_try = ['openpyxl', 'xlrd']
        df_preview = None
        df = None
        
        for engine in engines_to_try:
            try:
                st.write(f"Trying to read Expiries file with {engine} engine...")
                
                # First, let's look at a preview to identify the header row
                df_preview = pd.read_excel(file, header=None, nrows=15, engine=engine)
                
                # Reset file pointer and read with header on row 10
                file.seek(0)
                df = pd.read_excel(file, header=10, engine=engine)
                
                st.success(f"✅ Successfully parsed Expiries file using {engine} engine")
                break
                
            except Exception as e:
                st.warning(f"Failed with {engine} engine: {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
                continue
        
        # If all engines failed
        if df is None or df_preview is None:
            st.error("❌ Could not parse the Expiries file with any Excel engine")
            return pd.DataFrame()
        
        # Convert to string for display
        df_preview_display = df_preview.astype(str)
        st.write("Expiries file preview (first 15 rows):")
        st.dataframe(df_preview_display)
        
        # Show columns found
        st.write("Expiries file columns:", df.columns.tolist())
        
        # Check if the required columns exist
        required_columns = [
            'Número Operación', 'Fecha Vencimiento', 'Fecha Pago Capital', 
            'Fecha Pago Interés', 'Moneda Liquidación', 
            'Monto Override Extranjero', 'Monto Override Local', 'Amortización Activa', 'Amortización Pasiva'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in expiries file: {', '.join(missing_columns)}")
            
            # Show available columns for debugging
            st.write("Available columns:", df.columns.tolist())
            
            # Try to find similar column names
            for missing_col in missing_columns:
                possible_matches = [col for col in df.columns if missing_col.lower() in col.lower()]
                if possible_matches:
                    st.write(f"Possible matches for '{missing_col}': {possible_matches}")
            
            return pd.DataFrame()
        
        # Convert date columns to datetime with specific format (DD/MM/YYYY)
        date_columns = ['Fecha Vencimiento', 'Fecha Pago Capital', 'Fecha Pago Interés']
        for col in date_columns:
            # First, convert the column to strings to handle any non-standard formats
            df[col] = df[col].astype(str)
            
            # Then parse with format parameter to handle DD/MM/YYYY
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            except Exception as e:
                st.warning(f"Error converting {col} to datetime: {str(e)}")
                # Show some sample values to assist in debugging
                st.write(f"Sample values from {col}:", df[col].head().tolist())
        
        # Convert numeric columns to appropriate types
        df['Número Operación'] = pd.to_numeric(df['Número Operación'], errors='coerce')
        df['Monto Override Extranjero'] = pd.to_numeric(df['Monto Override Extranjero'], errors='coerce')
        df['Monto Override Local'] = pd.to_numeric(df['Monto Override Local'], errors='coerce')
        df['Amortización Activa'] = pd.to_numeric(df['Amortización Activa'], errors='coerce')
        df['Amortización Pasiva'] = pd.to_numeric(df['Amortización Pasiva'], errors='coerce')
        
        # Map product names to standard instrument types
        if 'Producto' in df.columns:
            # Create mapping for instrument types
            product_mapping = {
                'Swap Moneda': 'Swap Moneda',
                'Swap Tasa': 'Swap Tasa',
                'Swap Promedio Cámara': 'Swap Cámara'
            }
            
            # Create instrument_type column based on Producto
            df['instrument_type'] = df['Producto'].map(product_mapping)
            
            # Handle any unmapped products
            unmapped_products = df[~df['Producto'].isin(product_mapping.keys())]['Producto'].unique()
            if len(unmapped_products) > 0:
                st.warning(f"Found unmapped products: {unmapped_products}")
                
                # Make a best guess for unmapped products
                for product in unmapped_products:
                    if isinstance(product, str):  # Make sure it's a string before checking
                        if 'moneda' in product.lower():
                            df.loc[df['Producto'] == product, 'instrument_type'] = 'Swap Moneda'
                        elif 'tasa' in product.lower():
                            df.loc[df['Producto'] == product, 'instrument_type'] = 'Swap Tasa'
                        elif 'cámara' in product.lower() or 'camara' in product.lower():
                            df.loc[df['Producto'] == product, 'instrument_type'] = 'Swap Cámara'
        else:
            st.warning("'Producto' column not found, can't map to instrument types")
            df['instrument_type'] = None
        
        # Display summary information
        st.write(f"Found {len(df)} expiring trades")
        
        # Show payment date summary
        today = pd.Timestamp.now().normalize()
        
        # Format dates for display
        df_display = df.copy()
        for col in date_columns:
            if col in df_display.columns:
                df_display[f'{col}_formatted'] = df_display[col].dt.strftime('%d/%m/%Y')
        
        # Count items with payment dates today
        capital_today = sum(df['Fecha Pago Capital'].dt.normalize() == today)
        interest_today = sum(df['Fecha Pago Interés'].dt.normalize() == today)
        
        st.write(f"Capital payments due today: {capital_today}")
        st.write(f"Interest payments due today: {interest_today}")
        
        # Show currency distribution
        if 'Moneda Liquidación' in df.columns:
            currency_counts = df['Moneda Liquidación'].value_counts()
            st.write("Settlement currency counts:", currency_counts.to_dict())
        
        # Show instrument type distribution if available
        if 'instrument_type' in df.columns and df['instrument_type'].notna().any():
            instrument_counts = df['instrument_type'].value_counts(dropna=False)
            st.write("Instrument type counts:", instrument_counts.to_dict())
        
        # Display preview of the parsed data
        st.subheader("Expiries Preview")
        display_columns = [
            'Número Operación', 'Producto', 'instrument_type' if 'instrument_type' in df.columns else None,
            'Fecha Vencimiento', 'Fecha Pago Capital', 'Fecha Pago Interés',
            'Moneda Liquidación', 'Monto Override Extranjero', 'Monto Override Local'
        ]
        display_columns = [col for col in display_columns if col is not None and col in df.columns]
        
        st.dataframe(df[display_columns].head(10), use_container_width=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing expiries file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def parse_rules_file(file):
    """Parse the accounting rules Excel file."""
    
    # Get file name for logging
    file_name = getattr(file, 'name', 'unknown')
    st.write(f"Attempting to parse Rules file: {file_name}")
    
    # Try different engines and file formats
    engines_to_try = ['openpyxl', 'xlrd']
    df_preview = None
    df = None
    
    # First try Excel parsing
    for engine in engines_to_try:
        try:
            st.write(f"Trying to read Rules file as Excel with {engine} engine...")
            
            # Load preview
            df_preview = pd.read_excel(file, header=None, nrows=25, engine=engine)
            
            # Reset and load full data with header=0 (first row is header)
            file.seek(0)
            df = pd.read_excel(file, header=0, engine=engine)
            
            st.success(f"✅ Successfully parsed Rules file using {engine} engine")
            break
            
        except Exception as e:
            st.warning(f"Failed with {engine} engine: {str(e)}")
            file.seek(0)  # Reset file pointer for next attempt
            continue
    
    # If Excel failed, try CSV
    if df is None:
        try:
            st.write("Trying to read Rules file as CSV...")
            file.seek(0)
            
            # Detect delimiter
            sample = file.read(4096)
            file.seek(0)
            
            try:
                sample_str = sample.decode('utf-8')
            except UnicodeDecodeError:
                sample_str = sample.decode('latin1', errors='ignore')
            
            delimiter = ';' if ';' in sample_str else ','
            
            df = pd.read_csv(file, delimiter=delimiter, encoding='utf-8')
            df_preview = df.head(25)
            
            st.success(f"✅ Successfully parsed Rules file as CSV")
            
        except Exception as e:
            st.error(f"CSV parsing failed: {str(e)}")
    
    # If all parsing failed
    if df is None or df_preview is None:
        st.error("❌ Could not parse the Rules file with any method")
        return pd.DataFrame()
    
    # Process preview
    df_preview = df_preview.dropna(how='all')
    df_preview = df_preview.iloc[:, :14]
    
    # Convert to string before displaying
    df_preview = df_preview.astype(str)
    
    st.write("Rules file preview:")
    st.dataframe(df_preview)
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')

    # Remove the columns after number 14 as these aren't relevant
    df = df.iloc[:, :14]
        
    # Log the number of rows after removing empty rows
    st.write(f"Rules file has {len(df)} rows after removing empty rows")
    st.write("Rules file columns:", df.columns.tolist())
    
    # Identify key columns
    key_columns = {}
    for col in df.columns:
        col_str = str(col).lower()
        if 'sub' in col_str and 'producto' in col_str:
            key_columns['subproduct'] = col
        elif 'direcc' in col_str:
            key_columns['direction'] = col
        elif 'evento' in col_str:
            key_columns['event'] = col
        elif 'debe' in col_str:
            key_columns['debit_account'] = col
        elif 'haber' in col_str:
            key_columns['credit_account'] = col
        elif 'cobertura' in col_str:
            key_columns['coverage'] = col
        elif 'pata' in col_str:  # Added this line to recognize Pata column
            key_columns['pata'] = col
    
    st.write("Identified rule columns:", key_columns)
    
    # Rename columns to standardized names
    if key_columns:
        df = df.rename(columns={v: k for k, v in key_columns.items()})
    
    # Ensure we have the Pata column with proper name
    if 'pata' in df.columns:
        df = df.rename(columns={'pata': 'Pata'})
    elif 'Pata' not in df.columns:
        # If Pata column is missing, show available columns for debugging
        st.error("❌ Could not find 'Pata' column in rules file")
        st.write("Available columns after processing:", df.columns.tolist())
        st.write("Raw column names from file:", list(df.columns))
        return pd.DataFrame()
    
    # Map event codes
    if 'event' in df.columns:
        event_mapping = {
            'INICIO': 'Curse',
            'VALORIZACION': 'Valorización MTM',
            #'VENCIMIENTO': 'Vcto',
            'VENCIMIENTO': 'Vencimiento',
            #'TERMINO': 'Vcto',
            'TERMINO': 'Termino',
            'REVERSA_VALORIZACION': 'Reversa Valorización MTM'
        }
        df['event'] = df['event'].map(event_mapping)
    
    # Show event counts as dictionary not DataFrame to avoid Arrow issues
    if 'event' in df.columns:
        event_counts = df['event'].value_counts(dropna=False)
        st.write("Event counts in rules:", event_counts.to_dict())
    
    # Show Pata distribution
    if 'Pata' in df.columns:
        pata_counts = df['Pata'].value_counts(dropna=False)
        st.write("Pata distribution in rules:", pata_counts.to_dict())
    
    # Handle 'Swap All' by expanding to all swap types
    if 'subproduct' in df.columns:
        expanded_rules = []
        for _, rule in df.iterrows():
            if rule.get('subproduct') == 'Swap All':
                for swap_type in ['Swap Tasa', 'Swap Moneda', 'Swap Cámara']:
                    new_rule = rule.copy()
                    new_rule['subproduct'] = swap_type
                    expanded_rules.append(new_rule)
            else:
                expanded_rules.append(rule)
        
        df_expanded = pd.DataFrame(expanded_rules)
        
        # Final validation - ensure Pata column exists in expanded dataframe
        if 'Pata' not in df_expanded.columns:
            st.error("❌ Pata column missing after expansion")
            st.write("Columns in expanded dataframe:", df_expanded.columns.tolist())
            return pd.DataFrame()
            
        return df_expanded
    
    return df