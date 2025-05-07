import streamlit as st
st.set_page_config(
    page_title="Accounting Interface Validator",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Accounting Interface Validator for Bank Internacional"
    }
)

import pandas as pd
import numpy as np

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
    required_columns = ['Número Operación', 'Producto', 'Subproducto', 'Moneda Activa', 'Monto Activo']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in day trades file: {', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # Convert numeric columns to appropriate types
    df['Número Operación'] = pd.to_numeric(df['Número Operación'], errors='coerce')
    df['Monto Activo'] = pd.to_numeric(df['Monto Activo'], errors='coerce')
    
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
    preview_columns = ['Número Operación', 'Producto', 'Subproducto', 'instrument_type', 
                     'Moneda Activa', 'Monto Activo']
    st.dataframe(df[preview_columns], use_container_width=True)
    
    return df

def parse_interface_file(file):
    """Parse the accounting interface Excel file."""
    # Read the Excel file with headers on row 11 (index 10)
    df_preview = pd.read_excel(file, header=10, nrows=15)
    
    # Convert all columns to string before displaying
    df_preview = df_preview.astype(str)
    
    # Display preview
    st.write("Interface file preview (first 15 rows):")
    st.dataframe(df_preview)
    
    # Load with proper header row
    df = pd.read_excel(file, header=10)
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
        
        # Extract event type
        df['event_type'] = df[glosa_col].apply(lambda x: 
            'Valorización MTM' if 'Valorización' in x or 'MTM' in x 
            else ('Curse' if 'Curse' in x 
            else ('Vcto' if 'Vcto' in x 
            else ('Reversa Valorización MTM' if 'Reversa' in x or 'Reverso' in x 
            else None))))
        
        # Extract instrument type
        df['instrument_type'] = df[glosa_col].apply(lambda x: 
            'Swap Tasa' if 'Swap Tasa' in x 
            else ('Swap Moneda' if 'Swap Moneda' in x 
            else ('Swap Cámara' if 'Swap Cámara' in x else None)))
        
        # Extract currency pair using regex pattern matching
        import re
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
    
    return df, key_columns

def parse_mtm_file(file):
    """Parse the MTM Excel file and sum MTM values by deal."""
    df = pd.read_excel(file)
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

def parse_rules_file(file):
    """Parse the accounting rules Excel file."""
    # Load with headers on row 0 (first row)
    df_preview = pd.read_excel(file, header=None, nrows=25)
    df_preview = df_preview.dropna(how='all')
    df_preview = df_preview.iloc[:, :14]
    
    # Convert to string before displaying
    df_preview = df_preview.astype(str)
    
    st.write("Rules file preview:")
    st.dataframe(df_preview)
    
    # Since data now starts in A1, use header=0 (first row is header)
    df = pd.read_excel(file, header=0)
    
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
    
    st.write("Identified rule columns:", key_columns)
    
    # Rename columns to standardized names
    if key_columns:
        df = df.rename(columns={v: k for k, v in key_columns.items()})
    
    # Map event codes
    if 'event' in df.columns:
        event_mapping = {
            'INICIO': 'Curse',
            'VALORIZACION': 'Valorización MTM',
            'VENCIMIENTO': 'Vcto',
            'TERMINO': 'Vcto',
            'REVERSA_VALORIZACION': 'Reversa Valorización MTM'
        }
        df['event'] = df['event'].map(event_mapping)
    
    # Show event counts as dictionary not DataFrame to avoid Arrow issues
    if 'event' in df.columns:
        event_counts = df['event'].value_counts(dropna=False)
        st.write("Event counts in rules:", event_counts.to_dict())
    
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
        return df_expanded
    
    return df

def validate_day_trades(day_trades_df, interface_df, interface_cols, rules_df, debug_deal=None):
    """
    Validate day trades against accounting interface entries.
    
    For each day trade, checks that:
    1. Corresponding "Curse" entries exist in the accounting interface
    2. Correct account numbers are used based on instrument type
    3. Transaction amounts match the Monto Activo value
    4. Amounts are in the correct fields (debe/haber)
    5. The number of entries exactly matches what's expected from rules
    """
    # Filter rules for INICIO event
    inicio_rules = rules_df[rules_df['event'] == 'Curse'].copy()
    
    if len(inicio_rules) == 0:
        st.error("No INICIO/Curse rules found in rules file")
        return pd.DataFrame()
    
    st.subheader("INICIO/Curse Rules")
    st.dataframe(inicio_rules, use_container_width=True, hide_index=True)
    
    # Extract needed columns from interface
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    glosa_col = interface_cols['glosa']
    
    # Filter interface for "Curse" event_type entries
    curse_entries = interface_df[interface_df['event_type'] == 'Curse'].copy()
    st.write(f"Found {len(curse_entries)} Curse entries in interface file")
    
    # Ensure numeric values
    curse_entries[debit_col] = pd.to_numeric(curse_entries[debit_col], errors='coerce').fillna(0)
    curse_entries[credit_col] = pd.to_numeric(curse_entries[credit_col], errors='coerce').fillna(0)
    
    # Prepare validation results
    validation_results = []
    
    # Process each day trade
    for _, trade in day_trades_df.iterrows():
        trade_number = trade['Número Operación']
        
        # Skip if not the debug deal (when in debug mode)
        if debug_deal is not None and str(trade_number) != str(debug_deal):
            continue
            
        instrument_type = trade['instrument_type']
        monto_activo = trade['Monto Activo']
        currency = trade['Moneda Activa']
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG: Processing trade {trade_number}, instrument: {instrument_type}, amount: {monto_activo}, currency: {currency}")
        
        # Get applicable rules for this instrument type
        applicable_rules = inicio_rules[inicio_rules['subproduct'] == instrument_type]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'currency': currency,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No rule found for {instrument_type}'
            })
            continue
            
        # Count how many unique accounts are expected
        expected_accounts = []
        for _, rule in applicable_rules.iterrows():
            if pd.notna(rule['debit_account']):
                expected_accounts.append(str(rule['debit_account']))
            if pd.notna(rule['credit_account']):
                expected_accounts.append(str(rule['credit_account']))

        # Remove any "None" values
        expected_accounts = [acc for acc in expected_accounts if acc != "None" and acc != "nan"]
        expected_entry_count = len(expected_accounts)
            
        # Find interface entries for this trade
        # Filter by trade number and ensure the glosa contains "Curse"
        trade_entries = curse_entries[
            (curse_entries[trade_number_col] == trade_number)
        ]
        
        if debug_deal is not None:
            st.write(f"Found {len(trade_entries)} entries for trade {trade_number} in interface")
            st.write(f"Expected {expected_entry_count} entries based on rules")
            st.write(f"Expected account numbers: {expected_accounts}")
            st.dataframe(trade_entries, use_container_width=True, hide_index=True)
        
        if len(trade_entries) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'currency': currency,
                'status': 'Missing Entries',
                'interface_entries': 0,
                'expected_entries': expected_entry_count,
                'issue': f'No Curse entries found in interface for trade {trade_number}'
            })
            continue
        
        # Check if we found exactly the right number of entries
        if len(trade_entries) != expected_entry_count:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'currency': currency,
                'status': 'Entry Count Mismatch',
                'interface_entries': len(trade_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Expected {expected_entry_count} entries, found {len(trade_entries)}'
            })
            continue
            
        # Check if each expected account is present
        found_accounts = trade_entries[account_col].astype(str).unique().tolist()
        missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
        extra_accounts = [acc for acc in found_accounts if acc not in expected_accounts]

        if missing_accounts:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'currency': currency,
                'status': 'Missing Accounts',
                'interface_entries': len(trade_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Missing expected accounts: {", ".join(missing_accounts)}'
            })
            continue

        if extra_accounts:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'currency': currency,
                'status': 'Extra Accounts',
                'interface_entries': len(trade_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Found unexpected accounts: {", ".join(extra_accounts)}'
            })
            continue
        
        # At this point we have the right number of entries with the right accounts
        # Now check the amounts in each entry
        
        # Create a dictionary to track amount validation for each account
        account_validation = {}
        for account in expected_accounts:
            account_entries = trade_entries[trade_entries[account_col].astype(str) == account]
            
            # Determine if this is a debit or credit account based on rules
            is_debit_account = any(
                pd.notna(rule['debit_account']) and str(rule['debit_account']) == account 
                for _, rule in applicable_rules.iterrows()
            )
            
            is_credit_account = any(
                pd.notna(rule['credit_account']) and str(rule['credit_account']) == account 
                for _, rule in applicable_rules.iterrows()
            )
            
            # Get the total amount in the expected field
            if is_debit_account:
                amount = account_entries[debit_col].sum()
                expected_field = 'debit'
            elif is_credit_account:
                amount = account_entries[credit_col].sum()  
                expected_field = 'credit'
            else:
                amount = 0
                expected_field = 'unknown'
            
            # Check if amount matches the expected value
            is_matching = abs(amount - monto_activo) < 1.0
            
            account_validation[account] = {
                'expected_field': expected_field,
                'amount': amount,
                'matches': is_matching
            }
            
            if debug_deal is not None:
                if is_matching:
                    st.success(f"✓ Account {account} ({expected_field}): Expected {monto_activo}, Found {amount:.2f}")
                else:
                    st.warning(f"✗ Account {account} ({expected_field}): Expected {monto_activo}, Found {amount:.2f}")
        
        # Check if all amounts match
        all_match = all(val['matches'] for val in account_validation.values())
        
        if all_match:
            status = 'Full Match'
            issue = ''
        else:
            status = 'Amount Mismatch'
            mismatches = [
                f"{account} ({val['expected_field']}): Expected {monto_activo}, Found {val['amount']:.2f}"
                for account, val in account_validation.items() if not val['matches']
            ]
            issue = f"Amount mismatches: {'; '.join(mismatches)}"
        
        # Add to results
        validation_results.append({
            'trade_number': str(trade_number),
            'instrument_type': instrument_type,
            'monto_activo': monto_activo,
            'currency': currency,
            'status': status,
            'interface_entries': len(trade_entries),
            'expected_entries': expected_entry_count,
            'issue': issue
        })
    
    # Create validation results dataframe
    validation_df = pd.DataFrame(validation_results)
    
    if len(validation_df) > 0:
        # Format for display
        for col in validation_df.columns:
            if validation_df[col].dtype == 'object':
                validation_df[col] = validation_df[col].astype(str)
        
        # Round numeric columns
        numeric_cols = validation_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            validation_df[col] = validation_df[col].round(2)
        
        # Display validation results
        st.subheader("Day Trades Validation Results")
        st.write("Validates that trades in the day trades file have corresponding Curse entries in the accounting interface")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Calculate match statistics
        full_match_count = len(validation_df[validation_df['status'] == 'Full Match'])
        total_count = len(validation_df)
        match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_count)
        with col2:
            st.metric("Full Matches", full_match_count)
        with col3:
            st.metric("Match Rate", f"{match_percentage:.1f}%")
        
        # Status breakdown
        status_counts = validation_df['status'].value_counts().to_dict()
        st.subheader("Status Breakdown")
        st.write(status_counts)
        
        # Download results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            "Download Day Trades Validation Results",
            csv,
            "day_trades_validation_results.csv",
            "text/csv",
            key="download-csv-day-trades"
        )
    else:
        st.warning("No day trades validation results generated")
    
    return validation_df

def validate_mtm_entries(interface_df, interface_cols, mtm_df, mtm_sums, rules_df, 
                          event_type='Valorización MTM', 
                          rules_event_type=None,
                          key_suffix='', 
                          debug_deal=None):
    
    # Use rules_event_type if provided, otherwise default to event_type
    filter_event = rules_event_type if rules_event_type else event_type
    
    # Filter rules using the appropriate event type
    mtm_rules = rules_df[rules_df['event'] == filter_event].copy()
    
    """
    Validate MTM entries against expected entries from rules:
    1. Identify expected entries from rules
    2. Get MTM values from MTM file
    3. Match with accounting interface entries
    4. Report on full, partial or non-matches
    """
    if debug_deal is not None:
        st.write(f"DEBUG: Analyzing deal number {debug_deal}")
    
    # Filter rules for specified event type
    #mtm_rules = rules_df[rules_df['event'] == event_type].copy()
    st.write(f"Found {len(mtm_rules)} {event_type} rules")
    
    # Display MTM rules
    st.subheader(f"{event_type} Rules")
    st.dataframe(mtm_rules, use_container_width=True, hide_index=True)
    
    if len(mtm_rules) == 0:
        st.error(f"No {event_type} rules found in rules file")
        return pd.DataFrame()
    
    # Find required columns
    product_col = next((col for col in mtm_df.columns if 'product' in str(col).lower()), None)
    mtm_col = next((col for col in mtm_df.columns if 'm2m_clp' in str(col).lower()), None)
    deal_col = next((col for col in mtm_df.columns if 'deal' in str(col).lower() and 'number' in str(col).lower()), None)
    
    # Find trade number column in interface file
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    
    if not all([product_col, mtm_col, deal_col]):
        st.error("Could not find required columns in MTM file")
        return pd.DataFrame()
    
    if not trade_number_col:
        st.warning("Could not find trade number column in interface file")
        return pd.DataFrame()
    
    # Create product code mapping
    product_mapping = {
        'SWAP_TASA': 'Swap Tasa',
        'SWAP_MONE': 'Swap Moneda',
        'SWAP_ICP': 'Swap Cámara'
    }
    product_mapping_reverse = {v: k for k, v in product_mapping.items()}
    
    # Extract account interface entries for specified event type
    mtm_entries = interface_df[interface_df['event_type'] == event_type].copy()
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    
    # Ensure numeric values in MTM file and interface
    mtm_df[mtm_col] = pd.to_numeric(mtm_df[mtm_col], errors='coerce')
    mtm_entries[debit_col] = pd.to_numeric(mtm_entries[debit_col], errors='coerce').fillna(0)
    mtm_entries[credit_col] = pd.to_numeric(mtm_entries[credit_col], errors='coerce').fillna(0)
    
    st.write(f"Found {len(mtm_entries)} {event_type} entries in interface file")
    
    # Initialize validation results
    validation_results = []
    
    # Process each deal from MTM file
    for _, deal_row in mtm_sums.iterrows():
        deal_number = deal_row['deal_number']
        
        # Normalize deal numbers to integers for comparison
        normalized_deal = int(float(str(deal_number)))
        normalized_debug = int(float(str(debug_deal))) if debug_deal is not None else None
        
        # Skip if not the debug deal (when in debug mode)
        if debug_deal is not None and normalized_deal != normalized_debug:
            continue
        
        mtm_value = deal_row['total_mtm']
        mtm_abs = abs(mtm_value)
        
        # Direction is the same for both normal and reversal validation
        direction = 'POSITIVO' if mtm_value > 0 else 'NEGATIVO'
        
        # Find the instrument type for this deal
        deal_records = mtm_df[mtm_df[deal_col] == deal_number]
        if len(deal_records) == 0:
            continue
            
        product_code = deal_records[product_col].iloc[0]
        instrument_type = product_mapping.get(product_code)
        
        if not instrument_type:
            if debug_deal is not None:
                st.warning(f"Unknown product code: {product_code} for deal {deal_number}")
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Processing deal {deal_number}, instrument: {instrument_type}, direction: {direction}, MTM: {mtm_abs:.2f}")
        
        # STEP 1: Find applicable rules
        applicable_rules = mtm_rules[
            (mtm_rules['subproduct'] == instrument_type) & 
            (mtm_rules['direction'].str.upper() == direction)
        ]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'matched_entries': 0,
                'issue': f'No rule found for {instrument_type} with direction {direction}'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"Found {len(applicable_rules)} applicable rules:")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
        
        # STEP 2: Extract expected accounts from rules
        expected_accounts = []
        
        for _, rule in applicable_rules.iterrows():
            # Add debit account if present and not already added
            if pd.notna(rule['debit_account']):
                debit_account = str(rule['debit_account'])
                if debit_account not in [acc['account'] for acc in expected_accounts]:
                    expected_accounts.append({
                        'account': debit_account,
                        'type': 'debit',
                        'matched': False
                    })
            
            # Add credit account if present and not already added
            if pd.notna(rule['credit_account']):
                credit_account = str(rule['credit_account'])
                if credit_account not in [acc['account'] for acc in expected_accounts]:
                    expected_accounts.append({
                        'account': credit_account,
                        'type': 'credit',
                        'matched': False
                    })
        
        if debug_deal is not None:
            st.write(f"Expected accounts: {[acc['account'] for acc in expected_accounts]}")
        
        # STEP 3: Find matching entries in the interface file
        # Filter by trade number
        deal_entries = mtm_entries[
            (mtm_entries[trade_number_col] == deal_number)
        ]
        
        if debug_deal is not None:
            st.write(f"Found {len(deal_entries)} entries for deal {deal_number} in interface file")
            if len(deal_entries) > 0:
                st.dataframe(deal_entries, use_container_width=True, hide_index=True)
        
        if len(deal_entries) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'status': 'Missing Entries',
                'interface_entries': 0,
                'matched_entries': 0,
                'issue': f'No entries found in interface file for deal {deal_number}'
            })
            continue
        
        # STEP 4: Match entries with expected accounts
        matched_entries = 0
        value_matched_entries = 0
        total_debit_sum = 0
        total_credit_sum = 0
        
        for expected in expected_accounts:
            account = expected['account']
            entry_type = expected['type']

            # Filter by account
            account_entries = deal_entries[deal_entries[account_col].astype(str) == account]
            
            if len(account_entries) == 0:
                if debug_deal is not None:
                    st.warning(f"No entries found for account {account}")
                continue
            
            # Check for value match
            if entry_type == 'debit':
                # Sum debit values for this account
                entry_value = account_entries[debit_col].sum()
                total_debit_sum += entry_value
                if abs(entry_value - mtm_abs) < 1.0:
                    expected['matched'] = True
                    matched_entries += 1
                    value_matched_entries += 1
                    if debug_deal is not None:
                        st.success(f"✓ Found matching debit entry: Account {account}, Value {entry_value:.2f}")
                elif entry_value > 0:
                    # Account matches but value doesn't
                    matched_entries += 1
                    if debug_deal is not None:
                        st.warning(f"✗ Found debit entry with incorrect value: Account {account}, Expected {mtm_abs:.2f}, Found {entry_value:.2f}")
            else:  # credit
                # Sum credit values for this account
                entry_value = account_entries[credit_col].sum()
                total_credit_sum += entry_value
                if abs(entry_value - mtm_abs) < 1.0:
                    expected['matched'] = True
                    matched_entries += 1
                    value_matched_entries += 1
                    if debug_deal is not None:
                        st.success(f"✓ Found matching credit entry: Account {account}, Value {entry_value:.2f}")
                elif entry_value > 0:
                    # Account matches but value doesn't
                    matched_entries += 1
                    if debug_deal is not None:
                        st.warning(f"✗ Found credit entry with incorrect value: Account {account}, Expected {mtm_abs:.2f}, Found {entry_value:.2f}")
        
        # Determine overall match status
        if matched_entries == 0:
            status = 'No Match'
            issue = 'No matching accounts found'
        elif value_matched_entries == len(expected_accounts):
            status = 'Full Match'
            issue = ''
        elif matched_entries > 0:
            status = 'Partial Match'
            issue = f'Found {matched_entries} account matches, but only {value_matched_entries} with correct values. Expected MTM: {mtm_abs:.2f}, Found debit: {total_debit_sum:.2f}, Found credit: {total_credit_sum:.2f}'
        else:
            status = 'Unknown'
            issue = 'Validation logic error'
        
        # Add to validation results
        validation_results.append({
            'deal_number': str(deal_number),
            'instrument_type': instrument_type,
            'direction': direction,
            'mtm_value': mtm_abs,
            'status': status,
            'expected_entries': len(expected_accounts),
            'fully_matched_entries': value_matched_entries,
            'interface_entries': len(deal_entries),
            'issue': issue
        })
    
    # Create validation results dataframe
    validation_df = pd.DataFrame(validation_results)
    
    if len(validation_df) > 0:
        # Convert object columns to strings for Arrow compatibility
        for col in validation_df.columns:
            if validation_df[col].dtype == 'object':
                validation_df[col] = validation_df[col].astype(str)
            
        # Format numeric columns
        numeric_cols = validation_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            validation_df[col] = validation_df[col].round(2)
        
        # Display validation results
        st.subheader(f"{event_type} Validation Results")
        st.write(f"Matches made on combination of trade number, debe/haber, account number and MtM value. Partial matches are considered where the first three elements match but the MtM value does not.")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Calculate and display match statistics
        full_match_count = len(validation_df[validation_df['status'] == 'Full Match'])
        partial_match_count = len(validation_df[validation_df['status'] == 'Partial Match'])
        no_match_count = len(validation_df[validation_df['status'] == 'No Match'])
        total_count = len(validation_df)
        
        match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Deals", total_count)
        with col2:
            st.metric("Full Matches", full_match_count)
        with col3:
            st.metric("Partial Matches", partial_match_count)
        with col4:
            st.metric("Match Rate", f"{match_percentage:.1f}%")
        
        # Status breakdown
        status_counts = validation_df['status'].value_counts().to_dict()
        st.subheader(f"{event_type} Status Breakdown")
        st.write(status_counts)
        
        # Allow downloading results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            f"Download {event_type} Results",
            csv,
            f"{event_type.lower().replace(' ', '_')}_validation_results.csv",
            "text/csv",
            key=f'download-csv-{event_type.lower().replace(" ", "_")}{key_suffix}'
        )
    else:
        st.warning(f"No {event_type} validation results generated")
    
    return validation_df

# Streamlit UI
st.title("Accounting Interface Validator")

with st.sidebar:
    st.header("Upload Files")
    interface_file = st.file_uploader("Upload Accounting Interface File", type=["xls", "xlsx"])
    day_trades_file = st.file_uploader("Upload Day Trades File", type=["csv"], 
                                    help="New trades entered today")
    mtm_file = st.file_uploader("Upload MTM File (T)", type=["xlsx", "csv"])
    mtm_t1_file = st.file_uploader("Upload MTM File (T-1)", type=["xlsx", "csv"], 
                                 help="Previous day's MTM file for reversal validation")
    rules_file = st.file_uploader("Upload Accounting Rules", type=["xlsx", "csv"])
    
    # Debug options
    st.subheader("Debug Options")
    debug_deal = st.text_input("Debug specific deal number (optional)", "")
    debug_deal = debug_deal if debug_deal else None

# Main area
if interface_file and mtm_file and rules_file:
    st.header("File Analysis")
    
    # Parse files
    with st.expander("Interface File", expanded=False):
        interface_df, interface_cols = parse_interface_file(interface_file)
    
    with st.expander("MTM File (T)", expanded=False):
        mtm_df, mtm_sums = parse_mtm_file(mtm_file)
    
    # Parse MTM T-1 file if provided
    mtm_t1_df = None
    mtm_t1_sums = None
    if mtm_t1_file:
        with st.expander("MTM File (T-1)", expanded=False):
            mtm_t1_df, mtm_t1_sums = parse_mtm_file(mtm_t1_file)
    
    # Parse day trades file if provided
    day_trades_df = None
    if day_trades_file:
        with st.expander("Day Trades File", expanded=False):
            day_trades_df = parse_day_trades_file(day_trades_file)
            
    with st.expander("Rules File", expanded=False):
        rules_df = parse_rules_file(rules_file)
    
    # Validation options
    st.subheader("Validation Options")
    col1, col2 = st.columns(2)
    
    with col1:
        run_mtm_validation = st.checkbox("Run MTM Valorization Validation", value=True)
        run_day_trades_validation = st.checkbox("Run Day Trades Validation", 
                                           value=day_trades_df is not None)
    
    with col2:
        run_reversal_validation = st.checkbox("Run MTM Reversal Validation", value=mtm_t1_file is not None)
        if run_reversal_validation and mtm_t1_file is None:
            st.warning("MTM File (T-1) is required for reversal validation")
            run_reversal_validation = False
    
    # Run validations
    if st.button("Run Validation"):
        with st.spinner("Running validation..."):
            # Run day trades validation if selected
            if run_day_trades_validation and day_trades_df is not None:
                st.header("Day Trades Validation")
                day_trades_results = validate_day_trades(
                    day_trades_df,
                    interface_df,
                    interface_cols,
                    rules_df,
                    debug_deal=debug_deal
                )
            
            # Run MTM validation if selected
            if run_mtm_validation:
                st.header("MTM Valorization Validation")
                mtm_results = validate_mtm_entries(
                    interface_df, 
                    interface_cols, 
                    mtm_df, 
                    mtm_sums, 
                    rules_df, 
                    event_type='Valorización MTM',
                    debug_deal=debug_deal
                )
            
            # Run reversal validation if selected and T-1 file is provided
            if run_reversal_validation and mtm_t1_df is not None:
                st.header("MTM Reversal Validation")
                reversal_results = validate_mtm_entries(
                    interface_df, 
                    interface_cols, 
                    mtm_t1_df, 
                    mtm_t1_sums, 
                    rules_df, 
                    event_type='Valorización MTM',      # For interface file filtering
                    rules_event_type='Reversa Valorización MTM',  # For rules file filtering
                    key_suffix='-reversal',
                    debug_deal=debug_deal
                )
else:
    st.info("Please upload the required files to start validation.")