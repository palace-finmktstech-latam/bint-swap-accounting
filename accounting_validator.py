import streamlit as st
import pandas as pd
import numpy as np

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
            else ('Vcto' if 'Vcto' in x else None)))
        
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
            'TERMINO': 'Vcto'
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

def validate_mtm_entries(interface_df, interface_cols, mtm_df, mtm_sums, rules_df, debug_deal=None):
    """
    Validate MTM entries against expected entries from rules with a cleaner approach:
    1. Identify expected entries from rules
    2. Get MTM values from MTM file
    3. Match with accounting interface entries
    4. Report on full, partial or non-matches
    """
    #if debug_deal is not None:
    #    st.write(f"DEBUG: Analyzing deal number {debug_deal}")
    
    # Filter rules for MTM event
    mtm_rules = rules_df[rules_df['event'] == 'Valorización MTM'].copy()
    st.write(f"Found {len(mtm_rules)} MTM rules")
    
    # Display MTM rules
    st.subheader("MTM Rules")
    st.dataframe(mtm_rules, use_container_width=True, hide_index=True)
    
    if len(mtm_rules) == 0:
        st.error("No MTM rules found in rules file")
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
    
    # Extract account interface entries for MTM
    mtm_entries = interface_df[interface_df['event_type'] == 'Valorización MTM'].copy()
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    
    # Ensure numeric values in MTM file and interface
    mtm_df[mtm_col] = pd.to_numeric(mtm_df[mtm_col], errors='coerce')
    mtm_entries[debit_col] = pd.to_numeric(mtm_entries[debit_col], errors='coerce').fillna(0)
    mtm_entries[credit_col] = pd.to_numeric(mtm_entries[credit_col], errors='coerce').fillna(0)
    
    st.write(f"Found {len(mtm_entries)} MTM entries in interface file")
    
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
        
        #if debug_deal is not None:
        #    st.write(f"DEBUG: Processing deal {deal_number}, instrument: {instrument_type}, direction: {direction}, MTM: {mtm_abs:.2f}")
        
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
        
        #st.write(f"Debug_deal: {debug_deal}")
        #st.write(f"DEBUG: Applicable rules: {applicable_rules}")

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
        # Filter by trade number and instrument type
        #st.write(f"Deal number: {deal_number}")
        #st.write(f"Instrument type: {instrument_type}")
        #st.write(f"Trade number column: {trade_number_col}")
        #st.write(f"Trade number column type: {type(mtm_entries[trade_number_col])}")
        #st.write(f"Deal number type: {type(deal_number)}")

        # Debug trade number column values
        #st.write("Sample of trade numbers in mtm_entries:")
        #st.write(mtm_entries[trade_number_col].head().tolist())
        #st.write("Unique trade numbers in mtm_entries:")
        #st.write(mtm_entries[trade_number_col].unique().tolist())

        # Debug instrument type values
        #st.write("Sample of instrument types in mtm_entries:")
        #st.write(mtm_entries['instrument_type'].head().tolist())
        #st.write("Unique instrument types in mtm_entries:")
        #st.write(mtm_entries['instrument_type'].unique().tolist())

        # Try filtering step by step to see where it fails
        #st.write("Testing filters separately:")

        # Test trade number filter
        #trade_number_matches = mtm_entries[mtm_entries[trade_number_col] == deal_number]
        #st.write(f"Entries matching trade number {deal_number}: {len(trade_number_matches)}")

        # Test instrument type filter
        #instrument_matches = mtm_entries[mtm_entries['instrument_type'] == instrument_type]
        #st.write(f"Entries matching instrument type {instrument_type}: {len(instrument_matches)}")

        # Now apply the filter
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
        
        for expected in expected_accounts:
            account = expected['account']
            entry_type = expected['type']

            # Format debug information in a more readable way
            #st.write("---")
            #st.write("### Account Matching Details")
            #col1, col2 = st.columns(2)
            #with col1:
            #    st.write("**Expected Account Information:**")
            #    st.write(f"- Account: {account}")
            #    st.write(f"- Type: {entry_type}")
            #    st.write(f"- Expected MTM Value: {mtm_abs:.2f}")
            
            # Filter by account
            account_entries = deal_entries[deal_entries[account_col].astype(str) == account]
            
            #with col2:
            #    st.write("**Found Entries:**")
            #    if len(account_entries) > 0:
            #        st.write(f"- Number of entries found: {len(account_entries)}")
            #        if entry_type == 'debit':
            #            entry_value = account_entries[debit_col].sum()
            #        else:
            #            entry_value = account_entries[credit_col].sum()
            #        st.write(f"- Total value: {entry_value:.2f}")
            #        st.write(f"- Difference from expected: {abs(entry_value - mtm_abs):.2f}")
            #    else:
            #        st.write("- No matching entries found")

            #if len(account_entries) > 0:
            #    st.write("**Entry Details:**")
            #    st.dataframe(
            #        account_entries[[account_col, debit_col, credit_col]],
            #        use_container_width=True,
            #        hide_index=True
            #    )

            if len(account_entries) == 0:
                if debug_deal is not None:
                    st.warning(f"No entries found for account {account}")
                continue
            
            # Check for value match
            if entry_type == 'debit':
                # Sum debit values for this account
                entry_value = account_entries[debit_col].sum()
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
        
        #st.write(f"DEBUG: Matched entries: {matched_entries}")
        #st.write(f"DEBUG: Value matched entries: {value_matched_entries}")

        # Determine overall match status
        if matched_entries == 0:
            status = 'No Match'
            issue = 'No matching accounts found'
        elif value_matched_entries == len(expected_accounts):
            status = 'Full Match'
            issue = ''
        elif matched_entries > 0:
            status = 'Partial Match'
            issue = f'Found {matched_entries} account matches, but only {value_matched_entries} with correct values. Expected MTM: {mtm_abs:.2f} not found in Accounting Interface'
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
        st.subheader("Validation Results")
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
        st.subheader("Status Breakdown")
        st.write(status_counts)
    else:
        st.warning("No validation results generated")
    
    return validation_df

# Streamlit UI
st.title("Accounting Interface Validator")

with st.sidebar:
    st.header("Upload Files")
    interface_file = st.file_uploader("Upload Accounting Interface File", type=["xls", "xlsx"])
    mtm_file = st.file_uploader("Upload MTM File", type=["xlsx", "csv"])
    rules_file = st.file_uploader("Upload Accounting Rules", type=["xlsx", "csv"])

# Main area
if interface_file and mtm_file and rules_file:
    st.header("File Analysis")
    
    # Parse files
    with st.expander("Interface File", expanded=False):
        interface_df, interface_cols = parse_interface_file(interface_file)
    
    with st.expander("MTM File", expanded=False):
        mtm_df, mtm_sums = parse_mtm_file(mtm_file)
    
    with st.expander("Rules File", expanded=False):
        rules_df = parse_rules_file(rules_file)
    
    # Validate entries
    if st.button("Run Validation"):
        with st.spinner("Running validation..."):
            #results_df = validate_mtm_entries(interface_df, interface_cols, mtm_df, mtm_sums, rules_df, debug_deal="2497.0")
            results_df = validate_mtm_entries(interface_df, interface_cols, mtm_df, mtm_sums, rules_df)
else:
    st.info("Please upload the Accounting Interface, MTM, and Rules files to start validation.")