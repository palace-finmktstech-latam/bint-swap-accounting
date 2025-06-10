import streamlit as st
import pandas as pd
import numpy as np

def validate_day_trades(day_trades_df, interface_df, interface_cols, rules_df, debug_deal=None):
    """
    Validate day trades against accounting interface entries.
    
    For each day trade, checks that:
    1. Corresponding "Curse" entries exist in the accounting interface
    2. Correct account numbers are used based on instrument type and Pata
    3. Transaction amounts match the Monto Activo/Pasivo value based on Pata
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
        monto_pasivo = trade.get('Monto Pasivo', 0)  # Default to 0 if column doesn't exist
        currency = trade['Moneda Activa']
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG: Processing trade {trade_number}, instrument: {instrument_type}")
            st.write(f"DEBUG: Monto Activo: {monto_activo}, Monto Pasivo: {monto_pasivo}, Currency: {currency}")
        
        # Get applicable rules for this instrument type
        applicable_rules = inicio_rules[inicio_rules['subproduct'] == instrument_type]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'monto_pasivo': monto_pasivo,
                'currency': currency,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No rule found for {instrument_type}'
            })
            continue
        
        # Separate rules by Pata and build expected accounts
        activa_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Activa']
        pasiva_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Pasiva']
        
        if debug_deal is not None:
            st.write(f"DEBUG: Found {len(activa_rules)} Pata Activa rules, {len(pasiva_rules)} Pata Pasiva rules")
        
        # Build expected accounts list
        expected_accounts = []
        expected_amounts = {}  # Dictionary to track which amount each account should use
        
        # Process Pata Activa rules
        for _, rule in activa_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': monto_activo, 'pata': 'Activa', 'field': 'debit'}
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': monto_activo, 'pata': 'Activa', 'field': 'credit'}
        
        # Process Pata Pasiva rules
        for _, rule in pasiva_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': monto_pasivo, 'pata': 'Pasiva', 'field': 'debit'}
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': monto_pasivo, 'pata': 'Pasiva', 'field': 'credit'}

        # Remove any "None" values
        expected_accounts = [acc for acc in expected_accounts if acc != "None" and acc != "nan"]
        expected_entry_count = len(expected_accounts)
            
        # Find interface entries for this trade
        trade_entries = curse_entries[
            (curse_entries[trade_number_col] == trade_number)
        ]
        
        if debug_deal is not None:
            st.write(f"Found {len(trade_entries)} entries for trade {trade_number} in interface")
            st.write(f"Expected {expected_entry_count} entries based on rules")
            st.write(f"Expected account numbers: {expected_accounts}")
            st.write("Expected amounts by account:")
            for acc, info in expected_amounts.items():
                st.write(f"  {acc}: {info['amount']} ({info['pata']}, {info['field']})")
            st.dataframe(trade_entries, use_container_width=True, hide_index=True)
        
        if len(trade_entries) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'monto_activo': monto_activo,
                'monto_pasivo': monto_pasivo,
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
                'monto_pasivo': monto_pasivo,
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
                'monto_pasivo': monto_pasivo,
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
                'monto_pasivo': monto_pasivo,
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
            
            # Get expected amount and field from our mapping
            expected_info = expected_amounts[account]
            expected_amount = expected_info['amount']
            expected_field = expected_info['field']
            pata_type = expected_info['pata']
            
            # Get the total amount in the expected field
            if expected_field == 'debit':
                actual_amount = account_entries[debit_col].sum()
            elif expected_field == 'credit':
                actual_amount = account_entries[credit_col].sum()
            else:
                actual_amount = 0
            
            # Check if amount matches the expected value
            is_matching = abs(actual_amount - expected_amount) < 1.0
            
            account_validation[account] = {
                'expected_field': expected_field,
                'expected_amount': expected_amount,
                'actual_amount': actual_amount,
                'pata': pata_type,
                'matches': is_matching
            }
            
            if debug_deal is not None:
                if is_matching:
                    st.success(f"✓ Account {account} ({expected_field}, {pata_type}): Expected {expected_amount}, Found {actual_amount:.2f}")
                else:
                    st.warning(f"✗ Account {account} ({expected_field}, {pata_type}): Expected {expected_amount}, Found {actual_amount:.2f}")
        
        # Check if all amounts match
        all_match = all(val['matches'] for val in account_validation.values())
        
        if all_match:
            status = 'Full Match'
            issue = ''
        else:
            status = 'Amount Mismatch'
            mismatches = [
                f"{account} ({val['expected_field']}, {val['pata']}): Expected {val['expected_amount']}, Found {val['actual_amount']:.2f}"
                for account, val in account_validation.items() if not val['matches']
            ]
            issue = f"Amount mismatches: {'; '.join(mismatches)}"
        
        # Add to results
        validation_results.append({
            'trade_number': str(trade_number),
            'instrument_type': instrument_type,
            'monto_activo': monto_activo,
            'monto_pasivo': monto_pasivo,
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
        st.write("Now supports Pata Activa (uses Monto Activo) and Pata Pasiva (uses Monto Pasivo) rules")
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
    
    # DEBUGGING
    #st.write(f"Filter event: {filter_event}")

    # Filter rules using the appropriate event type
    mtm_rules = rules_df[rules_df['event'] == filter_event].copy()

    # DEBUGGING
    #st.write(f"MTM rules: {mtm_rules}")
    
    """
    Validate MTM entries against expected entries from rules:
    1. Identify expected entries from rules
    2. Get MTM values from MTM file
    3. Match with accounting interface entries
    4. Report on full, partial or non-matches
    """
    if debug_deal is not None:
        st.write(f"DEBUG: Analyzing deal number {debug_deal}")
    
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
            # DEBUGGING
            st.write(f"debug_deal: {debug_deal}")
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
        deal_number = int(float(str(deal_number)))
        #st.write(f"deal_number type: {type(deal_number)}, value: {deal_number}")
        #st.write(f"trade_number_col values type: {type(mtm_entries[trade_number_col].iloc[0])}")
        deal_entries = mtm_entries[
            (mtm_entries[trade_number_col] == deal_number)
        ]
        
        # DEBUGGING
        #st.write(f"Deal number: {deal_number}")
        #st.write("First 5 rows of mtm_entries:")
        #st.dataframe(mtm_entries.head())
        #st.write(f"trade_number_col: {trade_number_col}")
        #st.write(f"Deal entries: {deal_entries}")

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
            'interface_entries': len(deal_entries),
            'matched_entries': value_matched_entries,
            'expected_accounts': len(expected_accounts),
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
        st.write(f"Matches made on combination of trade number, debe/haber, account number and MTM value. Partial matches are considered where the first three elements match but the MTM value does not. A match is considered valid when the MTM value difference is less than 1.0 (absolute value).")
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