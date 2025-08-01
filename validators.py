import streamlit as st
import pandas as pd
import numpy as np
from file_parsers import detect_instrument_type_from_cartera_treasury, extract_estrategia_from_cartera_treasury

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
        cobertura = trade.get('Cobertura', 'No')  # Default to 'No' if column doesn't exist
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG: Processing trade {trade_number}, instrument: {instrument_type}")
            st.write(f"DEBUG: Monto Activo: {monto_activo}, Monto Pasivo: {monto_pasivo}, Currency: {currency}")
            st.write(f"DEBUG: Cobertura: {cobertura}")
        
        # Get applicable rules for this instrument type AND cobertura
        applicable_rules = inicio_rules[
            (inicio_rules['subproduct'] == instrument_type) & 
            (inicio_rules['coverage'] == cobertura)
        ]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'cobertura': cobertura,
                'monto_activo': monto_activo,
                'monto_pasivo': monto_pasivo,
                'currency': currency,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No rule found for {instrument_type} with Cobertura={cobertura}'
            })
            continue
        
        # Separate rules by Pata and build expected accounts
        activa_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Activa']
        pasiva_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Pasiva']
        
        if debug_deal is not None:
            st.write(f"Found {len(activa_rules)} Pata Activa rules, {len(pasiva_rules)} Pata Pasiva rules")
            st.write(f"All applicable rules for {instrument_type} with Cobertura={cobertura}:")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
        
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
                'cobertura': cobertura,
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
                'cobertura': cobertura,
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
                'cobertura': cobertura,
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
                'cobertura': cobertura,
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
            'cobertura': cobertura,
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

def validate_mtm_entries(interface_df, interface_cols, mtm_df, mtm_sums, rules_df, cartera_df=None,
                          event_type='Valorización MTM', 
                          rules_event_type=None,
                          key_suffix='', 
                          debug_deal=None,
                          combined_mtm_mode=False):
    
    # Use rules_event_type if provided, otherwise default to event_type
    filter_event = rules_event_type if rules_event_type else event_type
    
    # Filter rules using the appropriate event type
    mtm_rules = rules_df[rules_df['event'] == filter_event].copy()
    
    """
    Enhanced MTM validation with separated core vs extra entry checking.
    
    Status categories:
    - Full Match: Required entries correct, no extras
    - Correct + Extra Entries: Required entries correct, but unnecessary extras exist  
    - Amount Mismatch: Required entries have wrong amounts
    - Missing Accounts: Required accounts missing
    - Missing Entries: No entries found
    - Cartera Anomaly: Deal not in Cartera file
    """
    if debug_deal is not None:
        st.write(f"DEBUG: Analyzing deal number {debug_deal}")
    
    st.write(f"Found {len(mtm_rules)} {event_type} rules")
    st.write(f"Processing enhanced MTM validation with core vs extra entry separation")
    
    # Display MTM rules
    st.subheader(f"{event_type} Rules")
    st.dataframe(mtm_rules, use_container_width=True, hide_index=True)
    
    if len(mtm_rules) == 0:
        st.error(f"No {event_type} rules found in rules file")
        return pd.DataFrame()
    
    # Check if Cartera file is provided (now required)
    if cartera_df is None or cartera_df.empty:
        st.error("❌ Cartera file is required for MTM validation but was not provided")
        st.error("Please upload the Cartera Analytics file to proceed with MTM validation")
        return pd.DataFrame()
    
    # Create estrategia lookup from Cartera file
    estrategia_lookup = dict(zip(cartera_df['deal_number'], cartera_df['estrategia']))
    st.write(f"✅ Using Cartera file with estrategia data for {len(estrategia_lookup)} deals")
    
    # Check if rules have required columns (using existing column names)
    required_rule_columns = ['coverage', 'Estrategia']
    missing_rule_columns = [col for col in required_rule_columns if col not in mtm_rules.columns]
    
    if missing_rule_columns:
        st.error(f"❌ Missing required columns in rules file: {', '.join(missing_rule_columns)}")
        st.error("Rules file must contain both 'coverage' and 'Estrategia' columns for MTM validation")
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
    
    # Track anomalies
    cartera_anomalies = []
    
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
        
        # Get estrategia from Cartera file
        estrategia = estrategia_lookup.get(normalized_deal)
        
        if estrategia is None:
            # Handle missing deal in Cartera file - mark as anomaly and skip
            cartera_anomalies.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'issue': f'Deal {deal_number} not found in Cartera file'
            })
            
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'estrategia': 'MISSING',
                'status': 'Cartera Anomaly',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': 0,
                'extra_entries': 0,
                'issue': f'Deal {deal_number} not found in Cartera file - cannot determine estrategia'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Processing deal {deal_number}, instrument: {instrument_type}, direction: {direction}, MTM: {mtm_abs:.2f}")
            st.write(f"DEBUG: Estrategia from Cartera: {estrategia}")
        
        # Apply Estrategia/Cobertura filtering logic
        if estrategia == "NO":
            # Filter by coverage column where value is "No"
            applicable_rules = mtm_rules[
                (mtm_rules['subproduct'] == instrument_type) & 
                (mtm_rules['direction'].str.upper() == direction) &
                (mtm_rules['coverage'] == 'No')
            ]
            filter_criteria = f"coverage='No'"
        else:
            # Filter by Estrategia column where value matches the estrategia from Cartera
            applicable_rules = mtm_rules[
                (mtm_rules['subproduct'] == instrument_type) & 
                (mtm_rules['direction'].str.upper() == direction) &
                (mtm_rules['Estrategia'] == estrategia)
            ]
            filter_criteria = f"Estrategia='{estrategia}'"
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'estrategia': estrategia,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': 0,
                'extra_entries': 0,
                'issue': f'No rule found for {instrument_type} with direction {direction} and {filter_criteria}'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Found {len(applicable_rules)} applicable rules using filter: {filter_criteria}")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
            st.write(f"DEBUG: Combined MTM mode2: {combined_mtm_mode}")
        
        # Extract expected accounts from rules
        expected_accounts = []
        expected_amounts = {}  # Build expected amounts dictionary for validation
        
        for _, rule in applicable_rules.iterrows():
            # Add debit account if present and not already added
            if pd.notna(rule['debit_account']):
                debit_account = str(rule['debit_account'])
                if debit_account not in expected_accounts:
                    expected_accounts.append(debit_account)
                    expected_amounts[debit_account] = {'amount': mtm_abs, 'field': 'debit'}
            
            # Add credit account if present and not already added
            if pd.notna(rule['credit_account']):
                credit_account = str(rule['credit_account'])
                if credit_account not in expected_accounts:
                    expected_accounts.append(credit_account)
                    expected_amounts[credit_account] = {'amount': mtm_abs, 'field': 'credit'}
        
        if debug_deal is not None:
            st.write(f"Expected accounts: {expected_accounts}")
            st.write("Expected amounts by account:")
            for acc, info in expected_amounts.items():
                st.write(f"  {acc}: {info['amount']} ({info['field']})")
        
        # Find matching entries in the interface file
        # Filter by trade number
        deal_number = int(float(str(deal_number)))
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
                'estrategia': estrategia,
                'status': 'Missing Entries',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': len(expected_accounts),
                'extra_entries': 0,
                'issue': f'No entries found in interface file for deal {deal_number}'
            })
            continue
        
        if combined_mtm_mode:
            # In combined mode, expect entries for both current and reversal MTM
            adjusted_expected_count = len(expected_accounts) * 2  # Expect double entries
            
            if debug_deal is not None:
                st.write(f"DEBUG: Combined mode - expecting {adjusted_expected_count} total entries (2x {len(expected_accounts)})")
                st.write(f"DEBUG: Found {len(deal_entries)} actual entries")
            
            # Check core validation manually without flagging extras
            # Step 1: Check if all required accounts are present
            found_accounts = deal_entries[account_col].astype(str).unique().tolist()
            missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
            
            if missing_accounts:
                status = '❌ Missing Required'
                issue = f'Missing required accounts: {", ".join(missing_accounts)}'
                extra_entry_count = len(deal_entries)
            else:
                # Step 2: Check amounts for required accounts
                all_required_correct = True
                core_validation_issues = []
                
                for account in expected_accounts:
                    account_entries = deal_entries[deal_entries[account_col].astype(str) == account]
                    expected_amount = expected_amounts[account]['amount']
                    expected_field = expected_amounts[account]['field']
                    
                    if expected_field == 'debit':
                        actual_amount = account_entries[debit_col].sum()
                    else:
                        actual_amount = account_entries[credit_col].sum()
                    
                    is_matching = abs(actual_amount - expected_amount) < 1.0
                    
                    if not is_matching:
                        all_required_correct = False
                        core_validation_issues.append(f"{account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
                    
                    if debug_deal is not None:
                        status_icon = "✅" if is_matching else "❌"
                        st.write(f"{status_icon} Combined MTM Account {account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
                
                # Step 3: Determine status based on combined mode expectations
                if not all_required_correct:
                    status = '❌ Wrong Amounts'
                    issue = f"Required entries have wrong amounts: {'; '.join(core_validation_issues)}"
                    extra_entry_count = max(0, len(deal_entries) - adjusted_expected_count)
                else:
                    # Core validation passed - check entry count for combined mode
                    extra_entry_count = max(0, len(deal_entries) - adjusted_expected_count)
                    
                    if extra_entry_count == 0:
                        status = '✅ Perfect Match'
                        issue = 'All required entries correct (combined MTM mode)'
                        if debug_deal is not None:
                            st.success(f"🎯 PERFECT COMBINED MATCH - found exactly {len(deal_entries)} entries as expected for MTM+Reversal")
                    else:
                        status = '✅ Correct (+ Extras)'
                        issue = f'Required entries correct, but {extra_entry_count} extra entries beyond expected combined total'
                        if debug_deal is not None:
                            st.info(f"✅ All required entries CORRECT for combined MTM")
                            st.warning(f"⚠️ But found {extra_entry_count} entries beyond the expected {adjusted_expected_count} for MTM+Reversal")
        else:
            # Normal mode - original logic
            status, issue, extra_entry_count = _validate_entries_against_expected(
                deal_entries, expected_accounts, expected_amounts, 
                account_col, debit_col, credit_col, debug_deal, "MTM"
            )
        
        # Count correctly matched entries for the matched_entries column
        correctly_matched_entries = 0
        if status in ['✅ Perfect Match', '✅ Correct (+ Extras)']:
            correctly_matched_entries = len(expected_accounts)
        elif 'Amount Mismatch' not in status:
            # For other statuses, count how many accounts are actually present
            found_accounts = deal_entries[account_col].astype(str).unique().tolist()
            correctly_matched_entries = len([acc for acc in expected_accounts if acc in found_accounts])
        
        # Add to validation results
        validation_results.append({
            'deal_number': str(deal_number),
            'instrument_type': instrument_type,
            'direction': direction,
            'mtm_value': mtm_abs,
            'estrategia': estrategia,
            'status': status,
            'interface_entries': len(deal_entries),
            'matched_entries': correctly_matched_entries,
            'expected_accounts': len(expected_accounts),
            'extra_entries': extra_entry_count,
            'issue': issue
        })
    
    # Display Cartera anomalies if any
    if cartera_anomalies:
        st.error(f"⚠️ Found {len(cartera_anomalies)} deals in MTM file that are missing from Cartera file:")
        anomaly_df = pd.DataFrame(cartera_anomalies)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    
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
        st.write(f"Enhanced validation with separated core vs extra entry checking:")
        st.write(f"• **Full Match**: Required entries correct, no extras")
        st.write(f"• **Correct + Extra Entries**: Required entries correct, but unnecessary extras exist")
        st.write(f"• **Amount Mismatch**: Required entries have wrong amounts")
        st.write(f"• **Missing Accounts/Entries**: Required entries missing")
        st.write(f"• If Estrategia='NO' → Filter rules by coverage='No'")
        st.write(f"• If Estrategia≠'NO' → Filter rules by Estrategia=estrategia_value")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Show estrategia distribution
        if 'estrategia' in validation_df.columns:
            estrategia_counts = validation_df['estrategia'].value_counts()
            st.write("**Estrategia Distribution:**", estrategia_counts.to_dict())
        
        # Calculate and display match statistics with enhanced categories
        full_match_count = len(validation_df[validation_df['status'] == '✅ Perfect Match'])
        correct_plus_extra_count = len(validation_df[validation_df['status'] == '✅ Correct (+ Extras)'])
        amount_mismatch_count = len(validation_df[validation_df['status'] == '❌ Wrong Amounts'])
        missing_entries_count = len(validation_df[validation_df['status'] == '❌ No Entries'])
        missing_accounts_count = len(validation_df[validation_df['status'] == '❌ Missing Required'])
        anomaly_count = len(validation_df[validation_df['status'] == 'Cartera Anomaly'])
        missing_rule_count = len(validation_df[validation_df['status'] == 'Missing Rule'])
        total_count = len(validation_df)
        
        # Calculate success rates
        core_success_rate = ((full_match_count + correct_plus_extra_count) / total_count * 100) if total_count > 0 else 0
        perfect_match_rate = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Deals", total_count)
        with col2:
            st.metric("Perfect Matches", full_match_count)
        with col3:
            st.metric("Correct + Extra", correct_plus_extra_count)
        with col4:
            st.metric("Amount Issues", amount_mismatch_count)
        with col5:
            st.metric("Missing Issues", missing_entries_count + missing_accounts_count)
        with col6:
            st.metric("Core Success Rate", f"{core_success_rate:.1f}%")
        
        # Enhanced status breakdown
        status_counts = validation_df['status'].value_counts().to_dict()
        st.subheader(f"{event_type} Enhanced Status Breakdown")
        st.write("**Primary Concerns (Core Business Logic):**")
        st.write(f"• ✅ Perfect Match: {full_match_count}")
        st.write(f"• ✅ Correct + Extra Entries: {correct_plus_extra_count}")
        st.write(f"• ❌ Wrong Amounts: {amount_mismatch_count}")
        st.write(f"• ❌ Missing Required: {missing_accounts_count}")
        st.write(f"• ❌ No Entries: {missing_entries_count}")
        
        st.write("**Secondary Issues:**")
        st.write(f"• Cartera Anomaly: {anomaly_count}")
        st.write(f"• Missing Rule: {missing_rule_count}")
        
        # Show summary of extra entries if any exist
        total_extra_entries = validation_df[validation_df['status'] == '✅ Correct (+ Extras)']['extra_entries'].sum()
        if total_extra_entries > 0:
            st.info(f"📊 **Data Quality Summary**: {total_extra_entries} total unnecessary entries found across {correct_plus_extra_count} deals")
            st.write("These deals have correct accounting but also contain superfluous entries that should be cleaned up.")
        elif correct_plus_extra_count > 0:
            st.info(f"📊 **Data Quality Summary**: {correct_plus_extra_count} deals have correct core accounting with some extra entries to clean up.")
            
        # Allow downloading results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            f"Download Enhanced {event_type} Results",
            csv,
            f"enhanced_{event_type.lower().replace(' ', '_')}_validation_results.csv",
            "text/csv",
            key=f'download-csv-enhanced-{event_type.lower().replace(" ", "_")}{key_suffix}'
        )
    else:
        st.warning(f"No {event_type} validation results generated")
    
    return validation_df

def validate_vencimiento_entries(expiries_df, interface_df, interface_cols, rules_df, cartera_treasury_raw_df_or_cartera_df=None, debug_deal=None):
    """
    Validate both VENCIMIENTO and TERMINO entries against accounting interface entries.
    
    UPDATED: Now accepts either:
    1. Raw Cartera Treasury dataframe (new integrated approach) - extracts estrategia data automatically
    2. Pre-extracted Cartera Analytics dataframe (backward compatibility)
    
    VENCIMIENTO: Enhanced with Estrategia-based filtering from Cartera Treasury
    TERMINO: Uses existing logic (unchanged)
    
    Both validations run together since they're part of the same business process.
    """
    st.header("🔄 VENCIMIENTO & TERMINO Validation")
    
    # Check if we have any data source for estrategia
    if cartera_treasury_raw_df_or_cartera_df is None or cartera_treasury_raw_df_or_cartera_df.empty:
        st.error("❌ Cartera Treasury file is required for enhanced VENCIMIENTO validation but was not provided")
        st.error("Please upload the Cartera Treasury file to proceed with VENCIMIENTO validation")
        return pd.DataFrame()
    
    # Determine if we have raw Cartera Treasury data or pre-extracted estrategia data
    # Check if it has the expected Cartera Treasury columns or extracted format
    if 'estrategia' in cartera_treasury_raw_df_or_cartera_df.columns and 'deal_number' in cartera_treasury_raw_df_or_cartera_df.columns:
        # This is already extracted estrategia data (from main.py)
        cartera_df = cartera_treasury_raw_df_or_cartera_df
        st.write("✅ Using pre-extracted estrategia data")
    else:
        # This is raw Cartera Treasury data - extract estrategia
        st.write("🔄 Extracting estrategia data from Cartera Treasury file for VENCIMIENTO validation")
        cartera_df = extract_estrategia_from_cartera_treasury(cartera_treasury_raw_df_or_cartera_df)
        
        if cartera_df is None or cartera_df.empty:
            st.error("❌ Could not extract estrategia data from Cartera Treasury file")
            return pd.DataFrame()
    
    # Create estrategia lookup from Cartera data
    estrategia_lookup = dict(zip(cartera_df['deal_number'], cartera_df['estrategia']))
    st.write(f"✅ Using estrategia data for {len(estrategia_lookup)} deals")
    
    # Run VENCIMIENTO validation first (enhanced with Estrategia)
    st.subheader("📅 VENCIMIENTO Validation")
    vencimiento_results = _validate_vencimiento_only(expiries_df, interface_df, interface_cols, rules_df, estrategia_lookup, debug_deal)
    
    # Run TERMINO validation second (unchanged - no Estrategia filtering needed)
    st.subheader("🔚 TERMINO Validation")
    termino_results = _validate_termino_only(expiries_df, interface_df, interface_cols, rules_df, debug_deal)
    
    # Return both results
    return {
        'vencimiento': vencimiento_results,
        'termino': termino_results
    }

def _validate_vencimiento_only(expiries_df, interface_df, interface_cols, rules_df, estrategia_lookup, debug_deal=None):
    """
    Validate VENCIMIENTO (expiry) entries - the payment/settlement part.
    Uses Monto Override Extranjero/Local amounts.
    Enhanced with Estrategia-based rule filtering.
    """
    # Filter rules for VENCIMIENTO event
    vencimiento_rules = rules_df[rules_df['event'] == 'Vencimiento'].copy()
    
    if len(vencimiento_rules) == 0:
        st.error("No VENCIMIENTO rules found in rules file")
        return pd.DataFrame()
    
    st.write(f"Found {len(vencimiento_rules)} VENCIMIENTO rules")
    if debug_deal is not None:
        st.dataframe(vencimiento_rules, use_container_width=True, hide_index=True)
    
    # Check if rules have required columns
    required_rule_columns = ['coverage', 'Estrategia']
    missing_rule_columns = [col for col in required_rule_columns if col not in vencimiento_rules.columns]
    
    if missing_rule_columns:
        st.error(f"❌ Missing required columns in VENCIMIENTO rules file: {', '.join(missing_rule_columns)}")
        st.error("Rules file must contain both 'coverage' and 'Estrategia' columns for enhanced VENCIMIENTO validation")
        return pd.DataFrame()
    
    # Extract needed columns from interface
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    
    if not trade_number_col:
        st.error("Could not find trade number column in interface file")
        return pd.DataFrame()
    
    # Filter interface for "Vcto" event_type entries (both VENCIMIENTO and TERMINO share this)
    vcto_entries = interface_df[interface_df['event_type'] == 'Vcto'].copy()
    st.write(f"Found {len(vcto_entries)} Vcto entries in interface file (shared by VENCIMIENTO and TERMINO)")
    
    # Ensure numeric values
    vcto_entries[debit_col] = pd.to_numeric(vcto_entries[debit_col], errors='coerce').fillna(0)
    vcto_entries[credit_col] = pd.to_numeric(vcto_entries[credit_col], errors='coerce').fillna(0)
    
    # Prepare validation results
    validation_results = []
    
    # Track anomalies
    cartera_anomalies = []
    
    st.write(f"Processing {len(expiries_df)} trades for enhanced VENCIMIENTO validation")
    
    # Process each expiring trade
    for _, expiry in expiries_df.iterrows():
        trade_number = expiry['Número Operación']
        
        # FIXED: Handle NaN trade numbers
        if pd.isna(trade_number):
            st.warning(f"⚠️ Skipping expiry with missing trade number")
            continue
        
        # FIXED: Safely convert to integer
        try:
            trade_number = int(float(str(trade_number)))
        except (ValueError, TypeError):
            st.warning(f"⚠️ Skipping expiry with invalid trade number: {trade_number}")
            continue
        
        # Skip if not the debug deal (when in debug mode)
        if debug_deal is not None and str(trade_number) != str(debug_deal):
            continue
            
        instrument_type = expiry.get('instrument_type', 'Unknown')
        settlement_currency = expiry.get('Moneda Liquidación', 'Unknown')
        
        # NEW: Get estrategia from Cartera file
        normalized_deal = trade_number  # Already converted to int above
        estrategia = estrategia_lookup.get(normalized_deal)
        
        if estrategia is None:
            # Handle missing deal in Cartera file - mark as anomaly and skip
            cartera_anomalies.append({
                'deal_number': str(trade_number),
                'instrument_type': instrument_type,
                'issue': f'Deal {trade_number} not found in Cartera file'
            })
            
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'validation_type': 'VENCIMIENTO',
                'estrategia': 'MISSING',
                'status': 'Cartera Anomaly',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'Deal {trade_number} not found in Cartera file - cannot determine estrategia'
            })
            continue
        
        # Get the override amounts and handle NaN/null values
        monto_extranjero = expiry.get('Monto Override Extranjero', 0)
        monto_local = expiry.get('Monto Override Local', 0)
        
        # Handle NaN/NaT values
        if pd.isna(monto_extranjero):
            monto_extranjero = 0
        if pd.isna(monto_local):
            monto_local = 0
        
        # Convert to numeric
        try:
            monto_extranjero = float(monto_extranjero)
        except (ValueError, TypeError):
            monto_extranjero = 0
            
        try:
            monto_local = float(monto_local)
        except (ValueError, TypeError):
            monto_local = 0
        
        # Get Cobertura from expiries file (for backward compatibility)
        cobertura = expiry.get('Cobertura', 'No')
        
        # Determine direction and amount
        if monto_extranjero != 0:
            direction = 'Positivo' if monto_extranjero > 0 else 'Negativo'
            amount_to_use = abs(monto_extranjero)
            amount_source = 'Extranjero'
        elif monto_local != 0:
            direction = 'Positivo' if monto_local > 0 else 'Negativo'
            amount_to_use = abs(monto_local)
            amount_source = 'Local'
        else:
            direction = 'Positivo'
            amount_to_use = 0
            amount_source = 'None'
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG VENCIMIENTO: Processing expiry {trade_number}, instrument: {instrument_type}")
            st.write(f"DEBUG VENCIMIENTO: Estrategia from Cartera: {estrategia}")
            st.write(f"DEBUG VENCIMIENTO: Monto Extranjero: {monto_extranjero}, Monto Local: {monto_local}")
            st.write(f"DEBUG VENCIMIENTO: Cobertura: {cobertura}, Direction: {direction}, Using amount: {amount_to_use} (from {amount_source})")
        
        # NEW: Apply Estrategia/Coverage filtering logic
        if estrategia == "NO":
            # Filter by coverage column where value is "No"
            applicable_rules = vencimiento_rules[
                (vencimiento_rules['subproduct'] == instrument_type) & 
                (vencimiento_rules['coverage'] == 'No') &
                (vencimiento_rules['direction'] == direction)
            ]
            filter_criteria = f"coverage='No'"
        else:
            # Filter by Estrategia column where value matches the estrategia from Cartera
            applicable_rules = vencimiento_rules[
                (vencimiento_rules['subproduct'] == instrument_type) & 
                (vencimiento_rules['Estrategia'] == estrategia) &
                (vencimiento_rules['direction'] == direction)
            ]
            filter_criteria = f"Estrategia='{estrategia}'"
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'validation_type': 'VENCIMIENTO',
                'estrategia': estrategia,
                'direction': direction,
                'amount_used': amount_to_use,
                'amount_source': amount_source,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No VENCIMIENTO rule found for {instrument_type} with {filter_criteria}, Direction={direction}'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG VENCIMIENTO: Found {len(applicable_rules)} applicable rules using filter: {filter_criteria}")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
        
        # Build expected accounts (simple logic - no Pata for VENCIMIENTO)
        expected_accounts = []
        expected_amounts = {}
        
        for _, rule in applicable_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': amount_to_use, 'field': 'debit'}
            
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': amount_to_use, 'field': 'credit'}

        expected_accounts = [acc for acc in expected_accounts if acc != "None" and acc != "nan"]
        expected_entry_count = len(expected_accounts)
        
        # Find interface entries for this trade
        trade_entries = vcto_entries[vcto_entries[trade_number_col] == trade_number]

        # NEW: Build TERMINO accounts list to exclude them
        termino_rules = rules_df[rules_df['event'] == 'Termino'].copy()
        all_possible_termino_accounts = set()
        for _, rule in termino_rules.iterrows():
            if pd.notna(rule['debit_account']):
                all_possible_termino_accounts.add(str(rule['debit_account']))
            if pd.notna(rule['credit_account']):
                all_possible_termino_accounts.add(str(rule['credit_account']))

        # Filter to VENCIMIENTO entries: include accounts that are NOT TERMINO accounts
        vencimiento_entries = trade_entries[
            ~trade_entries[account_col].astype(str).isin(all_possible_termino_accounts)
        ]
        
        if debug_deal is not None:
            st.write(f"DEBUG VENCIMIENTO: Found {len(vencimiento_entries)} VENCIMIENTO entries for trade {trade_number}")
            st.write(f"DEBUG VENCIMIENTO: Expected {expected_entry_count} entries, Expected accounts: {expected_accounts}")
            if len(vencimiento_entries) > 0:
                st.dataframe(vencimiento_entries, use_container_width=True, hide_index=True)
        
        # Validation logic
        status, issue, extra_entry_count = _validate_entries_against_expected(
            vencimiento_entries, expected_accounts, expected_amounts, 
            account_col, debit_col, credit_col, debug_deal, "VENCIMIENTO"
        )
        
        validation_results.append({
            'trade_number': str(trade_number),
            'instrument_type': instrument_type,
            'validation_type': 'VENCIMIENTO',
            'estrategia': estrategia,
            'direction': direction,
            'amount_used': amount_to_use,
            'amount_source': amount_source,
            'status': status,
            'interface_entries': len(vencimiento_entries),
            'expected_entries': expected_entry_count,
            'extra_entries': extra_entry_count,
            'issue': issue
        })
    
    # Display Cartera anomalies if any
    if cartera_anomalies:
        st.error(f"⚠️ Found {len(cartera_anomalies)} expiry deals that are missing from Cartera file:")
        anomaly_df = pd.DataFrame(cartera_anomalies)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    
    # Display results
    return _display_validation_results_with_estrategia(validation_results, "VENCIMIENTO")

def _display_validation_results_with_estrategia(validation_results, validation_type):
    """Helper function to display validation results with Estrategia information"""
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
        st.write(f"Enhanced {validation_type} validation using Estrategia/Coverage filtering:")
        st.write(f"• If Estrategia='NO' → Filter rules by coverage='No'")
        st.write(f"• If Estrategia≠'NO' → Filter rules by Estrategia=estrategia_value")
        st.write("Uses whichever override amount is non-zero (Extranjero or Local)")
        st.write("Matches rules by instrument type, estrategia/coverage, and direction")
        
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Show estrategia distribution
        if 'estrategia' in validation_df.columns:
            estrategia_counts = validation_df['estrategia'].value_counts()
            st.write("**Estrategia Distribution:**", estrategia_counts.to_dict())
        
        # Calculate match statistics
        full_match_count = len(validation_df[validation_df['status'] == 'Full Match'])
        anomaly_count = len(validation_df[validation_df['status'] == 'Cartera Anomaly'])
        total_count = len(validation_df)
        match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Total {validation_type}", total_count)
        with col2:
            st.metric("Full Matches", full_match_count)
        with col3:
            st.metric("Anomalies", anomaly_count)
        with col4:
            st.metric("Match Rate", f"{match_percentage:.1f}%")
        
        # Status breakdown
        status_counts = validation_df['status'].value_counts().to_dict()
        st.write(f"**{validation_type} Status Breakdown:**", status_counts)
        
        # Download results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            f"Download {validation_type} Validation Results",
            csv,
            f"{validation_type.lower()}_validation_results.csv",
            "text/csv",
            key=f"download-csv-{validation_type.lower()}"
        )
    else:
        st.warning(f"No {validation_type} validation results generated")
    
    return validation_df

def _validate_termino_only(expiries_df, interface_df, interface_cols, rules_df, debug_deal=None):
    """
    Validate TERMINO (capital amortization) entries - the unwinding part.
    Uses Amortización Activa/Pasiva amounts with Pata-based logic.
    """
    # Filter rules for TERMINO event
    termino_rules = rules_df[rules_df['event'] == 'Termino'].copy()
    
    if len(termino_rules) == 0:
        st.error("No TERMINO rules found in rules file")
        return pd.DataFrame()
    
    st.write(f"Found {len(termino_rules)} TERMINO rules")
    if debug_deal is not None:
        st.dataframe(termino_rules, use_container_width=True, hide_index=True)
    
    # Extract needed columns from interface
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    
    # Use the same Vcto entries as VENCIMIENTO (they share the same glosa)
    vcto_entries = interface_df[interface_df['event_type'] == 'Vcto'].copy()
    vcto_entries[debit_col] = pd.to_numeric(vcto_entries[debit_col], errors='coerce').fillna(0)
    vcto_entries[credit_col] = pd.to_numeric(vcto_entries[credit_col], errors='coerce').fillna(0)
    
    validation_results = []
    
    processed_count = 0
    skipped_count = 0
    
    # Process each expiring trade
    for _, expiry in expiries_df.iterrows():
        trade_number = expiry['Número Operación']
        
        # FIXED: Handle NaN trade numbers (same as VENCIMIENTO)
        if pd.isna(trade_number):
            skipped_count += 1
            if debug_deal is not None:
                st.write(f"DEBUG TERMINO: Skipping expiry with missing trade number")
            continue
        
        # FIXED: Safely convert to integer
        try:
            trade_number = int(float(str(trade_number)))
        except (ValueError, TypeError):
            skipped_count += 1
            if debug_deal is not None:
                st.write(f"DEBUG TERMINO: Skipping expiry with invalid trade number: {trade_number}")
            continue
        
        # Skip if not the debug deal (when in debug mode)
        if debug_deal is not None and str(trade_number) != str(debug_deal):
            continue
            
        instrument_type = expiry.get('instrument_type', 'Unknown')
        
        # Get amortization amounts and handle NaN/null values
        amort_activa = expiry.get('Amortización Activa', 0)
        amort_pasiva = expiry.get('Amortización Pasiva', 0)
        
        # Handle NaN/null values
        if pd.isna(amort_activa):
            amort_activa = 0
        if pd.isna(amort_pasiva):
            amort_pasiva = 0
        
        # Convert to numeric
        try:
            amort_activa = float(amort_activa)
        except (ValueError, TypeError):
            amort_activa = 0
            
        try:
            amort_pasiva = float(amort_pasiva)
        except (ValueError, TypeError):
            amort_pasiva = 0
        
        # Skip if both amortization amounts are zero (no capital amortization)
        if amort_activa == 0 and amort_pasiva == 0:
            skipped_count += 1
            if debug_deal is not None:
                st.write(f"DEBUG TERMINO: Skipping trade {trade_number} - no amortization amounts")
            continue
        
        processed_count += 1
        
        # Get Cobertura from expiries file
        cobertura = expiry.get('Cobertura', 'No')
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG TERMINO: Processing expiry {trade_number}, instrument: {instrument_type}")
            st.write(f"DEBUG TERMINO: Amortización Activa: {amort_activa}, Amortización Pasiva: {amort_pasiva}")
            st.write(f"DEBUG TERMINO: Cobertura: {cobertura}")
        
        # Get applicable rules (no direction filter for TERMINO - it reverses INICIO)
        applicable_rules = termino_rules[
            (termino_rules['subproduct'] == instrument_type) & 
            (termino_rules['coverage'] == cobertura)
        ]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'trade_number': str(trade_number),
                'instrument_type': instrument_type,
                'validation_type': 'TERMINO',
                'cobertura': cobertura,
                'amort_activa': amort_activa,
                'amort_pasiva': amort_pasiva,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No TERMINO rule found for {instrument_type} with Cobertura={cobertura}'
            })
            continue
        
        # Apply Pata-based logic (like Day Trades)
        activa_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Activa']
        pasiva_rules = applicable_rules[applicable_rules['Pata'] == 'Pata Pasiva']
        
        if debug_deal is not None:
            st.write(f"DEBUG TERMINO: Found {len(activa_rules)} Pata Activa rules, {len(pasiva_rules)} Pata Pasiva rules")
        
        # Build expected accounts with Pata-based amounts
        expected_accounts = []
        expected_amounts = {}
        
        # Process Pata Activa rules (use Amortización Activa)
        for _, rule in activa_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(amort_activa), 'pata': 'Activa', 'field': 'debit'}
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(amort_activa), 'pata': 'Activa', 'field': 'credit'}
        
        # Process Pata Pasiva rules (use Amortización Pasiva)
        for _, rule in pasiva_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(amort_pasiva), 'pata': 'Pasiva', 'field': 'debit'}
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(amort_pasiva), 'pata': 'Pasiva', 'field': 'credit'}

        expected_accounts = [acc for acc in expected_accounts if acc != "None" and acc != "nan"]
        expected_entry_count = len(expected_accounts)
        
        # Find interface entries for this trade
        trade_entries = vcto_entries[vcto_entries[trade_number_col] == trade_number]
        
        # Filter to only entries that match our expected accounts (to separate TERMINO from VENCIMIENTO)
        if len(expected_accounts) > 0:
            termino_entries = trade_entries[trade_entries[account_col].astype(str).isin(expected_accounts)]
        else:
            termino_entries = pd.DataFrame()
        
        if debug_deal is not None:
            st.write(f"DEBUG TERMINO: Found {len(termino_entries)} TERMINO entries for trade {trade_number}")
            st.write(f"DEBUG TERMINO: Expected {expected_entry_count} entries, Expected accounts: {expected_accounts}")
            if len(termino_entries) > 0:
                st.dataframe(termino_entries, use_container_width=True, hide_index=True)
        
        # FIXED: Validation logic with Pata-aware amounts - now expecting 3 return values
        status, issue, extra_entry_count = _validate_entries_against_expected_with_pata(
            termino_entries, expected_accounts, expected_amounts, 
            account_col, debit_col, credit_col, debug_deal, "TERMINO"
        )
        
        validation_results.append({
            'trade_number': str(trade_number),
            'instrument_type': instrument_type,
            'validation_type': 'TERMINO',
            'cobertura': cobertura,
            'amort_activa': amort_activa,
            'amort_pasiva': amort_pasiva,
            'status': status,
            'interface_entries': len(termino_entries),
            'expected_entries': expected_entry_count,
            'extra_entries': extra_entry_count,  # NEW: Now capturing extra entries count
            'issue': issue
        })
    
    st.write(f"TERMINO Processing Summary: {processed_count} trades processed, {skipped_count} trades skipped (no amortization)")
    
    # Display results
    return _display_validation_results(validation_results, "TERMINO")

def _validate_entries_against_expected(entries_df, expected_accounts, expected_amounts, 
                                     account_col, debit_col, credit_col, debug_deal, validation_type):
    """
    Enhanced validation that prioritizes core business logic over entry count.
    
    Returns: (status, issue, extra_entry_count)
    
    NEW APPROACH:
    1. First check: Are ALL required entries present with correct amounts? (CORE VALIDATION)
    2. Second check: Are there extra entries beyond what's required? (DATA QUALITY)
    
    Status categories:
    - "✅ Perfect Match": Required entries correct + no extras
    - "✅ Correct (+ Extras)": Required entries correct + unnecessary extras exist
    - "❌ Missing Required": Some required entries are missing  
    - "❌ Wrong Amounts": Required entries present but wrong amounts
    - "❌ No Entries": No entries found at all
    """
    
    # Step 1: Basic check - do we have any entries?
    if len(entries_df) == 0:
        return '❌ No Entries', f'No {validation_type} entries found in interface', 0
    
    # Step 2: CORE VALIDATION - Check if ALL required accounts are present
    found_accounts = entries_df[account_col].astype(str).unique().tolist()
    missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
    
    if missing_accounts:
        extra_entry_count = len(entries_df)  # All entries are "extra" if required ones are missing
        return '❌ Missing Required', f'Missing required accounts: {", ".join(missing_accounts)}', extra_entry_count
    
    # Step 3: CORE VALIDATION - Check amounts for ALL required accounts
    core_validation_issues = []
    all_required_correct = True
    
    for account in expected_accounts:
        account_entries = entries_df[entries_df[account_col].astype(str) == account]
        expected_amount = expected_amounts[account]['amount']
        expected_field = expected_amounts[account]['field']
        
        if expected_field == 'debit':
            actual_amount = account_entries[debit_col].sum()
        else:
            actual_amount = account_entries[credit_col].sum()
        
        is_matching = abs(actual_amount - expected_amount) < 1.0
        
        if not is_matching:
            all_required_correct = False
            core_validation_issues.append(f"{account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
        
        if debug_deal is not None:
            status_icon = "✅" if is_matching else "❌"
            st.write(f"{status_icon} {validation_type} Account {account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
    
    # Step 4: If core validation failed, report that as the primary issue
    if not all_required_correct:
        extra_entry_count = max(0, len(entries_df) - len(expected_accounts))
        return '❌ Wrong Amounts', f"Required entries have wrong amounts: {'; '.join(core_validation_issues)}", extra_entry_count
    
    # Step 5: Core validation PASSED! Now check for data quality issues (extra entries)
    expected_entry_count = len(expected_accounts)
    actual_entry_count = len(entries_df)
    extra_entry_count = max(0, actual_entry_count - expected_entry_count)
    
    if extra_entry_count > 0:
        # Core is correct, but there are unnecessary extra entries
        extra_accounts = [acc for acc in found_accounts if acc not in expected_accounts]
        if extra_accounts:
            extra_info = f" Extra accounts: {', '.join(extra_accounts)}"
        else:
            extra_info = f" Duplicate entries in required accounts"
        
        if debug_deal is not None:
            st.info(f"✅ All required entries are CORRECT for {validation_type}")
            st.warning(f"⚠️ But found {extra_entry_count} unnecessary extra entries.{extra_info}")
        
        return '✅ Correct (+ Extras)', f"Required entries correct. Data quality issue: {extra_entry_count} extra entries.{extra_info}", extra_entry_count
    
    # Step 6: Perfect match - required entries present, correct, and no extras
    if debug_deal is not None:
        st.success(f"🎯 PERFECT MATCH for {validation_type} - all required entries correct, no extras!")
    
    #return '✅ Perfect Match', '', 0
    return 'Full Match', '', 0

def _validate_entries_against_expected_with_pata(entries_df, expected_accounts, expected_amounts, 
                                               account_col, debit_col, credit_col, debug_deal, validation_type):
    """
    Enhanced helper function for Pata-aware validation (TERMINO) with separated core vs extra checking.
    
    Returns: (status, issue, extra_entry_count)
    """
    if len(entries_df) == 0:
        return 'Missing Entries', f'No {validation_type} entries found in interface', 0
    
    expected_entry_count = len(expected_accounts)
    extra_entry_count = max(0, len(entries_df) - expected_entry_count)
    
    # STEP 1: Check if all required accounts are present (core validation)
    found_accounts = entries_df[account_col].astype(str).unique().tolist()
    missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
    
    if missing_accounts:
        return 'Missing Accounts', f'Missing expected accounts: {", ".join(missing_accounts)}', extra_entry_count
    
    # STEP 2: Check amounts for required accounts with Pata awareness (core validation)
    all_required_match = True
    mismatches = []
    
    for account in expected_accounts:
        account_entries = entries_df[entries_df[account_col].astype(str) == account]
        expected_amount = expected_amounts[account]['amount']
        expected_field = expected_amounts[account]['field']
        pata_type = expected_amounts[account]['pata']
        
        if expected_field == 'debit':
            actual_amount = account_entries[debit_col].sum()
        else:
            actual_amount = account_entries[credit_col].sum()
        
        is_matching = abs(actual_amount - expected_amount) < 1.0
        if not is_matching:
            all_required_match = False
            mismatches.append(f"{account} ({expected_field}, {pata_type}): Expected {expected_amount}, Found {actual_amount:.2f}")
        
        if debug_deal is not None:
            if is_matching:
                st.success(f"✓ {validation_type} Account {account} ({expected_field}, {pata_type}): Expected {expected_amount}, Found {actual_amount:.2f}")
            else:
                st.warning(f"✗ {validation_type} Account {account} ({expected_field}, {pata_type}): Expected {expected_amount}, Found {actual_amount:.2f}")
    
    # STEP 3: Determine status based on core validation + extra entries
    if not all_required_match:
        # Core validation failed - this is the most important issue
        return 'Amount Mismatch', f"Required entry mismatches: {'; '.join(mismatches)}", extra_entry_count
    
    # Core validation passed - now check for extra entries
    if extra_entry_count > 0:
        # Required entries are correct, but there are unnecessary extra entries
        extra_accounts = [acc for acc in found_accounts if acc not in expected_accounts]
        if extra_accounts:
            extra_info = f" (extra accounts: {', '.join(extra_accounts)})"
        else:
            extra_info = f" (duplicate entries in expected accounts)"
        
        return 'Correct + Extra Entries', f"All required entries correct, but {extra_entry_count} unnecessary entries found{extra_info}", extra_entry_count
    
    # Perfect match - required entries present and correct, no extras
    return 'Full Match', '', extra_entry_count

def _display_validation_results(validation_results, validation_type):
    """Helper function to display validation results"""
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
        st.write(f"Validates {validation_type} entries in the accounting interface")
        if validation_type == "VENCIMIENTO":
            st.write("Uses whichever override amount is non-zero (Extranjero or Local)")
            st.write("Matches rules by instrument type, cobertura, and direction")
        else:
            st.write("Uses Amortización Activa (Pata Activa) and Amortización Pasiva (Pata Pasiva)")
            st.write("Matches rules by instrument type and cobertura (no direction filter)")
        
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Calculate match statistics
        full_match_count = len(validation_df[validation_df['status'] == 'Full Match'])
        total_count = len(validation_df)
        match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Total {validation_type}", total_count)
        with col2:
            st.metric("Full Matches", full_match_count)
        with col3:
            st.metric("Match Rate", f"{match_percentage:.1f}%")
        
        # Status breakdown
        status_counts = validation_df['status'].value_counts().to_dict()
        st.write(f"**{validation_type} Status Breakdown:**", status_counts)
        
        # Download results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            f"Download {validation_type} Validation Results",
            csv,
            f"{validation_type.lower()}_validation_results.csv",
            "text/csv",
            key=f"download-csv-{validation_type.lower()}"
        )
    else:
        st.warning(f"No {validation_type} validation results generated")
    
    return validation_df

def validate_incumplimiento_entries(incumplimientos_df, interface_df, interface_cols, rules_df, counterparties_df=None, debug_deal=None):
    """
    Validate incumplimiento entries against accounting interface entries.
    
    For each incumplimiento event, checks that:
    1. Corresponding "Incumplimiento" entries exist in the accounting interface
    2. Correct account numbers are used based on instrument type, counterparty type, and flow currency
    3. Transaction amounts match the incumplimiento amount
    4. Amounts are in the correct fields (debe/haber)
    5. The number of entries exactly matches what's expected from rules
    
    Counterparty type is determined by checking RUT against counterparties_df:
    - If RUT found in counterparties file -> "Instituciones Financieras"
    - If RUT not found or no counterparties file -> "Otras Instituciones"
    """
    # Filter rules for INCUMPLIMIENTO event
    incumplimiento_rules = rules_df[rules_df['event'] == 'Incumplimiento'].copy()
    
    if len(incumplimiento_rules) == 0:
        st.error("No INCUMPLIMIENTO rules found in rules file")
        return pd.DataFrame()
    
    st.subheader("INCUMPLIMIENTO Rules")
    st.dataframe(incumplimiento_rules, use_container_width=True, hide_index=True)
    
    # Create RUT lookup set from counterparties if provided
    instituciones_financieras_ruts = set()
    if counterparties_df is not None and not counterparties_df.empty:
        instituciones_financieras_ruts = set(counterparties_df['rut'].astype(str).str.strip())
        st.write(f"✅ Using counterparties file with {len(instituciones_financieras_ruts)} Instituciones Financieras RUTs")
    else:
        st.warning("⚠️ No counterparties file provided - all entities will be classified as 'Otras Instituciones'")
    
    # Extract needed columns from interface
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    glosa_col = interface_cols['glosa']
    
    if not trade_number_col:
        st.error("Could not find trade number column in interface file")
        return pd.DataFrame()
    
    # Filter interface for "Incumplimiento" event_type entries
    incumplimiento_entries = interface_df[interface_df['event_type'] == 'Incumplimiento'].copy()
    st.write(f"Found {len(incumplimiento_entries)} Incumplimiento entries in interface file")
    
    # Ensure numeric values
    incumplimiento_entries[debit_col] = pd.to_numeric(incumplimiento_entries[debit_col], errors='coerce').fillna(0)
    incumplimiento_entries[credit_col] = pd.to_numeric(incumplimiento_entries[credit_col], errors='coerce').fillna(0)
    
    # Get instrument type from interface entries (extract from glosa)
    if len(incumplimiento_entries) > 0:
        # Add instrument type extraction for interface entries if not already present
        if 'instrument_type' not in incumplimiento_entries.columns:
            incumplimiento_entries['instrument_type'] = incumplimiento_entries[glosa_col].apply(lambda x: 
                'Swap Tasa' if 'Swap Tasa' in x 
                else ('Swap Moneda' if 'Swap Moneda' in x 
                else ('Swap Cámara' if 'Swap Cámara' in x or 'ICP' in x else None)))
    
    # Prepare validation results
    validation_results = []
    
    st.write(f"Processing {len(incumplimientos_df)} incumplimiento events for validation")
    
    # Process each incumplimiento event
    for _, incumplimiento in incumplimientos_df.iterrows():
        deal_number = incumplimiento['numero_operacion']
        
        # Skip if not the debug deal (when in debug mode)
        if debug_deal is not None and str(deal_number) != str(debug_deal):
            continue
            
        monto = incumplimiento['monto']
        moneda = incumplimiento['moneda']
        cliente = incumplimiento['nombre_cliente']
        rut_cliente = str(incumplimiento['rut_cliente']).strip()
        
        # NEW: Determine counterparty type based on RUT lookup
        if rut_cliente in instituciones_financieras_ruts:
            tipo_contraparte = "Instituciones Financieras"
        else:
            tipo_contraparte = "Otras Instituciones"
        
        # Map moneda_flujo based on the actual currency
        if moneda == 'CLP':
            moneda_flujo = 'CLP'
        elif moneda in ['USD', 'EUR']:
            moneda_flujo = 'MX'
        else:
            # Handle unexpected currencies - default to the actual currency
            moneda_flujo = moneda
            st.warning(f"Unexpected currency '{moneda}' for deal {deal_number}, using as-is for moneda_flujo")
        
        # Display debug info if requested
        if debug_deal is not None:
            st.write(f"DEBUG: Processing incumplimiento {deal_number}")
            st.write(f"DEBUG: RUT Cliente: {rut_cliente}")
            st.write(f"DEBUG: Tipo Contraparte: {tipo_contraparte} (determined from RUT lookup)")
            st.write(f"DEBUG: Monto: {monto}, Moneda: {moneda}")
            st.write(f"DEBUG: Moneda Flujo: {moneda_flujo} (mapped from {moneda})")
            st.write(f"DEBUG: Cliente: {cliente}")
        
        # Find corresponding interface entries for this deal
        deal_interface_entries = incumplimiento_entries[
            incumplimiento_entries[trade_number_col] == deal_number
        ].copy()
        
        if len(deal_interface_entries) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Missing Interface Entries',
                'interface_entries': 0,
                'expected_entries': 0,
                'issue': f'No Incumplimiento entries found in interface for deal {deal_number}'
            })
            continue
        
        # Extract instrument type from interface entries
        instrument_types = deal_interface_entries['instrument_type'].dropna().unique()
        
        if len(instrument_types) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Unknown Instrument Type',
                'interface_entries': len(deal_interface_entries),
                'expected_entries': 0,
                'issue': f'Could not determine instrument type from interface entries'
            })
            continue
        
        # Use the first instrument type found (assuming all entries for a deal have same instrument type)
        instrument_type = instrument_types[0]
        
        if debug_deal is not None:
            st.write(f"DEBUG: Detected instrument type: {instrument_type}")
            st.write(f"DEBUG: Found {len(deal_interface_entries)} interface entries for deal {deal_number}")
            st.dataframe(deal_interface_entries, use_container_width=True, hide_index=True)
        
        # Get applicable rules for this combination
        # Include rules where counterparty_type matches OR is blank (applies to all)
        applicable_rules = incumplimiento_rules[
            (incumplimiento_rules['subproduct'] == instrument_type) & 
            (
                (incumplimiento_rules['counterparty_type'] == tipo_contraparte) |
                (incumplimiento_rules['counterparty_type'].isna()) |
                (incumplimiento_rules['counterparty_type'] == '') |
                (incumplimiento_rules['counterparty_type'] == 'nan')
            ) &
            (incumplimiento_rules['flow_currency'] == moneda_flujo)
        ]
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Missing Rule',
                'interface_entries': len(deal_interface_entries),
                'expected_entries': 0,
                'issue': f'No rule found for {instrument_type} + {tipo_contraparte} + {moneda_flujo}'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Found {len(applicable_rules)} applicable rules:")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
        
        # Build expected accounts list from rules (no Pata logic for Incumplimiento)
        expected_accounts = []
        expected_amounts = {}
        
        for _, rule in applicable_rules.iterrows():
            if pd.notna(rule['debit_account']):
                account = str(rule['debit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(monto), 'field': 'debit'}
            
            if pd.notna(rule['credit_account']):
                account = str(rule['credit_account'])
                expected_accounts.append(account)
                expected_amounts[account] = {'amount': abs(monto), 'field': 'credit'}

        # Remove any "None" values
        expected_accounts = [acc for acc in expected_accounts if acc != "None" and acc != "nan"]
        expected_entry_count = len(expected_accounts)
        
        if debug_deal is not None:
            st.write(f"DEBUG: Expected {expected_entry_count} entries")
            st.write(f"DEBUG: Expected account numbers: {expected_accounts}")
            st.write("DEBUG: Expected amounts by account:")
            for acc, info in expected_amounts.items():
                st.write(f"  {acc}: {info['amount']} ({info['field']})")
        
        # Check if we found exactly the right number of entries
        if len(deal_interface_entries) != expected_entry_count:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Entry Count Mismatch',
                'interface_entries': len(deal_interface_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Expected {expected_entry_count} entries, found {len(deal_interface_entries)}'
            })
            continue
            
        # Check if each expected account is present
        found_accounts = deal_interface_entries[account_col].astype(str).unique().tolist()
        missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
        extra_accounts = [acc for acc in found_accounts if acc not in expected_accounts]

        if missing_accounts:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Missing Accounts',
                'interface_entries': len(deal_interface_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Missing expected accounts: {", ".join(missing_accounts)}'
            })
            continue

        if extra_accounts:
            validation_results.append({
                'deal_number': str(deal_number),
                'cliente': cliente,
                'rut_cliente': rut_cliente,
                'tipo_contraparte': tipo_contraparte,
                'moneda_flujo': moneda_flujo,
                'moneda_original': moneda,
                'monto': monto,
                'status': 'Extra Accounts',
                'interface_entries': len(deal_interface_entries),
                'expected_entries': expected_entry_count,
                'issue': f'Found unexpected accounts: {", ".join(extra_accounts)}'
            })
            continue
        
        # At this point we have the right number of entries with the right accounts
        # Now check the amounts in each entry
        account_validation = {}
        for account in expected_accounts:
            account_entries = deal_interface_entries[deal_interface_entries[account_col].astype(str) == account]
            
            # Get expected amount and field from our mapping
            expected_info = expected_amounts[account]
            expected_amount = expected_info['amount']
            expected_field = expected_info['field']

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
                'matches': is_matching
            }
            
            if debug_deal is not None:
                if is_matching:
                    st.success(f"✓ Account {account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
                else:
                    st.warning(f"✗ Account {account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
        
        # Check if all amounts match
        all_match = all(val['matches'] for val in account_validation.values())
        
        if all_match:
            status = 'Full Match'
            issue = ''
        else:
            status = 'Amount Mismatch'
            mismatches = [
                f"{account} ({val['expected_field']}): Expected {val['expected_amount']}, Found {val['actual_amount']:.2f}"
                for account, val in account_validation.items() if not val['matches']
            ]
            issue = f"Amount mismatches: {'; '.join(mismatches)}"
        
        # Add to results
        validation_results.append({
            'deal_number': str(deal_number),
            'cliente': cliente,
            'rut_cliente': rut_cliente,
            'tipo_contraparte': tipo_contraparte,
            'moneda_flujo': moneda_flujo,
            'moneda_original': moneda,
            'monto': monto,
            'status': status,
            'interface_entries': len(deal_interface_entries),
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
        st.subheader("Incumplimiento Validation Results")
        st.write("Validates that incumplimiento events have corresponding entries in the accounting interface")
        st.write("Counterparty type determined by RUT lookup in counterparties file:")
        st.write("• If RUT found in counterparties → 'Instituciones Financieras'")
        st.write("• If RUT not found → 'Otras Instituciones'")
        st.write("Flow currency mapping: CLP→CLP, USD/EUR→MX")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Show counterparty type breakdown
        if 'tipo_contraparte' in validation_df.columns:
            tipo_counts = validation_df['tipo_contraparte'].value_counts()
            st.write("**Counterparty Type Distribution:**", tipo_counts.to_dict())
        
        # Calculate match statistics
        full_match_count = len(validation_df[validation_df['status'] == 'Full Match'])
        total_count = len(validation_df)
        match_percentage = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Incumplimientos", total_count)
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
            "Download Incumplimiento Validation Results",
            csv,
            "incumplimiento_validation_results.csv",
            "text/csv",
            key="download-csv-incumplimiento"
        )
    else:
        st.warning("No incumplimiento validation results generated")
    
    return validation_df

def validate_mtm_entries_new_format(interface_df, interface_cols, cartera_treasury_raw_df, cartera_treasury_processed_df, 
                                   rules_df, cartera_treasury_raw_df_for_estrategia=None,
                                   event_type='Valorización MTM', 
                                   rules_event_type=None,
                                   key_suffix='', 
                                   debug_deal=None,
                                   combined_mtm_mode=False):
    """
    Enhanced MTM validation using the new Cartera Treasury format for both MTM values and estrategia data.
    
    Args:
        interface_df: Accounting interface dataframe
        interface_cols: Column mapping for interface file
        cartera_treasury_raw_df: Raw Cartera Treasury dataframe (for instrument type detection)
        cartera_treasury_processed_df: Processed dataframe with deal_number, total_mtm, direction
        rules_df: Accounting rules dataframe
        cartera_treasury_raw_df_for_estrategia: Raw Cartera Treasury dataframe for estrategia extraction (defaults to cartera_treasury_raw_df)
        event_type: Event type for interface filtering
        rules_event_type: Event type for rules filtering (defaults to event_type)
        key_suffix: Suffix for unique keys
        debug_deal: Specific deal to debug
        combined_mtm_mode: Whether both MTM and reversal are expected in same interface
    
    Returns:
        pd.DataFrame: Validation results
    """
    
    # Use rules_event_type if provided, otherwise default to event_type
    filter_event = rules_event_type if rules_event_type else event_type
    
    # Filter rules using the appropriate event type
    mtm_rules = rules_df[rules_df['event'] == filter_event].copy()
    
    if debug_deal is not None:
        st.write(f"DEBUG: Analyzing deal number {debug_deal}")
    
    st.write(f"Found {len(mtm_rules)} {event_type} rules")
    st.write(f"Processing enhanced MTM validation with new Cartera Treasury format")
    st.write(f"Using consolidated MTM values and estrategia data from Cartera Treasury file")
    
    # Display MTM rules
    st.subheader(f"{event_type} Rules")
    st.dataframe(mtm_rules, use_container_width=True, hide_index=True)
    
    if len(mtm_rules) == 0:
        st.error(f"No {event_type} rules found in rules file")
        return pd.DataFrame()
    
    # Extract estrategia data from Cartera Treasury file (replaces separate Cartera Analytics file)
    estrategia_source_df = cartera_treasury_raw_df_for_estrategia if cartera_treasury_raw_df_for_estrategia is not None else cartera_treasury_raw_df
    cartera_df = extract_estrategia_from_cartera_treasury(estrategia_source_df)
    
    if cartera_df is None or cartera_df.empty:
        st.error("❌ Could not extract estrategia data from Cartera Treasury file")
        st.error("Estrategia data is required for MTM validation")
        return pd.DataFrame()
    
    # Create estrategia lookup from extracted Cartera Treasury data
    estrategia_lookup = dict(zip(cartera_df['deal_number'], cartera_df['estrategia']))
    st.write(f"✅ Using estrategia data extracted from Cartera Treasury file for {len(estrategia_lookup)} deals")
    
    # Check if rules have required columns
    required_rule_columns = ['coverage', 'Estrategia']
    missing_rule_columns = [col for col in required_rule_columns if col not in mtm_rules.columns]
    
    if missing_rule_columns:
        st.error(f"❌ Missing required columns in rules file: {', '.join(missing_rule_columns)}")
        st.error("Rules file must contain both 'coverage' and 'Estrategia' columns for MTM validation")
        return pd.DataFrame()
    
    # Find trade number column in interface file
    trade_number_col = next((col for col in interface_df.columns if any(x in str(col).lower() for x in ['operación', 'operacion', 'nro.'])), None)
    
    if not trade_number_col:
        st.warning("Could not find trade number column in interface file")
        return pd.DataFrame()
    
    # Extract account interface entries for specified event type
    mtm_entries = interface_df[interface_df['event_type'] == event_type].copy()
    debit_col = interface_cols['debit']
    credit_col = interface_cols['credit']
    account_col = interface_cols['account']
    
    # Ensure numeric values in interface
    mtm_entries[debit_col] = pd.to_numeric(mtm_entries[debit_col], errors='coerce').fillna(0)
    mtm_entries[credit_col] = pd.to_numeric(mtm_entries[credit_col], errors='coerce').fillna(0)
    
    st.write(f"Found {len(mtm_entries)} {event_type} entries in interface file")
    st.write(f"Found {len(cartera_treasury_processed_df)} deals in Cartera Treasury file")
    
    # Initialize validation results
    validation_results = []
    
    # Track anomalies
    cartera_anomalies = []
    
    # Process each deal from Cartera Treasury file
    for _, deal_row in cartera_treasury_processed_df.iterrows():
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
        
        # Try to determine instrument type from Cartera Treasury file
        instrument_type = detect_instrument_type_from_cartera_treasury(cartera_treasury_raw_df, deal_number)
        
        # If we can't determine from Cartera Treasury, we'll need to skip or use a fallback
        if not instrument_type:
            if debug_deal is not None:
                st.warning(f"Could not determine instrument type for deal {deal_number} from Cartera Treasury file")
            
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': 'UNKNOWN',
                'direction': direction,
                'mtm_value': mtm_abs,
                'estrategia': 'UNKNOWN',
                'status': 'Unknown Instrument',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': 0,
                'extra_entries': 0,
                'issue': f'Could not determine instrument type for deal {deal_number} from Cartera Treasury file'
            })
            continue
        
        # Get estrategia from extracted Cartera Treasury data
        estrategia = estrategia_lookup.get(normalized_deal)
        
        if estrategia is None:
            # Handle missing deal in extracted estrategia data - mark as anomaly and skip
            cartera_anomalies.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'issue': f'Deal {deal_number} not found in extracted estrategia data from Cartera Treasury file'
            })
            
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'estrategia': 'MISSING',
                'status': 'Cartera Anomaly',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': 0,
                'extra_entries': 0,
                'issue': f'Deal {deal_number} not found in extracted estrategia data - cannot determine estrategia'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Processing deal {deal_number}, instrument: {instrument_type}, direction: {direction}, MTM: {mtm_abs:.2f}")
            st.write(f"DEBUG: Estrategia from Cartera Treasury: {estrategia}")
        
        # Apply Estrategia/Cobertura filtering logic (enhanced to handle new MX values)
        if estrategia == "NO":
            # Filter by coverage column where value is "No"
            applicable_rules = mtm_rules[
                (mtm_rules['subproduct'] == instrument_type) & 
                (mtm_rules['direction'].str.upper() == direction) &
                (mtm_rules['coverage'] == 'No')
            ]
            filter_criteria = f"coverage='No'"
        else:
            # Filter by Estrategia column where value matches the estrategia from Cartera Treasury
            # This now includes the new MX values: COB_MX_ACTIVOS, COB_MX_PASIVOS
            applicable_rules = mtm_rules[
                (mtm_rules['subproduct'] == instrument_type) & 
                (mtm_rules['direction'].str.upper() == direction) &
                (mtm_rules['Estrategia'] == estrategia)
            ]
            filter_criteria = f"Estrategia='{estrategia}'"
        
        if len(applicable_rules) == 0:
            validation_results.append({
                'deal_number': str(deal_number),
                'instrument_type': instrument_type,
                'direction': direction,
                'mtm_value': mtm_abs,
                'estrategia': estrategia,
                'status': 'Missing Rule',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': 0,
                'extra_entries': 0,
                'issue': f'No rule found for {instrument_type} with direction {direction} and {filter_criteria}'
            })
            continue
        
        if debug_deal is not None:
            st.write(f"DEBUG: Found {len(applicable_rules)} applicable rules using filter: {filter_criteria}")
            st.dataframe(applicable_rules, use_container_width=True, hide_index=True)
            st.write(f"DEBUG: Combined MTM mode: {combined_mtm_mode}")
        
        # Extract expected accounts from rules (same logic as before)
        expected_accounts = []
        expected_amounts = {}
        
        for _, rule in applicable_rules.iterrows():
            # Add debit account if present and not already added
            if pd.notna(rule['debit_account']):
                debit_account = str(rule['debit_account'])
                if debit_account not in expected_accounts:
                    expected_accounts.append(debit_account)
                    expected_amounts[debit_account] = {'amount': mtm_abs, 'field': 'debit'}
            
            # Add credit account if present and not already added
            if pd.notna(rule['credit_account']):
                credit_account = str(rule['credit_account'])
                if credit_account not in expected_accounts:
                    expected_accounts.append(credit_account)
                    expected_amounts[credit_account] = {'amount': mtm_abs, 'field': 'credit'}
        
        if debug_deal is not None:
            st.write(f"Expected accounts: {expected_accounts}")
            st.write("Expected amounts by account:")
            for acc, info in expected_amounts.items():
                st.write(f"  {acc}: {info['amount']} ({info['field']})")
        
        # Find matching entries in the interface file
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
                'estrategia': estrategia,
                'status': 'Missing Entries',
                'interface_entries': 0,
                'matched_entries': 0,
                'expected_accounts': len(expected_accounts),
                'extra_entries': 0,
                'issue': f'No entries found in interface file for deal {deal_number}'
            })
            continue
        
        # Use the same validation logic as before
        if combined_mtm_mode:
            # In combined mode, expect entries for both current and reversal MTM
            adjusted_expected_count = len(expected_accounts) * 2  # Expect double entries
            
            if debug_deal is not None:
                st.write(f"DEBUG: Combined mode - expecting {adjusted_expected_count} total entries (2x {len(expected_accounts)})")
                st.write(f"DEBUG: Found {len(deal_entries)} actual entries")
            
            # Check core validation manually without flagging extras
            # Step 1: Check if all required accounts are present
            found_accounts = deal_entries[account_col].astype(str).unique().tolist()
            missing_accounts = [acc for acc in expected_accounts if acc not in found_accounts]
            
            if missing_accounts:
                status = '❌ Missing Required'
                issue = f'Missing required accounts: {", ".join(missing_accounts)}'
                extra_entry_count = len(deal_entries)
            else:
                # Step 2: Check amounts for required accounts
                all_required_correct = True
                core_validation_issues = []
                
                for account in expected_accounts:
                    account_entries = deal_entries[deal_entries[account_col].astype(str) == account]
                    expected_amount = expected_amounts[account]['amount']
                    expected_field = expected_amounts[account]['field']
                    
                    if expected_field == 'debit':
                        actual_amount = account_entries[debit_col].sum()
                    else:
                        actual_amount = account_entries[credit_col].sum()
                    
                    is_matching = abs(actual_amount - expected_amount) < 1.0
                    
                    if not is_matching:
                        all_required_correct = False
                        core_validation_issues.append(f"{account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
                    
                    if debug_deal is not None:
                        status_icon = "✅" if is_matching else "❌"
                        st.write(f"{status_icon} Combined MTM Account {account} ({expected_field}): Expected {expected_amount}, Found {actual_amount:.2f}")
                
                # Step 3: Determine status based on combined mode expectations
                if not all_required_correct:
                    status = '❌ Wrong Amounts'
                    issue = f"Required entries have wrong amounts: {'; '.join(core_validation_issues)}"
                    extra_entry_count = max(0, len(deal_entries) - adjusted_expected_count)
                else:
                    # Core validation passed - check entry count for combined mode
                    extra_entry_count = max(0, len(deal_entries) - adjusted_expected_count)
                    
                    if extra_entry_count == 0:
                        status = '✅ Perfect Match'
                        issue = 'All required entries correct (combined MTM mode)'
                        if debug_deal is not None:
                            st.success(f"🎯 PERFECT COMBINED MATCH - found exactly {len(deal_entries)} entries as expected for MTM+Reversal")
                    else:
                        status = '✅ Correct (+ Extras)'
                        issue = f'Required entries correct, but {extra_entry_count} extra entries beyond expected combined total'
                        if debug_deal is not None:
                            st.info(f"✅ All required entries CORRECT for combined MTM")
                            st.warning(f"⚠️ But found {extra_entry_count} entries beyond the expected {adjusted_expected_count} for MTM+Reversal")
        else:
            # Normal mode - use the existing validation helper function
            status, issue, extra_entry_count = _validate_entries_against_expected(
                deal_entries, expected_accounts, expected_amounts, 
                account_col, debit_col, credit_col, debug_deal, "MTM"
            )
        
        # Count correctly matched entries for the matched_entries column
        correctly_matched_entries = 0
        if status in ['✅ Perfect Match', '✅ Correct (+ Extras)']:
            correctly_matched_entries = len(expected_accounts)
        elif 'Amount Mismatch' not in status:
            # For other statuses, count how many accounts are actually present
            found_accounts = deal_entries[account_col].astype(str).unique().tolist()
            correctly_matched_entries = len([acc for acc in expected_accounts if acc in found_accounts])
        
        # Add to validation results
        validation_results.append({
            'deal_number': str(deal_number),
            'instrument_type': instrument_type,
            'direction': direction,
            'mtm_value': mtm_abs,
            'estrategia': estrategia,
            'status': status,
            'interface_entries': len(deal_entries),
            'matched_entries': correctly_matched_entries,
            'expected_accounts': len(expected_accounts),
            'extra_entries': extra_entry_count,
            'issue': issue
        })
    
    # Display Cartera anomalies if any
    if cartera_anomalies:
        st.error(f"⚠️ Found {len(cartera_anomalies)} deals in Cartera Treasury file with estrategia extraction issues:")
        anomaly_df = pd.DataFrame(cartera_anomalies)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    
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
        st.subheader(f"{event_type} Validation Results (Cartera Treasury Integrated)")
        st.write(f"Enhanced validation using Cartera Treasury file for both MTM values and estrategia data:")
        st.write(f"• **No separate Cartera Analytics file needed** - everything from Cartera Treasury")
        st.write(f"• **Enhanced estrategia support** - includes new MX values (COB_MX_ACTIVOS/COB_MX_PASIVOS)")
        st.write(f"• **✅ Perfect Match**: Required entries correct, no extras")
        st.write(f"• **✅ Correct (+ Extras)**: Required entries correct, but unnecessary extras exist")
        st.write(f"• **❌ Wrong Amounts**: Required entries have wrong amounts")
        st.write(f"• **❌ Missing Required/Entries**: Required entries missing")
        st.write(f"• If Estrategia='NO' → Filter rules by coverage='No'")
        st.write(f"• If Estrategia≠'NO' → Filter rules by Estrategia=estrategia_value")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        # Show estrategia distribution including new MX values
        if 'estrategia' in validation_df.columns:
            estrategia_counts = validation_df['estrategia'].value_counts()
            st.write("**Estrategia Distribution (from Cartera Treasury):**", estrategia_counts.to_dict())
            
            # Highlight new MX values
            mx_deals = validation_df[validation_df['estrategia'].str.contains('COB_MX', na=False)]
            if len(mx_deals) > 0:
                st.info(f"ℹ️ Found {len(mx_deals)} deals using new MX estrategia values")
        
        # Calculate and display match statistics with enhanced categories
        full_match_count = len(validation_df[validation_df['status'] == '✅ Perfect Match'])
        correct_plus_extra_count = len(validation_df[validation_df['status'] == '✅ Correct (+ Extras)'])
        amount_mismatch_count = len(validation_df[validation_df['status'] == '❌ Wrong Amounts'])
        missing_entries_count = len(validation_df[validation_df['status'] == '❌ No Entries'])
        missing_accounts_count = len(validation_df[validation_df['status'] == '❌ Missing Required'])
        anomaly_count = len(validation_df[validation_df['status'] == 'Cartera Anomaly'])
        missing_rule_count = len(validation_df[validation_df['status'] == 'Missing Rule'])
        unknown_instrument_count = len(validation_df[validation_df['status'] == 'Unknown Instrument'])
        total_count = len(validation_df)
        
        # Calculate success rates
        core_success_rate = ((full_match_count + correct_plus_extra_count) / total_count * 100) if total_count > 0 else 0
        perfect_match_rate = (full_match_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Deals", total_count)
        with col2:
            st.metric("Perfect Matches", full_match_count)
        with col3:
            st.metric("Correct + Extra", correct_plus_extra_count)
        with col4:
            st.metric("Amount Issues", amount_mismatch_count)
        with col5:
            st.metric("Missing Issues", missing_entries_count + missing_accounts_count)
        with col6:
            st.metric("Core Success Rate", f"{core_success_rate:.1f}%")
        
        # Enhanced status breakdown
        st.subheader(f"{event_type} Enhanced Status Breakdown (Cartera Treasury Integrated)")
        st.write("**Primary Concerns (Core Business Logic):**")
        st.write(f"• ✅ Perfect Match: {full_match_count}")
        st.write(f"• ✅ Correct + Extra Entries: {correct_plus_extra_count}")
        st.write(f"• ❌ Wrong Amounts: {amount_mismatch_count}")
        st.write(f"• ❌ Missing Required: {missing_accounts_count}")
        st.write(f"• ❌ No Entries: {missing_entries_count}")
        
        st.write("**Secondary Issues:**")
        st.write(f"• Cartera Anomaly: {anomaly_count}")
        st.write(f"• Missing Rule: {missing_rule_count}")
        st.write(f"• Unknown Instrument: {unknown_instrument_count}")
        
        # Show summary of extra entries if any exist
        total_extra_entries = validation_df[validation_df['status'] == '✅ Correct (+ Extras)']['extra_entries'].sum()
        if total_extra_entries > 0:
            st.info(f"📊 **Data Quality Summary**: {total_extra_entries} total unnecessary entries found across {correct_plus_extra_count} deals")
            st.write("These deals have correct accounting but also contain superfluous entries that should be cleaned up.")
        elif correct_plus_extra_count > 0:
            st.info(f"📊 **Data Quality Summary**: {correct_plus_extra_count} deals have correct core accounting with some extra entries to clean up.")
            
        # Allow downloading results
        csv = validation_df.to_csv().encode('utf-8')
        st.download_button(
            f"Download Enhanced {event_type} Results (Cartera Treasury Integrated)",
            csv,
            f"enhanced_{event_type.lower().replace(' ', '_')}_validation_results_cartera_treasury.csv",
            "text/csv",
            key=f'download-csv-enhanced-cartera-treasury-{event_type.lower().replace(" ", "_")}{key_suffix}'
        )
    else:
        st.warning(f"No {event_type} validation results generated")
    
    return validation_df