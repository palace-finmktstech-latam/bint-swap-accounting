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
        
        # Extract event type - Updated to handle INCUMPLIMIENTO
        df['event_type'] = df[glosa_col].apply(lambda x: 
            'Incumplimiento' if 'Incumplimiento' in x
            else ('Valorización MTM' if 'Valorización' in x or 'MTM' in x 
            else ('Curse' if 'Curse' in x 
            else ('Vcto' if 'Vcto' in x or 'Vencimiento' in x  # Both VENCIMIENTO and TERMINO use Vcto/Vencimiento glosa
            else ('Reversa Valorización MTM' if 'Reversa' in x or 'Reverso' in x 
            else None)))))
        
        # Extract instrument type
        df['instrument_type'] = df[glosa_col].apply(lambda x: 
            'Swap Tasa' if 'Swap Tasa' in x 
            else ('Swap Moneda' if 'Swap Moneda' in x 
            else ('Swap Cámara' if 'Swap Cámara' in x or 'ICP' in x else None)))
        
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
        
        # Show additional info about different entry types
        vcto_count = len(df[df['event_type'] == 'Vcto'])
        incumplimiento_count = len(df[df['event_type'] == 'Incumplimiento'])
        
        if vcto_count > 0:
            st.info(f"ℹ️ Found {vcto_count} 'Vcto' entries (includes both VENCIMIENTO and TERMINO entries)")
        if incumplimiento_count > 0:
            st.info(f"ℹ️ Found {incumplimiento_count} 'Incumplimiento' entries")
    
    return df, key_columns

def parse_incumplimientos_file(file):
    """Parse the incumplimientos Excel file."""
    try:
        # Get file name for logging
        file_name = getattr(file, 'name', 'unknown')
        st.write(f"Attempting to parse Incumplimientos file: {file_name}")
        
        # Try different engines
        engines_to_try = ['openpyxl', 'xlrd']
        df_preview = None
        df = None
        
        for engine in engines_to_try:
            try:
                st.write(f"Trying to read Incumplimientos file with {engine} engine...")
                
                # First, let's look at a preview to identify the header row
                df_preview = pd.read_excel(file, header=None, nrows=15, engine=engine)
                
                # Based on the analysis, headers are on row 11 (index 10)
                file.seek(0)
                df = pd.read_excel(file, header=10, engine=engine)
                
                st.success(f"✅ Successfully parsed Incumplimientos file using {engine} engine")
                break
                
            except Exception as e:
                st.warning(f"Failed with {engine} engine: {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
                continue
        
        # If all engines failed
        if df is None or df_preview is None:
            st.error("❌ Could not parse the Incumplimientos file with any Excel engine")
            return pd.DataFrame()
        
        # Convert to string for display
        df_preview_display = df_preview.astype(str)
        st.write("Incumplimientos file preview (first 15 rows):")
        st.dataframe(df_preview_display)
        
        # Show columns found
        st.write("Incumplimientos file columns:", df.columns.tolist())
        
        # Check for required columns based on the analysis
        required_columns = [
            'Fecha y Hora', 'Número de operación', 'Monto', 'Moneda', 
            'rut', 'Cliente', 'Acción', 'Usuario que realizó la acción'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in incumplimientos file: {', '.join(missing_columns)}")
            
            # Show available columns for debugging
            st.write("Available columns:", df.columns.tolist())
            
            # Try to find similar column names
            for missing_col in missing_columns:
                possible_matches = [col for col in df.columns if missing_col.lower() in col.lower()]
                if possible_matches:
                    st.write(f"Possible matches for '{missing_col}': {possible_matches}")
            
            return pd.DataFrame()
        
        # Clean and standardize column names for easier processing
        column_mapping = {
            'Fecha y Hora': 'fecha_hora',
            'Número de operación': 'numero_operacion', 
            'Monto': 'monto',
            'Moneda': 'moneda',
            'rut': 'rut_cliente',
            'Cliente': 'nombre_cliente',
            'Acción': 'accion',
            'Usuario que realizó la acción': 'usuario'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert data types
        df['numero_operacion'] = pd.to_numeric(df['numero_operacion'], errors='coerce')
        df['monto'] = pd.to_numeric(df['monto'], errors='coerce')
        
        # Convert fecha_hora to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['fecha_hora']):
            try:
                # Try to convert from Excel date format (days since 1900-01-01)
                df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], origin='1899-12-30', unit='D')
            except:
                try:
                    # Fallback to regular datetime parsing
                    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
                except:
                    st.warning("Could not convert fecha_hora to datetime format")
        
        # Filter only for Incumplimiento actions
        df = df[df['accion'].str.contains('Incumplimiento', case=False, na=False)].copy()
        
        # Ensure string formatting for text fields
        df['rut_cliente'] = df['rut_cliente'].astype(str)
        df['nombre_cliente'] = df['nombre_cliente'].astype(str)
        df['moneda'] = df['moneda'].astype(str)
        df['usuario'] = df['usuario'].astype(str)
        
        # Add counterparty type classification (this will need to be enhanced based on business rules)
        # For now, we'll create a simple classification based on RUT or client name patterns
        def classify_counterparty_type(row):
            cliente = str(row['nombre_cliente']).lower()
            if 'banco' in cliente or 'bank' in cliente:
                return 'Banco'
            elif 'corredora' in cliente or 'securities' in cliente:
                return 'Corredora'
            elif 'seguros' in cliente or 'insurance' in cliente:
                return 'Seguros'
            elif 'pension' in cliente or 'afp' in cliente:
                return 'AFP'
            else:
                return 'Otro'
        
        df['tipo_contraparte'] = df.apply(classify_counterparty_type, axis=1)
        
        # Add flow currency (moneda_flujo) - for now, same as moneda but can be enhanced
        df['moneda_flujo'] = df['moneda']
        
        # Display summary information
        st.write(f"Found {len(df)} incumplimiento events")
        
        # Show currency distribution
        currency_counts = df['moneda'].value_counts()
        st.write("Currency distribution:", currency_counts.to_dict())
        
        # Show counterparty type distribution
        counterparty_counts = df['tipo_contraparte'].value_counts()
        st.write("Counterparty type distribution:", counterparty_counts.to_dict())
        
        # Show user distribution
        user_counts = df['usuario'].value_counts()
        st.write("User distribution:", user_counts.to_dict())
        
        # Display preview of the parsed data
        st.subheader("Incumplimientos Preview")
        display_columns = [
            'fecha_hora', 'numero_operacion', 'monto', 'moneda', 
            'nombre_cliente', 'tipo_contraparte', 'moneda_flujo', 'usuario'
        ]
        st.dataframe(df[display_columns].head(10), use_container_width=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing incumplimientos file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def parse_mtm_file(file):
    """
    DEPRECATED: Use parse_cartera_treasury_file instead.
    
    This function is kept for backward compatibility but will show a warning.
    """
    st.warning("⚠️ parse_mtm_file is deprecated. Please use parse_cartera_treasury_file instead.")
    st.warning("The new function expects the consolidated Cartera Treasury format.")
    
    # Try to parse as old format for now, but recommend migration
    return parse_cartera_treasury_file(file, "Legacy")

def parse_cartera_treasury_file(file, file_type="T0"):
    """
    Parse the new Cartera Treasury Excel file format.
    
    This replaces the old parse_mtm_file function for the new consolidated format.
    
    Args:
        file: Uploaded file object
        file_type: "T0" for current day, "T-1" for previous day (for logging purposes)
    
    Returns:
        tuple: (raw_df, processed_df) where processed_df has the format expected by validation
    """
    # Get file name for logging
    file_name = getattr(file, 'name', 'unknown')
    st.write(f"Attempting to parse Cartera Treasury file ({file_type}): {file_name}")
    
    # Try to read with different engines
    df = None
    parsing_engine = None
    
    engines_to_try = ['openpyxl', 'xlrd']
    
    for engine in engines_to_try:
        try:
            st.write(f"Trying to read Cartera Treasury file with {engine} engine...")
            
            # Data starts on row 11 (index 10) with headers
            df = pd.read_excel(file, header=10, engine=engine)
            
            parsing_engine = engine
            st.success(f"✅ Successfully parsed Cartera Treasury file using {engine} engine")
            break
            
        except Exception as e:
            st.warning(f"Failed with {engine} engine: {str(e)}")
            file.seek(0)  # Reset file pointer for next attempt
            continue
    
    # If all engines failed
    if df is None:
        st.error("❌ Could not parse the Cartera Treasury file with any Excel engine")
        st.error("Please ensure the file is a valid Excel (.xlsx/.xls) file")
        return pd.DataFrame(), pd.DataFrame()
    
    # Show original columns
    st.write("Cartera Treasury file columns:", df.columns.tolist())
    
    # Find the key columns
    deal_col = None
    mtm_col = None
    
    for col in df.columns:
        col_str = str(col).lower()
        if 'número' in col_str and 'operación' in col_str:
            deal_col = col
        elif 'valor m2m' in col_str and 'clp' in col_str:
            mtm_col = col
    
    if not deal_col:
        st.error("Could not find 'Número Operación' column in Cartera Treasury file")
        return pd.DataFrame(), pd.DataFrame()
    
    if not mtm_col:
        st.error("Could not find 'Valor M2M (CLP)' column in Cartera Treasury file")
        return pd.DataFrame(), pd.DataFrame()
    
    st.write(f"✅ Found key columns: Deal='{deal_col}', MTM='{mtm_col}'")
    
    # Clean the data
    # Remove rows where deal number is NaN
    df = df.dropna(subset=[deal_col]).copy()
    
    # Convert deal numbers to numeric
    df[deal_col] = pd.to_numeric(df[deal_col], errors='coerce')
    df = df.dropna(subset=[deal_col]).copy()
    
    # Convert MTM values to numeric
    df[mtm_col] = pd.to_numeric(df[mtm_col], errors='coerce')
    df = df.fillna({mtm_col: 0}).copy()
    
    # Create the processed dataframe in the format expected by validation
    processed_df = pd.DataFrame({
        'deal_number': df[deal_col].astype(int),
        'total_mtm': df[mtm_col].round(0),  # Round to integers like before
    })
    
    # Add direction column
    processed_df['direction'] = processed_df['total_mtm'].apply(
        lambda x: 'POSITIVO' if x > 0 else 'NEGATIVO'
    )
    
    # Display summary
    st.write(f"Found {len(processed_df)} deals with MTM values")
    
    # Show direction distribution
    direction_counts = processed_df['direction'].value_counts()
    st.write("MTM direction counts:", direction_counts.to_dict())
    
    # Show sample of processed data
    st.write("Sample of processed MTM data:")
    sample_df = processed_df.head().copy()
    sample_df['deal_number'] = sample_df['deal_number'].astype(str)
    st.dataframe(sample_df)
    
    # Show basic statistics
    total_positive = len(processed_df[processed_df['total_mtm'] > 0])
    total_negative = len(processed_df[processed_df['total_mtm'] < 0])
    total_zero = len(processed_df[processed_df['total_mtm'] == 0])
    
    st.write(f"📊 MTM Statistics:")
    st.write(f"• Positive MTM: {total_positive} deals")
    st.write(f"• Negative MTM: {total_negative} deals")
    st.write(f"• Zero MTM: {total_zero} deals")
    
    return df, processed_df

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
    """Parse the expiries Excel file with correct column expectations."""
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
                
                # Reset file pointer and read with header on row 10 (index 10)
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
        
        # CORRECTED: Only check for columns that actually exist in the expiries file
        # The amortization and cobertura columns will come from the complementary file
        required_columns = [
            'Número Operación', 'Fecha Vencimiento', 'Fecha Pago Capital', 
            'Fecha Pago Interés', 'Moneda Liquidación', 
            'Monto Override Extranjero', 'Monto Override Local'
            # ❌ REMOVED: 'Amortización Activa', 'Amortización Pasiva', 'Cobertura'
            # These come from the complementary file!
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
        
        # ✅ DO NOT add the missing columns here - they will be added during enrichment
        
        # Display summary information
        st.write(f"Found {len(df)} expiring trades")
        
        # Show payment date summary
        today = pd.Timestamp.now().normalize()
        
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
            # ❌ REMOVED: 'Amortización Activa', 'Amortización Pasiva', 'Cobertura'
        ]
        display_columns = [col for col in display_columns if col is not None and col in df.columns]
        
        st.dataframe(df[display_columns].head(10), use_container_width=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing expiries file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def parse_expiry_complementary_file(file):
    """Parse the expiry complementary Excel file with additional information."""
    
    # Get file name for logging
    file_name = getattr(file, 'name', 'unknown')
    st.write(f"Attempting to parse Expiry Complementary file: {file_name}")
    
    # Try different engines
    engines_to_try = ['openpyxl', 'xlrd']
    df = None
    
    for engine in engines_to_try:
        try:
            st.write(f"Trying to read Expiry Complementary file with {engine} engine...")
            
            # Read the Excel file - assuming headers on first row
            df = pd.read_excel(file, header=0, engine=engine)
            
            st.success(f"✅ Successfully parsed Expiry Complementary file using {engine} engine")
            break
            
        except Exception as e:
            st.warning(f"Failed with {engine} engine: {str(e)}")
            file.seek(0)  # Reset file pointer for next attempt
            continue
    
    # If all engines failed
    if df is None:
        st.error("❌ Could not parse the Expiry Complementary file with any Excel engine")
        return pd.DataFrame()
    
    st.write("Expiry Complementary file columns:", df.columns.tolist())
    st.write(f"Found {len(df)} deals in expiry complementary file")
    
    # Check if required columns exist
    required_columns = ['DealNumber', 'pataActivaAmortizacion', 'pataPasivaAmortizacion', 'hedgeAccounting']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in expiry complementary file: {', '.join(missing_columns)}")
        st.write("Available columns:", df.columns.tolist())
        
        # Try to find similar column names
        for missing_col in missing_columns:
            possible_matches = [col for col in df.columns if missing_col.lower() in col.lower()]
            if possible_matches:
                st.write(f"Possible matches for '{missing_col}': {possible_matches}")
        
        return pd.DataFrame()
    
    # Convert DealNumber to numeric for matching
    df['DealNumber'] = pd.to_numeric(df['DealNumber'], errors='coerce')
    
    # Convert amortization columns to numeric
    df['pataActivaAmortizacion'] = pd.to_numeric(df['pataActivaAmortizacion'], errors='coerce').fillna(0)
    df['pataPasivaAmortizacion'] = pd.to_numeric(df['pataPasivaAmortizacion'], errors='coerce').fillna(0)
    
    # Convert hedgeAccounting and map YES/NO to Sí/No
    df['hedgeAccounting'] = df['hedgeAccounting'].astype(str).str.strip().str.upper()
    hedge_mapping = {
        'YES': 'Sí',
        'NO': 'No',
        'Y': 'Sí',
        'N': 'No',
        '1': 'Sí',
        '0': 'No'
    }
    df['hedgeAccounting_mapped'] = df['hedgeAccounting'].map(hedge_mapping).fillna('No')
    
    # Display summary
    st.write("Expiry Complementary file summary:")
    st.write(f"- Total deals: {len(df)}")
    st.write(f"- Deals with Activa amortization > 0: {len(df[df['pataActivaAmortizacion'] > 0])}")
    st.write(f"- Deals with Pasiva amortization > 0: {len(df[df['pataPasivaAmortizacion'] > 0])}")
    
    # Show hedge accounting distribution
    hedge_counts = df['hedgeAccounting_mapped'].value_counts()
    st.write("Hedge Accounting distribution:", hedge_counts.to_dict())
    
    # Show preview
    st.subheader("Expiry Complementary Preview")
    preview_cols = ['DealNumber', 'pataActivaAmortizacion', 'pataPasivaAmortizacion', 
                   'hedgeAccounting', 'hedgeAccounting_mapped']
    st.dataframe(df[preview_cols].head(10), use_container_width=True)
    
    return df

def enrich_expiries_with_complementary_data(expiries_df, complementary_df):
    """
    Enrich expiries dataframe with data from expiry complementary file.
    
    Merges on Número Operación (expiries) = DealNumber (complementary)
    """
    if expiries_df.empty or complementary_df.empty:
        st.warning("Cannot enrich expiries - one of the dataframes is empty")
        return expiries_df
    
    st.write("🔗 Enriching expiries file with complementary data...")
    
    # Ensure both key columns are numeric for proper matching
    expiries_df['Número Operación'] = pd.to_numeric(expiries_df['Número Operación'], errors='coerce')
    complementary_df['DealNumber'] = pd.to_numeric(complementary_df['DealNumber'], errors='coerce')
    
    # Create the mapping dataframe with only needed columns
    complementary_mapping = complementary_df[['DealNumber', 'pataActivaAmortizacion', 'pataPasivaAmortizacion', 'hedgeAccounting_mapped']].copy()
    complementary_mapping = complementary_mapping.rename(columns={
        'DealNumber': 'Número Operación',
        'pataActivaAmortizacion': 'Amortización Activa',
        'pataPasivaAmortizacion': 'Amortización Pasiva', 
        'hedgeAccounting_mapped': 'Cobertura'
    })
    
    # Remove duplicates based on deal number (keep first occurrence)
    complementary_mapping = complementary_mapping.drop_duplicates(subset=['Número Operación'], keep='first')
    
    # Perform left merge to keep all expiries records
    enriched_df = expiries_df.merge(complementary_mapping, on='Número Operación', how='left')
    
    # Fill NaN values for unmatched records
    enriched_df['Amortización Activa'] = enriched_df['Amortización Activa'].fillna(0)
    enriched_df['Amortización Pasiva'] = enriched_df['Amortización Pasiva'].fillna(0)  
    enriched_df['Cobertura'] = enriched_df['Cobertura'].fillna('No')
    
    # Show enrichment statistics
    total_expiries = len(expiries_df)
    matched_expiries = len(enriched_df[enriched_df['Cobertura'].notna() & (enriched_df['Cobertura'] != 'No')])
    unmatched_expiries = total_expiries - len(enriched_df.merge(complementary_mapping, on='Número Operación', how='inner'))
    
    st.write("📊 Enrichment Results:")
    st.write(f"- Total expiries: {total_expiries}")
    st.write(f"- Matched with complementary data: {total_expiries - unmatched_expiries}")
    st.write(f"- Unmatched (will use defaults): {unmatched_expiries}")
    
    # Show cobertura distribution in enriched data
    cobertura_counts = enriched_df['Cobertura'].value_counts()
    st.write("Final Cobertura distribution:", cobertura_counts.to_dict())
    
    # Show amortization summary
    activa_nonzero = len(enriched_df[enriched_df['Amortización Activa'] > 0])
    pasiva_nonzero = len(enriched_df[enriched_df['Amortización Pasiva'] > 0])
    st.write(f"Trades with Amortización Activa > 0: {activa_nonzero}")
    st.write(f"Trades with Amortización Pasiva > 0: {pasiva_nonzero}")
    
    return enriched_df

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
    df_preview = df_preview.iloc[:, :16]  # Updated to show more columns for Incumplimiento
    
    # Convert to string before displaying
    df_preview = df_preview.astype(str)
    
    st.write("Rules file preview:")
    st.dataframe(df_preview)
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')

    # Remove the columns after number 16 as these aren't relevant (updated for new columns)
    df = df.iloc[:, :16]
        
    # Log the number of rows after removing empty rows
    st.write(f"Rules file has {len(df)} rows after removing empty rows")
    st.write("Rules file columns:", df.columns.tolist())
    
    # Identify key columns - Updated to include new Incumplimiento columns
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
        elif 'pata' in col_str:
            key_columns['pata'] = col
        elif 'tipo' in col_str and 'contraparte' in col_str:  # New for Incumplimiento
            key_columns['counterparty_type'] = col
        elif 'moneda' in col_str and 'flujo' in col_str:  # New for Incumplimiento
            key_columns['flow_currency'] = col
    
    st.write("Identified rule columns:", key_columns)
    
    # Rename columns to standardized names
    if key_columns:
        df = df.rename(columns={v: k for k, v in key_columns.items()})
    
    # Ensure we have the Pata column with proper name
    if 'pata' in df.columns:
        df = df.rename(columns={'pata': 'Pata'})
    elif 'Pata' not in df.columns:
        # If Pata column is missing, show available columns for debugging
        # Note: Pata might not be relevant for all event types (like Incumplimiento)
        st.info("ℹ️ 'Pata' column not found - this is normal for some event types like Incumplimiento")
    
    # Map event codes
    if 'event' in df.columns:
        event_mapping = {
            'INICIO': 'Curse',
            'VALORIZACION': 'Valorización MTM',
            'VENCIMIENTO': 'Vencimiento',
            'TERMINO': 'Termino',
            'REVERSA_VALORIZACION': 'Reversa Valorización MTM',
            'INCUMPLIMIENTO': 'Incumplimiento'  # New event type
        }
        df['event'] = df['event'].map(event_mapping)
    
    # Show event counts as dictionary not DataFrame to avoid Arrow issues
    if 'event' in df.columns:
        event_counts = df['event'].value_counts(dropna=False)
        st.write("Event counts in rules:", event_counts.to_dict())
    
    # Show Pata distribution if column exists
    if 'Pata' in df.columns:
        pata_counts = df['Pata'].value_counts(dropna=False)
        st.write("Pata distribution in rules:", pata_counts.to_dict())
    
    # Show new Incumplimiento-specific distributions
    if 'counterparty_type' in df.columns:
        counterparty_counts = df['counterparty_type'].value_counts(dropna=False)
        st.write("Counterparty type distribution in rules:", counterparty_counts.to_dict())
    
    if 'flow_currency' in df.columns:
        flow_currency_counts = df['flow_currency'].value_counts(dropna=False)
        st.write("Flow currency distribution in rules:", flow_currency_counts.to_dict())
    
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
        
        # Final validation - check column existence based on event type
        event_types = df_expanded['event'].unique() if 'event' in df_expanded.columns else []
        
        # Check if Pata column exists for events that need it
        pata_required_events = ['Curse', 'Termino']
        needs_pata = any(event in pata_required_events for event in event_types)
        
        if needs_pata and 'Pata' not in df_expanded.columns:
            st.error("❌ Pata column missing for events that require it")
            st.write("Events requiring Pata:", [e for e in event_types if e in pata_required_events])
            st.write("Columns in expanded dataframe:", df_expanded.columns.tolist())
            return pd.DataFrame()
        
        # Check if new Incumplimiento columns exist for Incumplimiento events
        if 'Incumplimiento' in event_types:
            missing_incump_cols = []
            if 'counterparty_type' not in df_expanded.columns:
                missing_incump_cols.append('Tipo Contraparte')
            if 'flow_currency' not in df_expanded.columns:
                missing_incump_cols.append('Moneda Flujo')
            
            if missing_incump_cols:
                st.error(f"❌ Missing required columns for Incumplimiento events: {', '.join(missing_incump_cols)}")
                st.write("Available columns:", df_expanded.columns.tolist())
                return pd.DataFrame()
            else:
                st.success("✅ Found all required columns for Incumplimiento validation")
        
        return df_expanded
    
    return df

def parse_counterparties_file(file):
    """Parse the counterparties Excel file with Instituciones Financieras."""
    try:
        # Get file name for logging
        file_name = getattr(file, 'name', 'unknown')
        st.write(f"Attempting to parse Counterparties file: {file_name}")
        
        # Try different engines
        engines_to_try = ['openpyxl', 'xlrd']
        df_preview = None
        df = None
        
        for engine in engines_to_try:
            try:
                st.write(f"Trying to read Counterparties file with {engine} engine...")
                
                # Read the Excel file with headers on first row (index 0)
                df_preview = pd.read_excel(file, header=0, nrows=15, engine=engine)
                
                # Reset file pointer and read full data
                file.seek(0)
                df = pd.read_excel(file, header=0, engine=engine)
                
                st.success(f"✅ Successfully parsed Counterparties file using {engine} engine")
                break
                
            except Exception as e:
                st.warning(f"Failed with {engine} engine: {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
                continue
        
        # If all engines failed
        if df is None or df_preview is None:
            st.error("❌ Could not parse the Counterparties file with any Excel engine")
            return pd.DataFrame()
        
        # Convert to string for display
        df_preview_display = df_preview.astype(str)
        st.write("Counterparties file preview (first 15 rows):")
        st.dataframe(df_preview_display)
        
        # Show columns found
        st.write("Counterparties file columns:", df.columns.tolist())
        
        # Check for required columns
        required_columns = ['Rut', 'Nombre', 'Tipo']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in counterparties file: {', '.join(missing_columns)}")
            
            # Show available columns for debugging
            st.write("Available columns:", df.columns.tolist())
            
            # Try to find similar column names
            for missing_col in missing_columns:
                possible_matches = [col for col in df.columns if missing_col.lower() in col.lower()]
                if possible_matches:
                    st.write(f"Possible matches for '{missing_col}': {possible_matches}")
            
            return pd.DataFrame()
        
        # Clean and standardize column names for easier processing
        column_mapping = {
            'Rut': 'rut',
            'Nombre': 'nombre',
            'Tipo': 'tipo'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure string formatting for text fields
        df['nombre'] = df['nombre'].astype(str)
        df['tipo'] = df['tipo'].astype(str)
        
        # Filter to only keep Instituciones Financieras (as mentioned in requirements)
        df = df[df['tipo'].str.contains('Instituciones Financieras', case=False, na=False)].copy()
        
        # Remove any empty RUTs
        df = df[df['rut'].str.strip() != ''].copy()
        
        # Display summary information
        st.write(f"Found {len(df)} Instituciones Financieras")
        
        # Show tipo distribution (should all be Instituciones Financieras)
        tipo_counts = df['tipo'].value_counts()
        st.write("Tipo distribution:", tipo_counts.to_dict())
        
        # Display preview of the parsed data
        st.subheader("Counterparties Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Create a set of RUTs for efficient lookup
        rut_set = set(df['rut'].str.strip().str.upper())
        st.write(f"Created RUT lookup set with {len(rut_set)} unique RUTs")
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing counterparties file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()
    
def parse_cartera_file(file):
    """Parse the Cartera Analytics CSV file to extract deal_number and estrategia (hedge_accounting) data."""
    try:
        # Get file name for logging
        file_name = getattr(file, 'name', 'unknown')
        st.write(f"Attempting to parse Cartera file: {file_name}")
        
        # Try CSV parsing with different delimiters and encodings
        df = None
        parsing_method = None
        
        # First, try to detect delimiter by reading a sample
        file.seek(0)
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
                st.success(f"✅ Successfully parsed Cartera file with {encoding} encoding")
                break
            except Exception as e:
                st.warning(f"Failed with {encoding} encoding: {str(e)}")
                continue
        
        # If CSV parsing failed
        if df is None:
            st.error("❌ Could not parse the Cartera file with any method")
            st.error("Please ensure the file is a valid CSV file")
            return pd.DataFrame()
        
        st.write(f"✅ File successfully parsed using: {parsing_method}")
        st.write("Cartera file columns:", df.columns.tolist())
        
        # Check for required columns
        required_columns = ['deal_number', 'hedge_accounting']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in Cartera file: {', '.join(missing_columns)}")
            
            # Show available columns for debugging
            st.write("Available columns:", df.columns.tolist())
            
            # Try to find similar column names
            for missing_col in missing_columns:
                possible_matches = [col for col in df.columns if missing_col.lower() in col.lower()]
                if possible_matches:
                    st.write(f"Possible matches for '{missing_col}': {possible_matches}")
            
            return pd.DataFrame()
        
        # Convert deal_number to numeric
        df['deal_number'] = pd.to_numeric(df['deal_number'], errors='coerce')
        
        # Rename hedge_accounting to estrategia for consistency with business terminology
        df = df.rename(columns={'hedge_accounting': 'estrategia'})
        
        # Ensure string formatting for estrategia
        df['estrategia'] = df['estrategia'].astype(str).str.strip()
        
        # Remove rows with invalid deal numbers
        initial_count = len(df)
        df = df.dropna(subset=['deal_number']).copy()
        df['deal_number'] = df['deal_number'].astype(int)
        final_count = len(df)
        
        if initial_count != final_count:
            st.write(f"Removed {initial_count - final_count} rows with invalid deal numbers")
        
        # Display summary information
        st.write(f"Found {len(df)} deals in Cartera file")
        
        # Show estrategia distribution
        estrategia_counts = df['estrategia'].value_counts()
        st.write("Estrategia distribution:", estrategia_counts.to_dict())
        
        # Show some additional useful info
        if 'product' in df.columns:
            product_counts = df['product'].value_counts()
            st.write("Product distribution:", product_counts.to_dict())
        
        # Display preview of the parsed data (showing key columns)
        st.subheader("Cartera Preview")
        preview_columns = ['deal_number', 'estrategia']
        if 'product' in df.columns:
            preview_columns.append('product')
        if 'counterparty_name' in df.columns:
            preview_columns.append('counterparty_name')
        
        st.dataframe(df[preview_columns].head(10), use_container_width=True)
        
        # Create a lookup dictionary for efficient access during validation
        estrategia_lookup = dict(zip(df['deal_number'], df['estrategia']))
        st.write(f"Created estrategia lookup for {len(estrategia_lookup)} deals")
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing Cartera file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def detect_instrument_type_from_cartera_treasury(df, deal_number):
    """
    Detect instrument type from Cartera Treasury file.
    
    Args:
        df: Raw Cartera Treasury dataframe
        deal_number: Deal number to look up
        
    Returns:
        str: Instrument type ('Swap Tasa', 'Swap Moneda', 'Swap Cámara') or None
    """
    # Find the actual deal number column name (handles newlines in column names)
    deal_col = None
    for col in df.columns:
        if 'Número' in str(col) and 'Operación' in str(col):
            deal_col = col
            break
    
    if not deal_col:
        return None
    
    # Find the actual product column name (handles newlines in column names)
    product_col = None
    for col in df.columns:
        if 'Producto' in str(col):
            product_col = col
            break
    
    if not product_col:
        return None
    
    # Find the row for this deal
    deal_row = df[df[deal_col] == deal_number]
    
    if len(deal_row) == 0:
        return None
    
    # Get the product value
    producto = deal_row[product_col].iloc[0]
    
    # Map the product values to our standard instrument types
    product_mapping = {
        'Swap Tasa': 'Swap Tasa',
        'Swap Moneda': 'Swap Moneda',
        'Swap Promedio Cámara': 'Swap Cámara'
    }
    
    return product_mapping.get(producto, None)