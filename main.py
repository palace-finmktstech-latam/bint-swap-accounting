import streamlit as st
import pandas as pd
import numpy as np

# Import our custom modules
from file_parsers import (
    parse_interface_file, 
    parse_cartera_treasury_file,  # NEW: Replaces parse_mtm_file
    parse_day_trades_file, 
    parse_expiries_file, 
    parse_rules_file,
    parse_expiry_complementary_file,
    enrich_expiries_with_complementary_data,
    parse_incumplimientos_file,
    parse_counterparties_file,
    extract_estrategia_from_cartera_treasury  # NEW: For VENCIMIENTO validation
)
from validators import (
    validate_day_trades, 
    validate_mtm_entries_new_format,  # NEW: Replaces validate_mtm_entries
    validate_vencimiento_entries, 
    validate_incumplimiento_entries
)

# Streamlit page configuration
st.set_page_config(
    page_title="Swap Accounting Interface Validator",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Swap Accounting Interface Validator"
    }
)

# Streamlit UI
st.title("Swap Accounting Interface Validator")

with st.sidebar:
    st.header("Upload Files")
    
    # Core required files
    st.subheader("Required Files")
    interface_file = st.file_uploader("Upload Accounting Interface File", type=["xls", "xlsx"])
    rules_file = st.file_uploader("Upload Accounting Rules", type=["xlsx", "csv"])
    
    # Optional files for specific validations
    st.subheader("Optional Files")
    day_trades_file = st.file_uploader("Upload Day Trades File", type=["csv"], 
                                    help="Required for Day Trades validation")
    cartera_treasury_t0_file = st.file_uploader("Upload Cartera Treasury (T0)", type=["xlsx", "xls"],
                               help="Required for MTM Valorization validation - provides both MTM values and estrategia data")
    cartera_treasury_t1_file = st.file_uploader("Upload Cartera Treasury (T-1)", type=["xlsx", "xls"], 
                                 help="Required for MTM Reversal validation - provides both MTM values and estrategia data")
    expiries_file = st.file_uploader("Upload Expiries File", type=["xls", "xlsx"], 
                                 help="Required for VENCIMIENTO validation")                             
    expiry_complementary_file = st.file_uploader("Upload Expiry Complementary Data", type=["xlsx", "xls"],
                                   help="Provides amortization and hedge accounting data for expiries")
    incumplimientos_file = st.file_uploader("Upload Incumplimientos File", type=["xls", "xlsx"],
                                          help="Required for INCUMPLIMIENTO validation")
    counterparties_file = st.file_uploader("Upload Counterparties File", type=["xlsx", "xls", "csv"],
                                         help="Contains RUTs of Instituciones Financieras for INCUMPLIMIENTO validation")
    
    # Debug options
    st.subheader("Debug Options")
    debug_deal = st.text_input("Debug specific deal number (optional)", "")
    debug_deal = debug_deal if debug_deal else None

# Show file requirements info
st.sidebar.markdown("---")
st.sidebar.subheader("File Requirements")
st.sidebar.markdown("""
**Day Trades Validation**: Interface + Rules + Day Trades files  
**MTM Valorization**: Interface + Rules + Cartera Treasury (T0) files  
**MTM Reversal**: Interface + Rules + Cartera Treasury (T-1) files  
**VENCIMIENTO Validation**: Interface + Rules + Expiries + Cartera Treasury (T0) files  
**Enhanced VENCIMIENTO**: Also upload Expiry Complementary Data for complete data  
**INCUMPLIMIENTO Validation**: Interface + Rules + Incumplimientos + Counterparties files

*Note: Cartera Treasury files now provide both MTM values AND estrategia data - no separate Cartera Analytics file needed!*
*Note: Enhanced support for new MX estrategia values (COB_MX_ACTIVOS/COB_MX_PASIVOS)*
""")

# Main area - only show if core files are uploaded
if interface_file and rules_file:
    st.header("File Analysis")
    
    # Parse core files
    with st.expander("Interface File", expanded=False):
        interface_df, interface_cols = parse_interface_file(interface_file)
        
    with st.expander("Rules File", expanded=False):
        rules_df = parse_rules_file(rules_file)
    
    # Parse new Cartera Treasury files
    cartera_treasury_t0_df = None
    cartera_treasury_t0_processed = None
    if cartera_treasury_t0_file:
        with st.expander("Cartera Treasury (T0)", expanded=False):
            cartera_treasury_t0_df, cartera_treasury_t0_processed = parse_cartera_treasury_file(cartera_treasury_t0_file, "T0")
    
    cartera_treasury_t1_df = None
    cartera_treasury_t1_processed = None
    if cartera_treasury_t1_file:
        with st.expander("Cartera Treasury (T-1)", expanded=False):
            cartera_treasury_t1_df, cartera_treasury_t1_processed = parse_cartera_treasury_file(cartera_treasury_t1_file, "T-1")
    
    # Parse other files (unchanged)
    day_trades_df = None
    if day_trades_file:
        with st.expander("Day Trades File", expanded=False):
            day_trades_df = parse_day_trades_file(day_trades_file)
            
    # Parse complementary data first
    expiry_complementary_df = None
    if expiry_complementary_file:
        with st.expander("Expiry Complementary Data", expanded=False):
            expiry_complementary_df = parse_expiry_complementary_file(expiry_complementary_file)

    # Parse and enrich expiries
    expiries_df = None
    if expiries_file:
        with st.expander("Expiries File", expanded=False):
            expiries_df_raw = parse_expiries_file(expiries_file)
            
            # Enrich with complementary data if available
            if expiry_complementary_df is not None and not expiry_complementary_df.empty:
                expiries_df = enrich_expiries_with_complementary_data(expiries_df_raw, expiry_complementary_df)
                st.success("✅ Expiries file enriched with complementary data")
            else:
                expiries_df = expiries_df_raw
                st.info("ℹ️ Using expiries file without complementary data enrichment")
                
                # Add default columns if they don't exist
                if 'Amortización Activa' not in expiries_df.columns:
                    expiries_df['Amortización Activa'] = 0
                if 'Amortización Pasiva' not in expiries_df.columns:
                    expiries_df['Amortización Pasiva'] = 0
                if 'Cobertura' not in expiries_df.columns:
                    expiries_df['Cobertura'] = 'No'

    # Parse incumplimientos file
    incumplimientos_df = None
    if incumplimientos_file:
        with st.expander("Incumplimientos File", expanded=False):
            incumplimientos_df = parse_incumplimientos_file(incumplimientos_file)

    # Parse counterparties file
    counterparties_df = None
    if counterparties_file:
        with st.expander("Counterparties File", expanded=False):
            counterparties_df = parse_counterparties_file(counterparties_file)

    # MOVED: Validation options - only show available validations AFTER all parsing is complete
    st.subheader("Available Validations")
    
    # Check what validations are possible - Updated for integrated Cartera Treasury approach
    can_validate_day_trades = day_trades_df is not None and not day_trades_df.empty
    can_validate_mtm = (cartera_treasury_t0_processed is not None and 
                       not cartera_treasury_t0_processed.empty)  # No separate cartera_df needed
    can_validate_reversal = (cartera_treasury_t1_processed is not None and 
                            not cartera_treasury_t1_processed.empty)  # No separate cartera_df needed
    can_validate_vencimiento = (expiries_file is not None and 
                              cartera_treasury_t0_df is not None and not cartera_treasury_t0_df.empty)  # Uses T0 for estrategia data
    can_validate_incumplimiento = incumplimientos_df is not None and not incumplimientos_df.empty
    
    if not any([can_validate_day_trades, can_validate_mtm, can_validate_reversal, can_validate_vencimiento, can_validate_incumplimiento]):
        st.warning("⚠️ No optional files uploaded. Please upload at least one optional file to enable validations.")
        st.info("📁 Upload Day Trades file for trade validation, Cartera Treasury (T0) for MTM valorization validation, Cartera Treasury (T-1) for MTM reversal validation, Expiries file for VENCIMIENTO validation, or Incumplimientos file for INCUMPLIMIENTO validation.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if can_validate_day_trades:
                run_day_trades_validation = st.checkbox("✅ Run Day Trades Validation", value=True)
            else:
                st.checkbox("❌ Run Day Trades Validation", value=False, disabled=True)
                st.caption("Requires Day Trades file")
                run_day_trades_validation = False
                
            if can_validate_mtm:
                run_mtm_validation = st.checkbox("✅ Run MTM Valorization Validation", value=True)
            else:
                st.checkbox("❌ Run MTM Valorization Validation", value=False, disabled=True)
                st.caption("Requires Cartera Treasury (T0) file")
                run_mtm_validation = False
            
            if can_validate_incumplimiento:
                run_incumplimiento_validation = st.checkbox("✅ Run INCUMPLIMIENTO Validation", value=True)
            else:
                st.checkbox("❌ Run INCUMPLIMIENTO Validation", value=False, disabled=True)
                st.caption("Requires Incumplimientos file")
                run_incumplimiento_validation = False
        
        with col2:
            if can_validate_reversal:
                run_reversal_validation = st.checkbox("✅ Run MTM Reversal Validation", value=True)
            else:
                st.checkbox("❌ Run MTM Reversal Validation", value=False, disabled=True)
                st.caption("Requires Cartera Treasury (T-1) file")
                run_reversal_validation = False
                
            if can_validate_vencimiento:
                run_vencimiento_validation = st.checkbox("✅ Run VENCIMIENTO Validation", value=True)
            else:
                st.checkbox("❌ Run VENCIMIENTO Validation", value=False, disabled=True)
                st.caption("Requires Expiries and Cartera Treasury (T0) files")
                run_vencimiento_validation = False
        
        # Run validations button - only show if at least one validation is selected
        available_validations = sum([run_day_trades_validation, run_mtm_validation, run_reversal_validation, run_vencimiento_validation, run_incumplimiento_validation])
        
        if available_validations > 0:
            if st.button(f"🚀 Run {available_validations} Validation(s)"):
                with st.spinner("Running validation..."):
                    # Run day trades validation if selected
                    if run_day_trades_validation:
                        st.header("📊 Day Trades Validation")
                        day_trades_results = validate_day_trades(
                            day_trades_df,
                            interface_df,
                            interface_cols,
                            rules_df,
                            debug_deal=debug_deal
                        )
                    
                    # Check if both MTM validations are selected (combined mode)
                    both_mtm_selected = run_mtm_validation and run_reversal_validation
                    
                    # Run MTM validation if selected - UPDATED to use integrated approach
                    if run_mtm_validation:
                        if both_mtm_selected:
                            st.header("💰 MTM Valorization Validation (Combined Mode)")
                            st.info("ℹ️ Combined mode: Accounting for both current MTM and reversal entries in same interface file")
                        else:
                            st.header("💰 MTM Valorization Validation")
                            
                        mtm_results = validate_mtm_entries_new_format(
                            interface_df, 
                            interface_cols, 
                            cartera_treasury_t0_df,  # Raw data for instrument type detection
                            cartera_treasury_t0_processed,  # Processed data for validation
                            rules_df,
                            # REMOVED: cartera_df parameter - now extracted from Cartera Treasury
                            event_type='Valorización MTM',
                            debug_deal=debug_deal,
                            combined_mtm_mode=both_mtm_selected
                        )
                    
                    # Run reversal validation if selected - UPDATED to use integrated approach
                    if run_reversal_validation:
                        if both_mtm_selected:
                            st.header("🔄 MTM Reversal Validation (Combined Mode)")
                            st.info("ℹ️ Combined mode: Accounting for both current MTM and reversal entries in same interface file")
                        else:
                            st.header("🔄 MTM Reversal Validation")
                            
                        reversal_results = validate_mtm_entries_new_format(
                            interface_df, 
                            interface_cols, 
                            cartera_treasury_t1_df,  # Raw data for instrument type detection
                            cartera_treasury_t1_processed,  # Processed data for validation
                            rules_df,
                            # REMOVED: cartera_df parameter - now extracted from Cartera Treasury
                            event_type='Valorización MTM',      # For interface file filtering
                            rules_event_type='Reversa Valorización MTM',  # For rules file filtering
                            key_suffix='-reversal',
                            debug_deal=debug_deal,
                            combined_mtm_mode=both_mtm_selected
                        )
                    
                    # Run VENCIMIENTO validation if selected - UPDATED to use integrated approach
                    if run_vencimiento_validation:
                        st.header("📅 VENCIMIENTO Validation")
                        
                        # Extract estrategia data from Cartera Treasury T0 for VENCIMIENTO validation
                        vencimiento_cartera_df = extract_estrategia_from_cartera_treasury(cartera_treasury_t0_df)
                        
                        vencimiento_results = validate_vencimiento_entries(
                            expiries_df,
                            interface_df,
                            interface_cols,
                            rules_df,
                            vencimiento_cartera_df,  # Now uses estrategia data extracted from Cartera Treasury
                            debug_deal=debug_deal
                        )
                    
                    # Run INCUMPLIMIENTO validation if selected (unchanged)
                    if run_incumplimiento_validation:
                        st.header("⚠️ INCUMPLIMIENTO Validation")
                        incumplimiento_results = validate_incumplimiento_entries(
                            incumplimientos_df,
                            interface_df,
                            interface_cols,
                            rules_df,
                            counterparties_df,
                            debug_deal=debug_deal
                        )
        else:
            st.info("Please select at least one validation to run.")

elif interface_file and not rules_file:
    st.warning("⚠️ Please upload the Rules file to continue.")
elif rules_file and not interface_file:
    st.warning("⚠️ Please upload the Accounting Interface file to continue.")
else:
    st.info("📁 Please upload the required files to start validation.")
    st.markdown("""
    ### Required Files:
    - **Accounting Interface File**: Contains the accounting entries to validate
    - **Accounting Rules**: Rules for validation logic
    
    ### Optional Files (choose based on validation needs):
    - **Day Trades File**: For validating new trades entered today
    - **Cartera Treasury (T0)**: For validating current day's MTM valorization (provides both MTM values and estrategia data)
    - **Cartera Treasury (T-1)**: For validating reversal of previous day's MTM (provides both MTM values and estrategia data)
    - **Expiries File**: For validating VENCIMIENTO entries
    - **Expiry Complementary Data**: Provides amortization and hedge accounting data for enhanced VENCIMIENTO validation
    - **Incumplimientos File**: For validating INCUMPLIMIENTO entries
    - **Counterparties File**: Contains RUTs of Instituciones Financieras for INCUMPLIMIENTO validation
    
    ### 🎉 NEW: Simplified File Requirements!
    - **No separate Cartera Analytics file needed** - estrategia data is now extracted directly from Cartera Treasury files
    - **Enhanced estrategia support** - includes new MX values (COB_MX_ACTIVOS/COB_MX_PASIVOS)
    - **Single source of truth** - Cartera Treasury provides both MTM values and estrategia data
    """)