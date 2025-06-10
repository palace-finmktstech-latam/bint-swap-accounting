import streamlit as st
import pandas as pd
import numpy as np

# Import our custom modules
from file_parsers import (
    parse_interface_file, 
    parse_mtm_file, 
    parse_day_trades_file, 
    parse_expiries_file, 
    parse_rules_file
)
from validators import validate_day_trades, validate_mtm_entries

# Streamlit page configuration
st.set_page_config(
    page_title="Accounting Interface Validator",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Accounting Interface Validator for Non-Hedge Swaps"
    }
)

# Streamlit UI
st.title("Accounting Interface Validator for Non-Hedge Swaps")

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
    mtm_file = st.file_uploader("Upload MTM File (T)", type=["xlsx", "csv"],
                               help="Required for MTM Valorization validation")
    mtm_t1_file = st.file_uploader("Upload MTM File (T-1)", type=["xlsx", "csv"], 
                                 help="Required for MTM Reversal validation")
    expiries_file = st.file_uploader("Upload Expiries File", type=["xls", "xlsx"], 
                                 help="For future expiries validation")                             
    
    # Debug options
    st.subheader("Debug Options")
    debug_deal = st.text_input("Debug specific deal number (optional)", "")
    debug_deal = debug_deal if debug_deal else None

# Show file requirements info
st.sidebar.markdown("---")
st.sidebar.subheader("File Requirements")
st.sidebar.markdown("""
**Day Trades Validation**: Interface + Rules + Day Trades files  
**MTM Valorization**: Interface + Rules + MTM (T) files  
**MTM Reversal**: Interface + Rules + MTM (T-1) files
""")

# Main area - only show if core files are uploaded
if interface_file and rules_file:
    st.header("File Analysis")
    
    # Parse core files
    with st.expander("Interface File", expanded=False):
        interface_df, interface_cols = parse_interface_file(interface_file)
    
    with st.expander("Rules File", expanded=False):
        rules_df = parse_rules_file(rules_file)
    
    # Parse optional files only if uploaded
    mtm_df = None
    mtm_sums = None
    if mtm_file:
        with st.expander("MTM File (T)", expanded=False):
            mtm_df, mtm_sums = parse_mtm_file(mtm_file)
    
    mtm_t1_df = None
    mtm_t1_sums = None
    if mtm_t1_file:
        with st.expander("MTM File (T-1)", expanded=False):
            mtm_t1_df, mtm_t1_sums = parse_mtm_file(mtm_t1_file)
    
    day_trades_df = None
    if day_trades_file:
        with st.expander("Day Trades File", expanded=False):
            day_trades_df = parse_day_trades_file(day_trades_file)
            
    expiries_df = None
    if expiries_file:
        with st.expander("Expiries File", expanded=False):
            expiries_df = parse_expiries_file(expiries_file)

    # Validation options - only show available validations
    st.subheader("Available Validations")
    
    # Check what validations are possible
    can_validate_day_trades = day_trades_df is not None
    can_validate_mtm = mtm_df is not None
    can_validate_reversal = mtm_t1_df is not None
    
    if not any([can_validate_day_trades, can_validate_mtm, can_validate_reversal]):
        st.warning("⚠️ No optional files uploaded. Please upload at least one optional file to enable validations.")
        st.info("📁 Upload Day Trades file for trade validation, MTM file for valorization validation, or MTM (T-1) file for reversal validation.")
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
                st.caption("Requires MTM (T) file")
                run_mtm_validation = False
        
        with col2:
            if can_validate_reversal:
                run_reversal_validation = st.checkbox("✅ Run MTM Reversal Validation", value=True)
            else:
                st.checkbox("❌ Run MTM Reversal Validation", value=False, disabled=True)
                st.caption("Requires MTM (T-1) file")
                run_reversal_validation = False
        
        # Run validations button - only show if at least one validation is selected
        available_validations = sum([run_day_trades_validation, run_mtm_validation, run_reversal_validation])
        
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
                    
                    # Run MTM validation if selected
                    if run_mtm_validation:
                        st.header("💰 MTM Valorization Validation")
                        mtm_results = validate_mtm_entries(
                            interface_df, 
                            interface_cols, 
                            mtm_df, 
                            mtm_sums, 
                            rules_df, 
                            event_type='Valorización MTM',
                            debug_deal=debug_deal
                        )
                    
                    # Run reversal validation if selected
                    if run_reversal_validation:
                        st.header("🔄 MTM Reversal Validation")
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
    - **MTM File (T)**: For validating current day's MTM valorization
    - **MTM File (T-1)**: For validating reversal of previous day's MTM
    - **Expiries File**: For future expiries validation (not yet implemented)
    """) 