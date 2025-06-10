# Accounting Interface Validator for Non-Hedge Swaps

A Streamlit application for validating accounting interface entries against various input files including day trades, MTM valuations, and expiry data.

## Project Structure

The project has been modularized for better organization and maintainability:

```
├── main.py                 # Main Streamlit application
├── file_parsers.py        # File parsing functions
├── validators.py          # Validation logic
├── utils.py              # Utility functions and constants
├── README.md             # This file
└── accounting_validator.py # Original monolithic file (can be removed)
```

## Modules

### `main.py`
The main Streamlit application that provides the user interface and orchestrates the file parsing and validation processes.

### `file_parsers.py`
Contains functions for parsing different file types:
- `parse_interface_file()` - Parses accounting interface Excel files
- `parse_mtm_file()` - Parses MTM Excel/CSV files
- `parse_day_trades_file()` - Parses day trades CSV files
- `parse_expiries_file()` - Parses expiry Excel files
- `parse_rules_file()` - Parses accounting rules Excel files

### `validators.py`
Contains validation logic:
- `validate_day_trades()` - Validates day trades against accounting interface
- `validate_mtm_entries()` - Validates MTM entries (normal and reversal)

### `utils.py`
Utility functions and constants:
- Product mappings
- Event mappings
- Helper functions for data normalization and formatting

## How to Run

1. Install required dependencies:
```bash
pip install streamlit pandas numpy openpyxl
```

2. Run the application:
```bash
streamlit run main.py
```

## File Requirements

### Required Files
- **Accounting Interface File** (.xls/.xlsx) - Contains the accounting entries to validate
- **MTM File (T)** (.xlsx/.csv) - Current day's MTM valuations
- **Rules File** (.xlsx/.csv) - Accounting rules for validation

### Optional Files
- **Day Trades File** (.csv) - New trades entered today
- **MTM File (T-1)** (.xlsx/.csv) - Previous day's MTM file for reversal validation
- **Expiries File** (.xls/.xlsx) - Trades with coupons expiring today

## Validation Types

### 1. Day Trades Validation
Validates that new trades in the day trades file have corresponding "Curse" entries in the accounting interface with correct:
- Account numbers based on instrument type
- Transaction amounts
- Entry types (debit/credit)

### 2. MTM Valorization Validation
Validates MTM entries against expected entries from rules by matching:
- Trade numbers
- Account numbers
- MTM values
- Entry directions (positive/negative)

### 3. MTM Reversal Validation
Validates reversal entries for previous day's MTM positions using T-1 MTM file.

## Debug Features

- Debug specific deal numbers by entering them in the sidebar
- Detailed logging and formatting for easier troubleshooting
- Step-by-step validation results with clear success/failure indicators

## Features

- **Modular Design**: Easy to maintain and extend
- **Comprehensive Validation**: Multiple validation types for different scenarios
- **User-Friendly Interface**: Clear file upload areas and validation options
- **Detailed Reporting**: Match statistics and downloadable results
- **Debug Support**: Specific deal debugging and detailed logging
- **Error Handling**: Robust error handling with clear error messages

## File Format Requirements

### Interface File
- Headers on row 11 (index 10)
- Must contain columns for: Glosa, Account, Debit, Credit, Trade Number
- Last two rows are automatically removed (assumed to be totals)

### MTM Files
- Must contain columns for: Product, Deal Number, MTM Value (M2M_CLP)
- Product codes: SWAP_TASA, SWAP_MONE, SWAP_ICP

### Day Trades File
- CSV format with columns: Número Operación, Producto, Subproducto, Moneda Activa, Monto Activo
- Encoding: latin1

### Rules File
- Must contain columns for: Subproducto, Dirección, Evento, Cuenta Debe, Cuenta Haber
- Events mapped: INICIO → Curse, VALORIZACION → Valorización MTM, etc.

## Output

The application provides:
- Match statistics and percentages
- Detailed validation results tables
- Status breakdowns
- Downloadable CSV results for each validation type
- Color-coded success/failure indicators in debug mode 