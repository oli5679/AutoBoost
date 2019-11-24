csv_path = "data/market-invoice-data.csv"

ignore_cols = [
    "Trade ID",
    "Seller ID",
    "Trade Type",
    "Advance Date",
    "Expected Payment Date",
    "Settlement Date",
    "In Arrears",
    "In Arrears on Date",
    "Crystallised Loss Date",
    "Payment State",
    "Crystallised Loss",
]

target_col = "arrears_target"

output_dir_path = "data/market-invoice-1.1"
