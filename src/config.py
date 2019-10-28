target_col = "target"

target_mapping = {1: 1, 0: 0}

csv_path = "../data/road-safety/raw.csv"

output_dir_path = "../output/road-safety/1.1"

drop_cols = [
    "Accident_Index",
    "Accident_Severity_label",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]

date_cols = ["Date", "Time"]
