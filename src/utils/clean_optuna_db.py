# src/utils/clean_optuna_db.py
"""
This script cleans up Optuna study database files by removing trials that were in
'FAIL' or 'RUNNING' states. This is a crucial step when a hyperparameter
tuning process, such as the one performed by `smbft_hyperparameter_tuning.py`,
is interrupted and needs to be resumed.

By deleting these incomplete or failed trials, the script ensures that when
the `smbft_hyperparameter_tuning.py` script is re-executed, it will re-process
these specific trials. This allows the hyperparameter tuning to continue
seamlessly from the point of interruption, rather than skipping the trials
that were previously recorded as incomplete. This process is essential for
ensuring the tuning is both comprehensive and efficient.

Usage:
    python clean_optuna_db.py

The script automatically locates Optuna database files within the
'reports/optuna_db/' directory of the project. It iterates through a
predefined list of dataset names and suffixes to identify and process
the relevant database files.
"""
import sqlite3
import os
from pathlib import Path
# Set project root and import custom dataset utilities
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"Project root: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.data_loader import DATASET_NAME_LIST


# Define the base directory.
base_dir = PROJECT_ROOT /'reports/optuna_db/'

# Define the list of suffixes.
suffixes = ['coarse', 'fine']

# Define the SQL query to delete the records.
delete_query = """
DELETE FROM "main"."trials"
WHERE "state" IN ('FAIL', 'RUNNING');
"""
# Iterate through each dataset name.
for dataset_name in DATASET_NAME_LIST:
    # Iterate through each suffix.
    for suffix in suffixes:
        # Construct the full file path.
        db_file_name = f'study_{dataset_name}_smbft_{suffix}.db'
        db_file_path = os.path.join(base_dir, db_file_name)

        # Check if the database file exists.
        if not os.path.isfile(db_file_path):
            print(f"\n❌ Error: The database file '{db_file_path}' does not exist. Skipping...")
            continue  # Move to the next suffix

        print(f"\n⚙️ Processing database: {db_file_path}")

        try:
            # Connect to the database with a timeout.
            with sqlite3.connect(db_file_path, timeout=20) as conn:
                cursor = conn.cursor()

                # Execute the SQL delete statement.
                cursor.execute(delete_query)

                # Commit the changes to the database.
                conn.commit()
                if cursor.rowcount > 0:
                    # Print a confirmation message.
                    print(f"✅  Successfully deleted {cursor.rowcount} rows with 'FAIL' or 'RUNNING' state from the trials table.\n")
                else:
                    print(f"ℹ️ No rows with 'FAIL' or 'RUNNING' state found in the trials table.")
        except sqlite3.Error as e:
            # Print an error message if something goes wrong.
            print(f"\n❌ An error occurred while processing '{db_file_path}': {e}")