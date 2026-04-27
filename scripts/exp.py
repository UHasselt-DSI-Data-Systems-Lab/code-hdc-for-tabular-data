import os
from pathlib import Path
import duckdb

# 1. Get the absolute path to the root directory
# __file__ is project_root/scripts/myfile.py
# .parent.parent moves up two levels to project_root/
ROOT_DIR = Path(__file__).resolve().parent.parent

def run_query():
    # 2. Construct the path to the file
    # This matches: project_root / data / raw / movie.csv
    csv_path = ROOT_DIR / "data" / "raw" / "movie_master.csv"

    # 3. Check and Read
    if not csv_path.exists():
        print(f"❌ Cannot find file at: {csv_path}")
        return

    print(f"✅ Found file! Loading into DuckDB...")
    
    # We use str() because DuckDB prefers a string path over a Path object
    rel = duckdb.read_csv(str(csv_path))
    print(rel.show())

if __name__ == "__main__":
    run_query()