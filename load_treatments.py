import sqlite3
import pandas as pd
from pathlib import Path

# Paths
csv_path = Path("src/db/treatments.csv")
db_path = Path("src/db/cropcare.db")

# Read CSV with error handling
df = pd.read_csv(csv_path, on_bad_lines="skip")  # ✅ skip malformed rows

# Ensure correct columns
df = df[["disease_name", "organic", "chemical", "prevention"]]

# Connect to DB
con = sqlite3.connect(db_path)
cur = con.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS treatments (
    disease_name TEXT PRIMARY KEY,
    organic TEXT,
    chemical TEXT,
    prevention TEXT
)
""")

# Insert rows
for _, row in df.iterrows():
    cur.execute("""
    INSERT OR IGNORE INTO treatments (disease_name, organic, chemical, prevention)
    VALUES (?, ?, ?, ?)
    """, (row["disease_name"], row["organic"], row["chemical"], row["prevention"]))

con.commit()
con.close()

print(f"✅ Loaded {len(df)} treatments into {db_path}")
