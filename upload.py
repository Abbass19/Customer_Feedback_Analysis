import psycopg2
import sqlparse

DATABASE_URL = 'postgresql://neondb_owner:npg_UjonENdPIh98@ep-cool-bush-adwrpqnm-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
schema_file = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Customer_Feedback_Analysis\schema.sql"

# Install sqlparse if not installed: pip install sqlparse

# Connect to Neon
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Read and parse the schema file
with open(schema_file, "r", encoding="utf-8") as f:
    sql_commands = f.read()

# Use sqlparse to split into individual statements
statements = sqlparse.split(sql_commands)

for statement in statements:
    statement = statement.strip()
    if statement:
        cur.execute(statement)

conn.commit()
cur.close()
conn.close()

print("Schema uploaded to Neon successfully!")
