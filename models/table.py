import sqlite3

# Connecting to the SQLite database
connection = sqlite3.connect('/home/azureuser/iSOA/data/output/metadata.db')
cursor = connection.cursor()

# Executing a query to select all records from the objects table
cursor.execute('''SELECT label FROM objects''')

# Fetch all results from the query
rows = cursor.fetchall()

# Print each row
for row in rows:
    print(row)

# Close the connection
connection.close()
