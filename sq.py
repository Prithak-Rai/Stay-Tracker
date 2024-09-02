import sqlite3

# Connect to the database
conn = sqlite3.connect("face_data.db")
cursor = conn.cursor()

# Retrieve all records
cursor.execute("SELECT * FROM face_data")
rows = cursor.fetchall()

# Print the records
for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Timestamp: {row[2]}")

# Close the connection
conn.close()

# Uncomment the following code to drop the table if needed
# conn = sqlite3.connect("face_data.db")
# cursor = conn.cursor()
# cursor.execute("DROP TABLE IF EXISTS face_data")
# conn.commit()
# conn.close()
# print("The face_data table has been dropped from the database.")
