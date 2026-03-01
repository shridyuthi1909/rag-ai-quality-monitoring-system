import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Load database
conn = sqlite3.connect("ai_quality.db")
df = pd.read_sql_query("SELECT * FROM results", conn)

conn.close()

print("\n--- AI QUALITY REPORT ---\n")
print(df)

print("\nAverage Groundedness:", round(df["groundedness"].mean(), 3))
print("Highest Score:", round(df["groundedness"].max(), 3))
print("Lowest Score:", round(df["groundedness"].min(), 3))

# Plot quality distribution
plt.figure()
plt.hist(df["groundedness"], bins=5)
plt.xlabel("Groundedness Score")
plt.ylabel("Frequency")
plt.title("AI Quality Distribution")

plt.show()