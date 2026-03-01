import sqlite3

def init_db():
    conn = sqlite3.connect("ai_quality.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        groundedness REAL
    )
    """)

    conn.commit()
    conn.close()

def insert_result(question, answer, groundedness):
    conn = sqlite3.connect("ai_quality.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO results (question, answer, groundedness)
    VALUES (?, ?, ?)
    """, (question, answer, groundedness))

    conn.commit()
    conn.close()