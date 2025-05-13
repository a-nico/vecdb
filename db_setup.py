import sqlite3


def setup_database(db_name):
    """Connects to the database and creates tables if they don't exist."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        # Create Documents table (removed Neighbors column)
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS Documents (
            _id INTEGER PRIMARY KEY AUTOINCREMENT,
            Text TEXT NOT NULL,
            LSH INTEGER NOT NULL,
            Embedding BLOB
        );
        """
        )

        # Create Neighbors table. For each LSH, we have closest other LSH, sorted by distance.
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS Neighbors (
            LSH INTEGER PRIMARY KEY,
            ClosestNeighbors TEXT -- stores JSON list of int (LSH) values
        );
        """
        )

        conn.commit()

        print(
            f"Database setup complete for '{db_name}'. Tables 'Documents' and 'Neighbors' are ready."
        )
    except sqlite3.Error as e:
        print(f"<!!!> An error occurred:\n{e}")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup the vector database. Creates necessary tables (Documents, Neighbors)."
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        required=True,
        help="Required: Tag or name for the database file (e.g., 'my_vec_library.db')",
    )
    args = parser.parse_args()
    setup_database(args.tag)
