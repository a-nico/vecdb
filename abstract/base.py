from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import sqlite3
import json
import heapq
from utils import BottomK

"""
These abstractions create the vector database for a RAG system.
"""


class Document(ABC):
    def __init__(self, text: str, embedding: Optional[np.ndarray] = None):
        self.text: str = text
        self._embedding: Optional[np.ndarray] = embedding

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: np.ndarray) -> None:
        self._embedding = embedding


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Given a list of documents, return their embeddings as a list of vectors.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_embedding_dimension() -> int:
        """
        Return the embedding dimension produced by this model.
        """
        pass

    @abstractmethod
    def distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute a distance between two embeddings.
        Lower values mean more similar.
        """
        pass

    def __call__(self, documents: List[str]) -> List[np.ndarray]:
        return self.embed_documents(documents)


class VectorDatabase(ABC):
    db_path: str
    conn: sqlite3.Connection
    cursor: sqlite3.Cursor
    embedding_dimension: Optional[int]
    num_hash_hyperplanes: Optional[int]
    hash_vectors: Optional[np.ndarray]
    embedding_model: EmbeddingModel

    @staticmethod
    def generate_hash_vectors(
        num_hash_hyperplanes: int, embedding_dim: int
    ) -> np.ndarray:
        """Generates random hash vectors (hyperplanes) for LSH."""
        import numpy as np

        return (2 * np.random.rand(num_hash_hyperplanes, embedding_dim) - 1).astype(
            np.float32
        )

    @classmethod
    def setup(cls, db_path: str, embedding_dim: int, num_hash_hyperplanes: int) -> None:
        """
        Idempotently sets up the Metadata table and inserts hash vectors if not already present.
        Also creates the Metadata table if it does not exist.
        """

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            # Create Metadata table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    embedding_dimension INTEGER NOT NULL,
                    num_hash_hyperplanes INTEGER NOT NULL,
                    hash_vectors BLOB NOT NULL
                );
                """
            )
            # Check if metadata row exists
            cursor.execute("SELECT 1 FROM Metadata WHERE id = 1;")
            if cursor.fetchone() is None:
                # Generate hash vectors and insert metadata
                hash_vectors = cls.generate_hash_vectors(
                    num_hash_hyperplanes, embedding_dim
                )
                cursor.execute(
                    """
                    INSERT INTO Metadata (id, embedding_dimension, num_hash_hyperplanes, hash_vectors)
                    VALUES (?, ?, ?, ?);
                    """,
                    (1, embedding_dim, num_hash_hyperplanes, hash_vectors.tobytes()),
                )
                conn.commit()
                print(f"Metadata initialized in '{db_path}'.")
            else:
                print(f"Metadata already exists in '{db_path}', skipping setup.")
        except sqlite3.Error as e:
            print(f"Error during metadata setup: {e}")
        finally:
            conn.close()

    def __init__(self, db_path: str):
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # Load metadata from the database
        self.embedding_dimension: Optional[int] = None
        self.num_hash_hyperplanes: Optional[int] = None
        self.hash_vectors: Optional[np.ndarray] = None
        self._load_metadata()

        self.embedding_model: EmbeddingModel = self.get_embedding_model()

    @abstractmethod
    def get_embedding_model(self) -> EmbeddingModel:
        pass

    def _load_metadata(self) -> None:
        """
        Load metadata from the database and store it in class properties.
        This includes embedding dimension, number of hash hyperplanes, and hash vectors.
        """
        try:
            self.cursor.execute(
                "SELECT embedding_dimension, num_hash_hyperplanes, hash_vectors FROM Metadata WHERE id = 1;"
            )
            meta = self.cursor.fetchone()
            if meta:
                self.embedding_dimension = meta[0]
                self.num_hash_hyperplanes = meta[1]
                vectors_blob = meta[2]
                # Convert blob back to numpy array
                self.hash_vectors = np.frombuffer(
                    vectors_blob, dtype=np.float32
                ).reshape(self.num_hash_hyperplanes, self.embedding_dimension)
            else:
                print("Warning: No metadata found in the database.")
        except sqlite3.Error as e:
            print(f"Error loading metadata: {e}")

    def add_documents(self, documents: List[str]) -> None:
        """
        Add a list of documents to the database.

        1) Embed using the embedding model.
        2) Compute its LSH.
        3) Store the text, embedding, and LSH in the database.

        """
        # Step 1: Embed documents
        embeddings = self.embedding_model(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        # Step 2: Compute LSH for all embeddings in one pass
        hash_results = (
            embeddings_array @ self.hash_vectors.T > 0
        )  # shape: (num_docs, num_hash_hyperplanes)
        lsh_values = [
            int("".join(["1" if x else "0" for x in row]), 2) for row in hash_results
        ]

        # Step 3: Store in database
        for text, emb, lsh in zip(documents, embeddings_array, lsh_values):
            emb_bytes = emb.astype(np.float32).tobytes()
            neighbors = []  # start as empty Python list
            self.cursor.execute(
                "INSERT INTO Documents (text, embedding, lsh, neighbors) VALUES (?, ?, ?, ?);",
                (text, emb_bytes, lsh, json.dumps(neighbors)),
            )
        self.conn.commit()

    def get_closest_documents(
        self, text: str, k: int, search_all: bool = False
    ) -> List[tuple[float, str]]:
        """
        Find the top-k closest documents to the input text using the embedding model's distance function.
        Uses BottomK to efficiently track the k smallest distances.
        If search_all is False, also searches neighboring clusters using a greedy approach.
        """
        # Step 1: Embed the input text
        embedding = self.embedding_model([text])
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding[0]

        # Step 2: Compute LSH hash for the input embedding
        hash_result = embedding @ self.hash_vectors.T > 0
        lsh_value = int("".join(["1" if x else "0" for x in hash_result]), 2)

        # Step 3: Generator for all documents with the same LSH hash
        def doc_generator_for_lsh(lsh: int):
            self.cursor.execute(
                "SELECT text, embedding FROM Documents WHERE lsh = ?;",
                (lsh,),
            )
            for row in self.cursor:
                yield row

        # Step 4: Compute distance and keep bottom-k using BottomK (for closest, use min-heap behavior)
        bottom_k_tracker = BottomK[tuple[float, str]](k)

        # Helper to process docs for a given LSH and update bottom_k_tracker
        def process_lsh_docs(lsh: int) -> bool:
            found_closer = False
            for doc_text, emb_blob in doc_generator_for_lsh(lsh):
                doc_emb = np.frombuffer(emb_blob, dtype=np.float32)
                dist = self.embedding_model.distance(embedding, doc_emb)
                docs_searched_nonlocal[0] += 1
                # If BottomK not full or this doc is closer than the farthest in BottomK, add it
                if len(bottom_k_tracker) < k or dist < bottom_k_tracker.peek()[0]:
                    bottom_k_tracker.push((dist, doc_text))
                    found_closer = True
            return found_closer

        docs_searched_nonlocal = [0]  # mutable for inner function

        if not search_all:
            # Always process the main LSH cluster
            process_lsh_docs(lsh_value)

            # Step 5: Greedily search neighbors
            neighbor_lshs = self._get_neighbor_lshs(lsh_value)
            consecutive_failures = 0
            for neighbor_lsh in neighbor_lshs:
                # Check if this neighbor cluster has any docs
                self.cursor.execute(
                    "SELECT 1 FROM Documents WHERE lsh = ? LIMIT 1;", (neighbor_lsh,)
                )
                if self.cursor.fetchone() is None:
                    # No docs in this cluster, skip and do not count as a failure
                    continue
                found_closer = process_lsh_docs(neighbor_lsh)
                if not found_closer:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                else:
                    consecutive_failures = 0
        else:
            # If search_all is True, fetch all documents (do NOT process main LSH cluster above)
            self.cursor.execute("SELECT text, embedding FROM Documents;")
            for doc_text, emb_blob in self.cursor:
                doc_emb = np.frombuffer(emb_blob, dtype=np.float32)
                dist = self.embedding_model.distance(embedding, doc_emb)
                docs_searched_nonlocal[0] += 1
                bottom_k_tracker.push((dist, doc_text))

        print(f"Docs searched: {docs_searched_nonlocal[0]}")
        # Retrieve the sorted list (ascending distance) from BottomK
        return bottom_k_tracker.to_list(sorted_list=True)

    def _get_neighbor_lshs(self, lsh_value: int) -> list[int]:
        """
        Returns a list of neighboring LSH values for the given LSH, sorted by closeness.
        """
        self.cursor.execute(
            "SELECT ClosestNeighbors FROM Neighbors WHERE LSH = ?;", (lsh_value,)
        )
        row = self.cursor.fetchone()
        if row and row[0]:
            try:
                neighbors = json.loads(row[0])
                # Ensure it's a list of ints
                return [int(x) for x in neighbors]
            except Exception:
                return []
        return []

    def discover_and_log_neighbors(self, k: int) -> None:
        """
        For each distinct LSH value, compute the average embedding, then find the k closest other LSHs
        (by distance between average embeddings), and store the result as JSON in the Neighbors table.
        This will overwrite previous neighbors for each LSH.
        """
        # Step 1: Query all distinct LSH values
        self.cursor.execute("SELECT DISTINCT lsh FROM Documents;")
        lsh_values = [row[0] for row in self.cursor.fetchall()]
        if not lsh_values:
            print("No LSH values found in Documents table.")
            return

        # Step 2: For each LSH, get all embeddings and compute average embedding
        lsh_to_avg_emb = {}
        for lsh in lsh_values:
            self.cursor.execute(
                "SELECT embedding FROM Documents WHERE lsh = ?;", (lsh,)
            )
            emb_blobs = [row[0] for row in self.cursor.fetchall()]
            if not emb_blobs:
                continue
            embeddings = np.stack(
                [np.frombuffer(emb, dtype=np.float32) for emb in emb_blobs]
            )
            avg_emb = self.average_direction(embeddings)
            lsh_to_avg_emb[lsh] = avg_emb

        # Step 3: For each LSH, find k closest other LSHs by distance
        for lsh, avg_emb in lsh_to_avg_emb.items():
            bottom_k = BottomK[tuple[float, int]](k)
            for other_lsh, other_emb in lsh_to_avg_emb.items():
                if other_lsh == lsh:
                    continue
                dist = self.embedding_model.distance(avg_emb, other_emb)
                bottom_k.push((dist, other_lsh))
            # Step 4: Sort by distance ascending and extract LSH values
            closest = [
                lsh_val for (_, lsh_val) in sorted(bottom_k.to_list(sorted_list=True))
            ]
            # Step 5: Store in Neighbors table as JSON (overwrite if there already)
            self.cursor.execute(
                "INSERT OR REPLACE INTO Neighbors (LSH, ClosestNeighbors) VALUES (?, ?);",
                (lsh, json.dumps(closest)),
            )
        self.conn.commit()

    def rehash_documents(self, num_hash_hyperplanes: int) -> None:
        """
        Rehash all documents using a new number of hash hyperplanes.
        Updates the Metadata table, rehashes all Documents, and updates neighbors.
        """
        # Step 1: Generate new hash vectors
        new_hash_vectors = VectorDatabase.generate_hash_vectors(
            num_hash_hyperplanes, self.embedding_dimension
        )

        # Step 2: Update Metadata table
        self.cursor.execute(
            """
            UPDATE Metadata
            SET num_hash_hyperplanes = ?, hash_vectors = ?
            WHERE id = 1;
            """,
            (num_hash_hyperplanes, new_hash_vectors.tobytes()),
        )
        self.conn.commit()

        # Step 3: Update instance variables
        self.num_hash_hyperplanes = num_hash_hyperplanes
        self.hash_vectors = new_hash_vectors

        # Step 4: Recompute LSH for all documents
        self.cursor.execute("SELECT rowid, embedding FROM Documents;")
        docs = self.cursor.fetchall()
        for rowid, emb_blob in docs:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            hash_result = emb @ self.hash_vectors.T > 0
            lsh_value = int("".join(["1" if x else "0" for x in hash_result]), 2)
            self.cursor.execute(
                "UPDATE Documents SET lsh = ? WHERE rowid = ?;",
                (lsh_value, rowid),
            )
        self.conn.commit()

        # Step 5: Reload metadata (in case other fields changed)
        self._load_metadata()

        # Step 6: Recompute neighbors
        self.discover_and_log_neighbors(
            k=5
        )  # Default k=5, or make k a parameter if needed

    def average_direction(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute the normalized average direction of the given vectors.
        Raises:
            ValueError: If the sum of vectors is zero (no unique direction).
        """
        total = np.sum(vectors, axis=0)
        norm = np.linalg.norm(total)
        if norm == 0:
            # this should never happen but just return a trivial unit vector
            unit = np.zeros_like(total)
            unit[0] = 1.0
            return unit
        return total / norm

    def close(self) -> None:
        """
        Commit any pending transactions and close the connection.
        """
        try:
            self.conn.commit()
        except Exception:
            pass
        finally:
            self.conn.close()

    def __enter__(self) -> "VectorDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        # make sure db lock is released
        try:
            self.close()
        except Exception:
            pass
