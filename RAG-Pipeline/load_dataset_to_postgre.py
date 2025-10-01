import os, json, sys
from typing import Iterable, Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

# -------- Config --------
load_dotenv()

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")

INPUT_PATH = os.getenv("INPUT_PATH", "/Users/ssr/Documents/startup-chatbot-surin/RAG-Pipeline/dataset/for_embedding.json")
TABLE_NAME = os.getenv("TABLE_NAME", "public.documents")

# Embedding target column and expected dimension (if known)
TARGET_EMB_COL = os.getenv("TARGET_EMB_COL", "embedding")           # e.g., embedding_1024
TARGET_EMB_DIM = int(os.getenv("TARGET_EMB_DIM", "0"))              # 0 = don't validate
PCA_PATH = os.getenv("PCA_PATH", "")                                # if set, project to this size

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

# -------- Helpers --------
def get_conn():
    return psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDATABASE, user=PGUSER, password=PGPASSWORD
    )

def stream_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for item in data:
                yield item
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    id_ = rec.get("id") or rec.get("_id") or rec.get("doc_id") or rec.get("uuid")
    content = rec.get("content") or rec.get("text") or rec.get("page_content") or ""
    metadata = rec.get("metadata") or rec.get("meta") or {}
    embedding = rec.get("embedding")

    if id_ is None:
        id_ = str(abs(hash(json.dumps(rec, ensure_ascii=False, sort_keys=True))))

    if isinstance(metadata, (str, int, float, bool)) or metadata is None:
        metadata = {"value": metadata}

    return {
        "id": str(id_),
        "content": str(content),
        "metadata": metadata,
        "embedding": embedding
    }

def table_has_column(conn, table: str, col: str) -> bool:
    schema, _, name = table.rpartition(".")
    schema = schema or "public"
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
            LIMIT 1
        """, (schema, name, col))
        return cur.fetchone() is not None

class OptionalPCA:
    def __init__(self, path: str):
        self.enabled = bool(path)
        if not self.enabled:
            self.C = None
            self.mu = None
            return
        try:
            import numpy as np
        except ImportError:
            print("ERROR: numpy is required when PCA_PATH is set.", file=sys.stderr)
            sys.exit(1)
        pz = np.load(path)
        self.C = pz["components"].astype("float32")  # (k, d)
        self.mu = pz["mean"].astype("float32")       # (d,)
        self.np = np

    def project(self, emb: List[float]) -> List[float]:
        if not self.enabled or emb is None:
            return emb
        x = self.np.asarray(emb, dtype="float32")
        y = self.C @ (x - self.mu)
        return y.astype("float32").tolist()

def main():
    use_pca = OptionalPCA(PCA_PATH)

    conn = get_conn()
    conn.autocommit = False

    # Check columns
    has_content = table_has_column(conn, TABLE_NAME, "content")
    has_metadata = table_has_column(conn, TABLE_NAME, "metadata")
    has_target_emb = table_has_column(conn, TABLE_NAME, TARGET_EMB_COL)

    if not (has_content and has_metadata):
        print(f"ERROR: Table {TABLE_NAME} must have 'content' and 'metadata' columns.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    rows = []
    total = 0

    # Build dynamic SQL depending on embedding column availability
    if has_target_emb:
        upsert_sql = f"""
            INSERT INTO {TABLE_NAME} (id, content, metadata, {TARGET_EMB_COL})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata,
              {TARGET_EMB_COL} = COALESCE(EXCLUDED.{TARGET_EMB_COL}, {TABLE_NAME}.{TARGET_EMB_COL})
        """
    else:
        upsert_sql = f"""
            INSERT INTO {TABLE_NAME} (id, content, metadata)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata
        """

    try:
        for rec in tqdm(stream_records(INPUT_PATH), desc="Upserting"):
            n = normalize(rec)

            # Prepare embedding value
            emb_val = n["embedding"]
            if use_pca.enabled and emb_val:
                emb_val = use_pca.project(emb_val)

            # Optional dimension check (if user provided)
            if TARGET_EMB_DIM and emb_val:
                if len(emb_val) != TARGET_EMB_DIM:
                    # Mismatch; skip embedding for this row but still upsert other fields
                    emb_val = None

            if has_target_emb:
                rows.append((
                    n["id"],
                    n["content"],
                    json.dumps(n["metadata"], ensure_ascii=False),
                    emb_val
                ))
            else:
                rows.append((
                    n["id"],
                    n["content"],
                    json.dumps(n["metadata"], ensure_ascii=False)
                ))

            if len(rows) >= BATCH_SIZE:
                with conn.cursor() as cur:
                    execute_values(cur, upsert_sql, rows)
                conn.commit()
                total += len(rows)
                rows.clear()

        if rows:
            with conn.cursor() as cur:
                execute_values(cur, upsert_sql, rows)
            conn.commit()
            total += len(rows)

        print(f"Done. Upserted {total} rows into {TABLE_NAME}.")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()