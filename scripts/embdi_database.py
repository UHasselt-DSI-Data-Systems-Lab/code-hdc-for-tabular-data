import duckdb
import faiss
import numpy as np
from gensim.models import KeyedVectors
from scripts.experimentconfig import ExperimentConfig
from scripts.utils import load_config
from column_config import proj_columns_datasets


class EmbDIDatabase:
    def __init__(self, path: str, config: ExperimentConfig, name: str):
        self.config = config
        self.embeddings = KeyedVectors.load_word2vec_format(path, binary=False)
        self.name = name

        self.rows = []
        self.row_vectors = []
        self.values = {}
        self.val_indeces = {}
        self.encode_rows()
        self.encode_values()
        self.row_index = self.create_index(self.row_vectors)

    def encode_rows(self):
        row_vectors = []
        rows = []
        bad_rows = []
        for row in duckdb.sql(
            f"select * from db order by id limit {self.config.row_count}"
        ).fetchall():
            try:
                vector = self.embeddings[f"idx__{row[0] - 1}"]
                row_vectors.append(vector)
                rows.append(row)
            except KeyError:
                # print(e)
                bad_rows.append(row)
        self.rows = np.array(rows)
        self.row_vectors = np.array(row_vectors)
        if len(bad_rows) > 0:
            print(len(bad_rows), "bad rows")
            # print(set([r[0] for r in bad_rows]))

    def encode_values(self):
        tts = [
            (tt, self.embeddings[tt])
            for tt in self.embeddings.index_to_key
            if tt.startswith("tt__")
        ]
        vals = {col: {"values": [], "vectors": []} for col in self.config.columns}
        for tt, tt_vec in tts:
            for col in vals:
                if tt[4:].startswith(col):
                    vals[col]["values"].append(tt[4:].replace(f"{col}_", ""))
                    vals[col]["vectors"].append(tt_vec)
        val_indeces = {}
        for col in vals:
            val_indeces[col] = self.create_index(np.array(vals[col]["vectors"]))
        self.values = vals
        self.val_indeces = val_indeces

    def create_index(self, vectors):
        index = faiss.IndexFlatIP(self.embeddings.vector_size)
        vectors_copy = vectors.copy()
        faiss.normalize_L2(vectors_copy)
        index.add(vectors_copy)  # type: ignore
        return index

    def selection_vector(self, selections):
        """Generate a selection vector based on the given selection criteria.
        E.g. selections = [("a_1", "v_1", "="), ("a_2", "v_2", "!=")]"""
        vector = np.zeros((1, self.embeddings.vector_size))[0]
        for attribute, value, condition in selections:
            n = duckdb.sql(
                f"select count(*) as number from db where {attribute} = '{str(value)}'"
            ).fetchall()
            attr_rows = n[0][0]
            logarithm = np.log((self.config.row_count + 1) / (attr_rows + 1))
            idf = 0.1 + 0.9 * logarithm / np.log(self.config.row_count + 1)
            sign = -1 if condition == "!=" else 1
            try:
                tt_vector = self.embeddings[f"tt__{attribute}_{value}"]
            except:
                tt_vector = self.embeddings[f"tt__{attribute}_{value}.0"]
            vector = vector + sign * idf * tt_vector
        vector = 2 * (vector - vector.min()) / (vector.max() - vector.min()) - 1
        vector = np.array([vector])
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-12, a_max=None)
        vector = vector / norm
        return vector

    def select_rows(self, selections, threshold=0.5, top_k=0):
        vector = self.selection_vector(selections)
        if top_k > 0:
            D, K = self.row_index.search(vector, top_k)  # type: ignore
            J = K[0]
        else:
            _, D, J = self.row_index.range_search(vector, threshold)  # type: ignore
        return self.rows[J]

    def project(self, row_id, column):
        vector = self.row_vectors[row_id - 1]
        _, indexes = self.val_indeces[column].search(np.array([vector]), 1)
        return self.values[column]["values"][indexes[0][0]]


if __name__ == "__main__":
    table_name = "movie"
    cfg = load_config()
    config = ExperimentConfig.load_db(
        db_path=cfg["datasets"][table_name]["filename"],
        proj_columns=proj_columns_datasets[table_name],
        row_count=4,
        # row_count=cfg["datasets"][table_name]["row_count"],
        col_count=3,
    )
    # print(f"cells = {len(config.cells)}\n#rows = {config.row_count}\n#cols = {config.col_count}")
    print(config.db)
    # print(config.columns)
    # print(config.cells)
    # print(config.cells_r)

    hdc = EmbDIDatabase(
        path=cfg["datasets"][table_name]["embdi"].format(num_cols=config.col_count),
        config=config,
        name="random model",
    )
    r = hdc.select_rows(
        [
            ("actor_1", "cch_pounder", "=")  # ,
            # ("actor_2", "joel_david_moore", "="),
            # ("actor_3", "wes_studi", "="),
        ],
        threshold=-1,
        top_k=1,
    )
    print(r)
