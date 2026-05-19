import faiss
import torch
import torchhd
import duckdb
import numpy as np
from torch.fft import fft, ifft
from scripts.experimentconfig import ExperimentConfig
from scripts.utils import load_config
from column_config import proj_columns_datasets


class HDCDatabase:
    def __init__(self, dim: int, config: ExperimentConfig, name: str):
        self.dim = dim
        self.config = config
        self.name = name

    def bind(self, vec1, vec2):
        pass

    def bundle(self, vector_list):
        pass

    def complement(self, vector):
        pass

    def gen_random_vectors(self, num_vectors):
        pass

    def gen_non_random_vectors(self, value_list):
        pass

    def create_index(self, vectors):
        pass

    def get_item_memory(self):
        pass

    def encode_row(self, key_values):
        pass

    def encode_rows(self):
        pass

    def selection_vector(self, selection_criteria):
        pass

    def select_rows(self, selection_criteria):
        pass


class HRRDatabase(HDCDatabase):
    def __init__(
        self,
        dim: int,
        config: ExperimentConfig,
        name: str,
        model: None = None,
    ):
        """
        A class to represent a real-valued vector HDC database.
        Rows are encoded as key-value structures.

        :param dim: The dimensionality of the hypervectors.
        :param config: The experiment configuration.
        :param name: The name of the database for output files.
        :param model: An optional pre-trained model to generate non-random vectors for cell values. If None, all vectors will be random.
        """
        self.dim = dim
        self.config = config
        self.name = name
        self.model = model

        self.item_memory = self.get_item_memory()
        self.rows = []
        self.row_vectors = []
        self.encode_rows()
        self.val_index = self.create_index(self.item_memory[self.config.col_count :])
        self.row_index = self.create_index(self.row_vectors)

    @staticmethod
    def normalize(vector):
        """Normalize a hypervector to have unit length."""
        return torch.nn.functional.normalize(vector, p=2, dim=-1)

    def bind(self, vec1, vec2):
        """Bind two hypervectors together using circular convolution."""
        return self.normalize(vec1.bind(vec2))

    def bundle(self, vector_list):
        """Bundle a list of hypervectors together by summing and normalizing."""
        return self.normalize(
            torchhd.HRRTensor(torch.sum(torch.stack(vector_list), dim=-2))
        )

    def complement(self, vector):
        """Compute the complement of a hypervector."""
        return vector.negative()

    def complex_unit(self, vec):
        """Normalize a complex vector to have unit magnitude."""
        fft_result = fft(vec)
        magnitude = torch.abs(fft_result)
        epsilon = torch.finfo(magnitude.dtype).eps
        safe_magnitude = magnitude.clamp(min=epsilon)
        normalized_fft_vector = torch.divide(fft_result, safe_magnitude)
        result = self.normalize(torch.real(ifft(normalized_fft_vector)))
        return result

    def gen_random_vectors(self, num_vectors):
        """Generate random HRR hypervectors for the given number of vectors."""
        return torchhd.HRRTensor.random(num_vectors, self.dim)

    # def gen_non_random_vectors(self, value_list):
    #     """Generate non-random vectors for the given value list using the specified method."""
    #     embd_vectors = [self.model.get_vector(key) for key in value_list]  # type: ignore
    #     comb_vectors = torch.stack([*embd_vectors])  # type: ignore
    #     norm_vecs = self.complex_unit(comb_vectors)
    #     return norm_vecs

    def create_index(self, vectors):
        """Create a FAISS index for the given vectors."""
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)  # type: ignore
        return index

    def get_item_memory(self):
        """Generate the item memory for the database.
        Random vectors for columns and possibly non-random vectors for cells."""
        # if self.model is None:
        return self.gen_random_vectors(self.config.item_memory_size)
        # else:
        #     random_vectors = self.gen_random_vectors(
        #         self.config.item_memory_size - len(self.config.cells)
        #     )
        #     embd_vectors = self.gen_non_random_vectors(self.config.cells)
        #     comb_vectors = torch.stack([*random_vectors] + [*embd_vectors])
        #     norm_vecs = self.complex_unit(comb_vectors)
        #     return norm_vecs

    def encode_row(self, row_tuple):
        """Encode a row tuple into a hypervector (excluding unique id)."""
        bounds_list = [
            self.bind(
                self.item_memory[i],
                self.item_memory[self.config.cells[row_tuple[i + 1]]],
            )
            for i in range(len(row_tuple) - 1)
        ]
        return self.bundle(bounds_list)

    def encode_rows(self):
        """Encode all rows in the database into hypervectors."""
        row_vectors = []
        rows = []
        for row in duckdb.sql(
            f"select * from db order by id limit {self.config.row_count}"
        ).fetchall():
            row_vector = self.encode_row(row)
            row_vectors.append(row_vector)
            rows.append(row)

        self.rows = np.array(rows)  # rows as symbolic tuples
        self.row_vectors = torch.stack(row_vectors)  # rows as hypervectors

    def selection_vector(self, selection_criteria):
        """Generate a selection hypervector based on the given selection criteria.
        E.g. selection_criteria = [("a_1", "v_1", "="), ("a_2", "v_2", "!=")]"""
        bundle_list = []
        for column, value, condition in selection_criteria:
            # column vector
            col_vector = self.item_memory[self.config.columns[column]]

            # value vector
            if value in self.config.cells:
                val_vector = self.item_memory[self.config.cells[value]]
            else:
                if self.model is not None:
                    val_vector = self.model.get_vector(value)
                else:
                    val_vector_tensor = self.gen_random_vectors(1)
                    val_vector = val_vector_tensor[0]

                # add new value to config and item memory
                _ = self.config.add_value(value)
                self.item_memory = torch.cat(
                    [self.item_memory, val_vector_tensor], dim=0
                )  # type: ignore
                self.val_index.add(val_vector_tensor)  # type: ignore

            if condition == "!=":
                bundle_list.append(self.bind(col_vector, self.complement(val_vector)))
            else:
                bundle_list.append(self.bind(col_vector, val_vector))
        vector = self.bundle(bundle_list)
        return vector

    def select_rows(self, selection_criteria, threshold=0.5, top_k=0):
        """Select rows from the database based on selection criteria
        and similarity threshold or top_k."""
        vector = self.selection_vector(selection_criteria)
        if top_k > 0:
            D, K = self.row_index.search(np.array([vector]), top_k)  # type: ignore
            J = K[0]
        else:
            _, D, J = self.row_index.range_search(np.array([vector]), threshold)  # type: ignore
        return self.rows[J]

    def project(self, row_id, column):
        vector = self.row_vectors[row_id - 1]
        col_vector = self.item_memory[self.config.columns[column]]
        estimation_vector = self.bind(vector, col_vector.inverse())

        _, indexes = self.val_index.search(np.array([estimation_vector]), 1)  # type: ignore
        return self.config.cells_r[indexes[0][0] + self.config.col_count]

    def get_similarities(self, selection_criteria):
        vector = self.selection_vector(selection_criteria)
        dots = (self.row_vectors @ vector.squeeze()).tolist()  # type: ignore
        return dots


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

    hdc = HRRDatabase(dim=300, config=config, name="random model")
    r = hdc.select_rows(
        [
            ("actor_1", "cch_pounder", "="),
            ("actor_2", "joel_david_moore", "="),
            ("actor_3", "wes_studi", "="),
        ],
        threshold=-1,
        top_k=1,
    )
    print(r)
