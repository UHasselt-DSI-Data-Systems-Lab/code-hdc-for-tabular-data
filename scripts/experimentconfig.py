from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import duckdb
from scripts.utils import load_config
from column_config import proj_columns_datasets
# from config import paths, proj_columns_datasets, row_counts


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Config class to encapsulate the relational db for different experimental settings
    """

    db: duckdb.DuckDBPyRelation
    proj_columns: List[Tuple[str, ...]]
    row_count: int
    col_count: int
    item_memory_size: int
    columns: Dict[str, Any]
    cells: Dict[str, Any]
    columns_r: List[str]
    cells_r: Dict[int, Any]

    @staticmethod
    def load_db(
        db_path: str,
        proj_columns: List[Tuple[str, ...]],
        row_count: int,
        col_count: int,
    ) -> "ExperimentConfig":
        # load the database
        fulldb = duckdb.read_csv(
            db_path.replace(".csv", f"_first_{col_count}.csv"),
            null_padding=True,
            delimiter=",",
            escapechar='"',
        )
        proj_str = ",".join(
            [f"{p[1]} as {p[0]}" for p in proj_columns[: col_count + 1]]
        )
        proj_str_wo_id = ",".join(
            [f"{p[1]} as {p[0]}" for p in proj_columns[1 : col_count + 1]]
        )
        duckdb.sql("drop table if exists db")
        duckdb.sql(
            f"create table db as select row_number() over () as id, {proj_str_wo_id} from fulldb limit {row_count}"
        )
        db = duckdb.sql(f"select {proj_str} from db limit {row_count}")

        # load column ids
        columns = {}
        columns_r = []
        for i, col in enumerate(duckdb.sql("describe select * from db").fetchall()):
            if col[0] == "id":
                continue
            columns[col[0]] = i - 1
            columns_r.append(col[0])

        # load cell values ids
        cells = {}
        cells_r = {}
        values_set = set()
        values = []
        for c, _ in proj_columns[1 : col_count + 1]:
            res = [row[0] for row in duckdb.sql(f"select {c} from db").fetchall()]
            for r in res:
                if r not in values_set:
                    values.append(r)
                    values_set.add(r)
        cells.update({label: len(columns_r) + i for i, label in enumerate(values)})
        cells_r.update({len(columns_r) + i: label for i, label in enumerate(values)})

        # memory size
        memory_size = len(columns_r) + len(values)
        return ExperimentConfig(
            db,
            proj_columns,
            row_count,
            len(columns_r),
            memory_size,
            columns,
            cells,
            columns_r,
            cells_r,
        )

    def add_value(self, value: str):
        if value not in self.cells:
            new_id = len(self.columns_r) + len(self.cells)  # - 1
            self.cells[value] = new_id
            self.cells_r[new_id] = value
            return new_id


if __name__ == "__main__":
    table_name = "movie"
    cfg = load_config()
    config = ExperimentConfig.load_db(
        db_path=cfg["datasets"][table_name]["filename"],
        proj_columns=proj_columns_datasets[table_name],
        row_count=4,
        # row_count=cfg["datasets"][table_name]["row_count"],
        col_count=1,
    )
    print(
        f"cells = {len(config.cells)}\n#rows = {config.row_count}\n#cols = {config.col_count}"
    )
    print(config.db)
    print(config.columns)
    print(config.cells)
    print(config.cells_r)
