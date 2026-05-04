import duckdb
import pandas as pd
import numpy as np
from scripts.embdi_database import EmbDIDatabase
from scripts.experimentconfig import ExperimentConfig
from scripts.models import PreTrainedModel
from scripts.hdc_database import BSCDatabase, HRRDatabase
from scripts.query_generation import query_parser
from scripts.utils import (
    load_config,
    duplicate_specific_quotes,
    precision_recall,
    save_dataframe,
)
from column_config import proj_columns_datasets
import json


def execute_projection_for_model(config, outputfile, model, m, rows, repid=1):
    results = []
    for row in rows:
        row_id = row[0]
        for col in config.columns:
            true_value = row[config.columns[col] + 1]
            projected_value = model.project(row_id, col)
            results.append(
                {
                    "m": m,
                    "col": col,
                    "model": model.name,
                    "row_id": row_id,
                    "match": true_value == projected_value,
                    "true_value": true_value,
                    "projected_value": projected_value,
                    "repetition": repid,
                }
            )
    save_dataframe(
        pd.DataFrame(results),
        outputfile,
    )


def run_embdi_projection(config, cnf, table_name, outputfile, m, dim, rows, repid=1):
    embdi_db = EmbDIDatabase(
        path=cnf["datasets"][table_name]["embdi"].format(num_cols=m, dim=dim),
        config=config,
        name=f"EmbDI_{dim}",
    )
    execute_projection_for_model(config, outputfile, embdi_db, m, rows, repid)


def run_bsc_projection(config, dim, outputfile, m, rows, repid=1):
    bsc_db = BSCDatabase(dim=dim, config=config, name=f"BSC_{dim}")
    execute_projection_for_model(config, outputfile, bsc_db, m, rows, repid)


def run_hrr_projection(config, dim, outputfile, m, rows, repid=1):
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}")
    execute_projection_for_model(config, outputfile, hrr_db, m, rows, repid)


def run_semhdc_projection(config, cnf, outputfile, m, rows, repid=1):
    model = PreTrainedModel(
        name=cnf["models"]["fasttext"]["name"],
        model_path=cnf["models"]["fasttext"]["path"],
    )
    sem_db = HRRDatabase(
        dim=model.vector_size,
        config=config,
        name="SemHDC_FastText_300",
        model=model,
    )
    execute_projection_for_model(config, outputfile, sem_db, m, rows, repid)


def process_retrieved_rows(
    retrieved_rows,
    true_rids,
    results_list,
    m,
    model,
    n,
    i,
    query,
    repid,
    config,
    t=None,
    topk=None,
):
    retrieved_rids = (
        set([str(r[0]) for r in retrieved_rows]) if len(retrieved_rows) > 0 else set()
    )

    true_positives = len(true_rids.intersection(retrieved_rids))
    false_positives = len(retrieved_rids - true_rids)
    false_negatives = len(true_rids - retrieved_rids)
    true_negatives = (
        config.row_count - true_positives - false_positives - false_negatives
    )
    precision, recall = precision_recall(
        n_truth=len(true_rids),
        n_retrieved=len(retrieved_rids),
        inter=true_positives,
    )
    results_list.append(
        (
            {
                "m": m,
                "n": n,
                "t": t,
                "topk": topk,
                "model": model.name,
                "gtruth": len(true_rids),
                "retrieved": len(retrieved_rids),
                "precision": precision,
                "recall": recall,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "tn": true_negatives,
                "repetition": repid,
                "qid": f"{n}_{i}",
                "query": query,
            }
        )
    )


def execute_selection_for_model(
    config, model, m, queries, min_t, max_t, step, repid=1, topk=False
):
    results = []
    for n in queries:
        n_len_queries = queries[n]
        for i, query in enumerate(n_len_queries):
            # ground truth
            true_res = [
                str(r[0])
                for r in duckdb.sql(
                    duplicate_specific_quotes(query).replace("select ", "select id,")
                ).fetchall()
            ]
            true_rids = set(true_res)

            parsed = query_parser(query)

            if topk:
                for k in [1, 2, 5, 10, 20]:
                    res = model.select_rows(parsed["selections"], top_k=k)
                    process_retrieved_rows(
                        retrieved_rows=res,
                        true_rids=true_rids,
                        results_list=results,
                        m=m,
                        model=model,
                        n=n,
                        i=i,
                        query=query,
                        repid=repid,
                        config=config,
                        t=None,
                        topk=k,
                    )
            else:
                t = min_t
                while t <= max_t:
                    res = model.select_rows(parsed["selections"], threshold=t)
                    process_retrieved_rows(
                        retrieved_rows=res,
                        true_rids=true_rids,
                        results_list=results,
                        m=m,
                        model=model,
                        n=n,
                        i=i,
                        query=query,
                        repid=repid,
                        config=config,
                        t=t,
                        topk=None,
                    )

                    t = round(t + step, 2)
    return results


def run_embdi_selection(
    config, cnf, table_name, outputfile, m, dim, queries, min_t, max_t, step, topk=False
):
    embdi_db = EmbDIDatabase(
        path=cnf["datasets"][table_name]["embdi"].format(num_cols=m, dim=dim),
        config=config,
        name=f"EmbDI_{dim}",
    )
    embdi_results = execute_selection_for_model(
        config, embdi_db, m, queries, min_t, max_t, step, topk=topk
    )
    save_dataframe(
        pd.DataFrame(embdi_results),
        outputfile,
    )


def run_bsc_selection(
    config, outputfile, dim, m, queries, min_t, max_t, step, ri, topk=False
):
    bsc_db = BSCDatabase(dim=dim, config=config, name=f"BSC_{dim}")
    bsc_results = execute_selection_for_model(
        config, bsc_db, m, queries, min_t, max_t, step, repid=ri, topk=topk
    )
    save_dataframe(
        pd.DataFrame(bsc_results),
        outputfile,
    )


def run_hrr_selection(
    config, outputfile, dim, m, queries, min_t, max_t, step, ri, topk=False
):
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}", model=None)
    hrr_results = execute_selection_for_model(
        config, hrr_db, m, queries, min_t, max_t, step, repid=ri, topk=topk
    )
    save_dataframe(
        pd.DataFrame(hrr_results),
        outputfile,
    )


def run_semhdc_selection(
    config, cnf, outputfile, m, queries, min_t, max_t, step, ri, topk=False
):
    model = PreTrainedModel(
        name=cnf["models"]["fasttext"]["name"],
        model_path=cnf["models"]["fasttext"]["path"],
    )
    sem_db = HRRDatabase(
        dim=model.vector_size,
        config=config,
        name="SemHDC_FastText_300",
        model=model,
    )
    sem_results = execute_selection_for_model(
        config, sem_db, m, queries, min_t, max_t, step, repid=ri, topk=topk
    )
    save_dataframe(
        pd.DataFrame(sem_results),
        outputfile,
    )


def run_threshold_selection(
    cnf,
    table_name="movie",
    q_path="equality",
    rep=3,
    min_t=0.1,
    max_t=0.95,
    step=0.05,
    topk=False,
):
    """Run experiments for all m (number of columns in the table)."""

    selection_family = "topk" if topk else "threshold"

    outputfile = (
        f"{cnf['paths']['processed']}/{selection_family}_{q_path}_{table_name}.csv"
    )

    for m in range(1, cnf["datasets"][table_name]["col_count"] + 1):
        # Load queries
        with open(
            f"{cnf['paths']['queries']}/{q_path}_{table_name}_first_{m}.json", "r"
        ) as file:
            queries = json.load(file)

        # common configuration for all databases
        config = ExperimentConfig.load_db(
            db_path=cnf["datasets"][table_name]["filename"],
            proj_columns=proj_columns_datasets[table_name],
            row_count=cnf["datasets"][table_name]["row_count"],
            col_count=m,
        )
        print(f"Running experiments for m={m}")

        print("Embdi model...")

        # =======  EMBDI
        for dim in [300, 512]:
            run_embdi_selection(
                config,
                cnf,
                table_name,
                outputfile,
                m,
                dim,
                queries,
                min_t,
                max_t,
                step,
                topk=topk,
            )

        for ri in range(rep):
            print(f"Repetition {ri + 1}/{rep} for HDC models...")

            # =======  Binary Spatter Codes (fully random)
            for dim in [300, 512, 1024, 2048, 4096, 8192]:
                run_bsc_selection(
                    config,
                    outputfile,
                    dim,
                    m,
                    queries,
                    min_t,
                    max_t,
                    step,
                    ri,
                    topk=topk,
                )

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024, 2048]:
                run_hrr_selection(
                    config,
                    outputfile,
                    dim,
                    m,
                    queries,
                    min_t,
                    max_t,
                    step,
                    ri,
                    topk=topk,
                )

            # =======  Semantic HDC (using FastText embeddings)
            run_semhdc_selection(
                config, cnf, outputfile, m, queries, min_t, max_t, step, ri, topk=topk
            )


def run_projection(cnf, table_name="movie", n_rows=100, rep=3):
    """Run projection experiments for all m (number of columns in the table)."""

    outputfile = f"{cnf['paths']['processed']}/projection_{table_name}_{n_rows}.csv"

    for m in range(1, cnf["datasets"][table_name]["col_count"] + 1):
        # common configuration for all databases
        config = ExperimentConfig.load_db(
            db_path=cnf["datasets"][table_name]["filename"],
            proj_columns=proj_columns_datasets[table_name],
            row_count=cnf["datasets"][table_name]["row_count"],
            col_count=m,
        )
        print(f"Running projection experiments for m={m}")

        # sample n_rows rows from the database
        row_idx = np.random.choice(config.row_count, n_rows)
        row_idx = row_idx + np.ones(n_rows)
        rows = duckdb.sql(f"select * from db where id in {row_idx.tolist()}").fetchall()

        # =======  EMBDI
        for dim in [300, 512]:
            run_embdi_projection(
                config, cnf, table_name, outputfile, m, dim, rows, repid=1
            )

        for ri in range(rep):
            print(f"Repetition {ri + 1}/{rep} for HDC models...")

            # =======  Binary Spatter Codes (fully random)
            for dim in [300, 512, 1024, 2048, 4096, 8192]:
                run_bsc_projection(config, dim, outputfile, m, rows, repid=ri)

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024, 2048]:
                run_hrr_projection(config, dim, outputfile, m, rows, repid=ri)

            # =======  Semantic HDC (using FastText embeddings)
            run_semhdc_projection(config, cnf, outputfile, m, rows, repid=ri)


if __name__ == "__main__":
    cnf = load_config()
    run_threshold_selection(
        cnf,
        table_name="dblp",
        q_path="nonequality",
        rep=3,
        min_t=-0.3,
        max_t=0.5,
        step=0.05,
        topk=False,
    )
    # run_projection(cnf, table_name="movie", n_rows=50, rep=3)
