import duckdb
import pandas as pd
import numpy as np
from scripts.embdi_database import EmbDIDatabase
from scripts.experimentconfig import ExperimentConfig
from scripts.hdc_database import HRRDatabase
from scripts.query_generation import query_parser
from scripts.utils import (
    load_config,
    duplicate_specific_quotes,
    precision_recall,
    save_dataframe,
    predicted_by_dim,
    set_seed,
)
from column_config import proj_columns_datasets
import json


def check_empty(
    cnf,
    table_name="movie",
    m=2,
):
    """
    Helper to validate zero-match query file.
    """
    config = ExperimentConfig.load_db(
        db_path=cnf["datasets"][table_name]["filename"],
        proj_columns=proj_columns_datasets[table_name],
        row_count=cnf["datasets"][table_name]["row_count"],
        col_count=m,
    )
    with open(
        f"{cnf['paths']['queries']}/empty_{table_name}_first_{m}.json", "r"
    ) as file:
        queries = json.load(file)

    for n in queries:
        n_len_queries = queries[n]
        for i, query in enumerate(n_len_queries):
            true_res = [
                str(r[0])
                for r in duckdb.sql(
                    duplicate_specific_quotes(query).replace("select ", "select id,")
                ).fetchall()
            ]
            if len(true_res) > 0:
                print(query)
                print(len(true_res))


def execute_projection_for_model(config, outputfile, model, m, rows, repid=1):
    """
    ATTRIBUTE PROJECTION for GIVEN model.
    1. Get true value for each column, for each row
    2. Get projected value for each column, for each row
    3. Store results
    """
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
    """
    ATTRIBUTE PROJECTION for EMBDI model.
    """
    embdi_db = EmbDIDatabase(
        path=cnf["datasets"][table_name]["embdi"].format(num_cols=m, dim=dim),
        config=config,
        name=f"EmbDI_{dim}",
    )
    execute_projection_for_model(config, outputfile, embdi_db, m, rows, repid)


def run_hrr_projection(config, dim, outputfile, m, rows, repid=1):
    """
    ATTRIBUTE PROJECTION for HRR model.
    """
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}")
    execute_projection_for_model(config, outputfile, hrr_db, m, rows, repid)


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
    """
    ROW RETREIVAL helper function to compute metrics
    """
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
    """
    ROW RETRIEVAL for GIVEN model
    for range of thresholds OR Top-k values, depending on topk flag.
    1. For each query, get the true result set (using duckdb)
    2. For each query, get the retrieved result set from the model, for each threshold or top-k value.
    3. Compute precision and recall, and store results in a list of dicts
    """
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
    """
    ROW RETREIVAL - EMBDI model
    for range of thresholds OR Top-k values, depending on topk flag.
    """
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


def run_hrr_selection(
    config, outputfile, dim, m, queries, min_t, max_t, step, ri, topk=False
):
    """
    ROW RETREIVAL - HRR model
    for range of thresholds OR Top-k values, depending on topk flag.
    """
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}", model=None)
    hrr_results = execute_selection_for_model(
        config, hrr_db, m, queries, min_t, max_t, step, repid=ri, topk=topk
    )
    save_dataframe(
        pd.DataFrame(hrr_results),
        outputfile,
    )


def run_hrr_similarities(config, outputfile, dim, m, queries, ri):
    """
    ROW RETRIEVAL variation: get similarity for every row.
    Used for debugging and to understand the distribution of similarities for different queries.
    """
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}", model=None)
    results = []
    for n in queries:
        n_len_queries = queries[n]
        for i, query in enumerate(n_len_queries):
            parsed = query_parser(query)
            sim = hrr_db.get_similarities(parsed["selections"])
            for row_id, s in enumerate(sim):
                results.append(
                    {
                        "m": m,
                        "n": n,
                        "similarity": s,
                    }
                )
    save_dataframe(
        pd.DataFrame(results),
        outputfile,
    )


def run_hrr_empty_query_labeling(config, outputfile, dim, m, queries, ri):
    """
    ROW RETRIEVAL for zero-match queries.
    Get number of results for each query, using ONLY advised threshold values.
    This evaluates the pertinence of the predicted thresholds for empty queries.
    """
    hrr_db = HRRDatabase(dim=dim, config=config, name=f"HRR_{dim}", model=None)
    results = []
    for n in queries:
        n_len_queries = queries[n]
        for i, query in enumerate(n_len_queries):
            parsed = query_parser(query)
            ids = hrr_db.select_rows(
                parsed["selections"], threshold=predicted_by_dim(m, int(n), dim)
            )
            results.append(
                {
                    "m": m,
                    "n": n,
                    "model": hrr_db.name,
                    "results": len(ids),
                    "qid": f"{m}_{n}_{i}",
                    "ri": ri,
                }
            )
    save_dataframe(
        pd.DataFrame(results),
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
    """
    ROW RETRIEVAL for range of thresholds OR Top-k values, depending on topk flag.
    Run experiments for all m (number of columns in the table).
    """
    print("=" * 60)
    print(
        f"Running selection table='{table_name}' with query='{q_path}' and selection='{'topk' if topk else 'threshold'}'..."
    )

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

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024]:
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


def run_threshold_selection_variance(
    cnf,
    table_name="movie",
    q_path="equality",
    rep=3,
    topk=False,
):
    """
    ROW RETRIEVAL only for the advised threshold values, no threshold sweep.
    Run experiments for all m (number of columns in the table).
    """
    print("=" * 60)
    print(f"Running variance thresholds table='{table_name}' with query='{q_path}'...")

    selection_family = "topk" if topk else "threshold"

    outputfile = f"{cnf['paths']['processed']}/predicted_{selection_family}_{q_path}_{table_name}.csv"

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

        for ri in range(rep):
            print(f"Repetition {ri + 1}/{rep} for HDC models...")

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024]:
                for k, v in queries.items():
                    n = int(k)
                    t = predicted_by_dim(m, n, dim)

                    run_hrr_selection(
                        config,
                        outputfile,
                        dim,
                        m,
                        {k: v},
                        t,
                        t,
                        1,
                        ri,
                        topk=topk,
                    )


def run_projection(cnf, table_name="movie", n_rows=100, rep=3):
    """
    ATTRIBUTE PROJECTION main method.
    Run projection experiments for all m (number of columns in the table)."""
    print("=" * 60)
    print(f"Running projection table='{table_name}' with {n_rows} rows...")

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
            # for dim in [300, 512, 1024, 2048, 4096, 8192]:
            #     run_bsc_projection(config, dim, outputfile, m, rows, repid=ri)

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024]:
                run_hrr_projection(config, dim, outputfile, m, rows, repid=ri)

            # =======  Semantic HDC (using FastText embeddings)
            # run_semhdc_projection(config, cnf, outputfile, m, rows, repid=ri)


def run_empty_queries_similarity(
    cnf,
    table_name="movie",
    rep=1,
):

    for m in range(2, cnf["datasets"][table_name]["col_count"] + 1):
        # Load queries
        with open(
            f"{cnf['paths']['queries']}/empty_{table_name}_first_{m}.json", "r"
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

        for ri in range(rep):
            print(f"Repetition {ri + 1}/{rep} for HDC models...")

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024]:
                outputfile = (
                    f"{cnf['paths']['processed']}/empty_{table_name}_hrr{dim}.csv"
                )
                run_hrr_similarities(config, outputfile, dim, m, queries, ri)
                # run_hrr_empty_query_labeling(config, outputfile, dim, m, queries, ri)


def run_empty_queries_labeling(
    cnf,
    table_name="movie",
    rep=1,
):
    """
    ROW RETRIEVAL for zero-match queries, main method.
    Run empty query labeling experiments for all m (number of columns in the table).
    """
    for m in range(1, cnf["datasets"][table_name]["col_count"] + 1):
        # Load queries
        with open(
            f"{cnf['paths']['queries']}/empty_{table_name}_first_{m}.json", "r"
        ) as file:
            queries = json.load(file)

        print(f"Running experiments for m={m}")
        outputfile = f"{cnf['paths']['processed']}/emptylabel_{table_name}.csv"

        for ri in range(rep):
            print(f"Repetition {ri + 1}/{rep} for HDC models...")

            # =======  Holographic Reduced Rep (fully random)
            for dim in [300, 512, 1024]:
                # configuration has to be loaded inside
                # so the models can add new values (for empty queries for m=1)
                config = ExperimentConfig.load_db(
                    db_path=cnf["datasets"][table_name]["filename"],
                    proj_columns=proj_columns_datasets[table_name],
                    row_count=cnf["datasets"][table_name]["row_count"],
                    col_count=m,
                )
                run_hrr_empty_query_labeling(config, outputfile, dim, m, queries, ri)


if __name__ == "__main__":
    cnf = load_config()
    set_seed(cnf.get("seed", 0))

    # run_threshold_selection(
    #     cnf,
    #     table_name="dblp",
    #     q_path="nonequality",
    #     rep=3,
    #     min_t=-0.3,
    #     max_t=0.5,
    #     step=0.05,
    #     topk=False,
    # )

    # run_threshold_selection(
    #     cnf,
    #     table_name="dblp",
    #     q_path="equality",
    #     rep=3,
    #     min_t=0.1,
    #     max_t=1,
    #     step=0.05,
    #     topk=False,
    # )

    # run_threshold_selection(
    #     cnf,
    #     table_name="dblp",
    #     q_path="equality",
    #     rep=3,
    #     min_t=0.1,
    #     max_t=1,
    #     step=0.05,
    #     topk=True,
    # )

    # run_threshold_selection(
    #     cnf,
    #     table_name="movie",
    #     q_path="nonequality",
    #     rep=3,
    #     min_t=-0.3,
    #     max_t=0.5,
    #     step=0.05,
    #     topk=False,
    # )

    # run_threshold_selection(
    #     cnf,
    #     table_name="movie",
    #     q_path="equality",
    #     rep=3,
    #     min_t=0.1,
    #     max_t=1,
    #     step=0.05,
    #     topk=False,
    # )

    # run_threshold_selection(
    #     cnf,
    #     table_name="movie",
    #     q_path="equality",
    #     rep=3,
    #     min_t=0.1,
    #     max_t=1,
    #     step=0.05,
    #     topk=True,
    # )

    # run_projection(cnf, table_name="dblp", n_rows=50, rep=3)

    # run_projection(cnf, table_name="movie", n_rows=50, rep=3)

    run_empty_queries_labeling(cnf, table_name="movie", rep=3)

    run_threshold_selection_variance(
        cnf,
        table_name="movie",
        q_path="equality",
        rep=3,
        topk=False,
    )

    # run_empty_queries_similarity(
    #     cnf,
    #     table_name="movie",
    #     rep=1,
    # )

    # for m in range(2, 16):
    #     check_empty(cnf, table_name="movie", m=m)
