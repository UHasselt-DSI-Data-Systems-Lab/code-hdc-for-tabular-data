import json

import pandas as pd
import random
from itertools import combinations
import sqlparse

from scripts.utils import set_seed, load_config


def generate_queries(dataset_path, queries_per_n, table_name="db", eq="equality"):
    df = pd.read_csv(dataset_path)
    columns = df.columns.tolist()
    m = len(columns)
    all_queries = {}

    for n in range(1, m + 1):
        all_queries[n] = []
        print(f"Generating queries for n={n}...")

        unique_queries = set()

        for _ in range(queries_per_n):
            # Pick a random row to get valid values (v1, v2, ... vn)
            while True:
                # Get all possible combinations of n columns
                col_combinations = list(combinations(columns, n))

                # 1. Pick a random combination of columns
                selected_cols = random.choice(col_combinations)
                random_row = df.sample(n=1).iloc[0]

                # 3. Construct the WHERE clause
                conditions = []
                for col in selected_cols:
                    val = random_row[col]
                    # Format value: add quotes if it's a string
                    symbol = "=" if eq == "equality" else "!="
                    conditions.append(
                        f"{col} {symbol} '{val}'"
                        if val != "NULL_VAL"
                        else f"{col} {symbol} 'nan'"
                    )

                query = f"select * from {table_name} where {' and '.join(conditions)}"
                if query not in unique_queries:
                    all_queries[n].append(query)
                    unique_queries.add(query)
                    break

    return all_queries


def generate_queries_for_all_n(
    dataset_name,
    max_m,
    queries_per_n,
    eq="nonequality",
    table_name="db",
    seed=0,
):
    set_seed(seed)
    for m in range(1, max_m + 1):
        dataset_path = f"data/interim/{dataset_name}_first_{m}.csv"
        queries = generate_queries(
            dataset_path, queries_per_n, table_name=table_name, eq=eq
        )
        with open(
            f"data/queries/{eq}_{dataset_name}_first_{m}.json",
            "w",
        ) as f:
            json.dump(queries, f)


def query_parser(query):
    # we assume the format is always select from where, with equality and non equality
    tokens = sqlparse.parse(query)[0]
    project = str(tokens[2])  # attribute to project
    table = str(tokens[6])  # from table
    selections = []  # filters

    ref_str = str(tokens[8]).split(" and ")
    for clause in ref_str:
        frags = clause.strip(" ").split(" ")
        if len(frags) == 4:
            frags.pop(0)
        # selections.append((frags[0], frags[2].replace("'", ""), frags[1]))
        selections.append((frags[0], frags[2][1:-1], frags[1]))
    return {"select": project, "table": table, "selections": selections}


def generate_empty_queries(dataset_path, queries_per_n, table_name="db"):
    df = pd.read_csv(dataset_path)
    columns = df.columns.tolist()
    m = len(columns)
    all_queries = {}

    def make_condition(col, val):
        return f"{col} = '{val}'" if val != "NULL_VAL" else f"{col} = 'nan'"

    def verify_empty(selected_cols, values):
        mask = pd.Series([True] * len(df))
        for col, val in zip(selected_cols, values):
            mask &= df[col].astype(str) == val
        return mask.sum() == 0

    max_attempts = queries_per_n * 1000

    for n in range(1, m + 1):
        all_queries[n] = []
        print(f"Generating zero-result queries for n={n}...")

        unique_queries = set()
        col_combinations = list(combinations(columns, n))

        attempts = 0
        while len(all_queries[n]) < queries_per_n and attempts < max_attempts:
            attempts += 1

            selected_cols = random.choice(col_combinations)

            if m == 1:
                print("  m=1 special case: no other columns available, skipping.")
                break

            if n == 1:
                # Value from a different column, verified absent from the selected column
                col = selected_cols[0]
                other_cols = [c for c in columns if c != col]
                source_col = random.choice(other_cols)
                val = str(df[source_col].sample(n=1).iloc[0])

                if val in set(df[col].astype(str).tolist()):
                    continue

                query = f"select * from {table_name} where {make_condition(col, val)}"

            else:
                # Pick a base row: n-1 cols take real values from it
                base_row = df.sample(n=1).iloc[0]
                correct_cols = list(selected_cols[:-1])
                corrupt_col = selected_cols[-1]

                # Corrupt col: pick a value that EXISTS in that column but breaks the combination
                col_values = list(
                    set(df[corrupt_col].astype(str).tolist())
                    - {str(base_row[corrupt_col])}
                )
                if not col_values:
                    continue
                corrupt_val = random.choice(col_values)

                values = [str(base_row[col]) for col in correct_cols] + [corrupt_val]

                if not verify_empty(selected_cols, values):
                    continue

                conditions = [
                    make_condition(col, val) for col, val in zip(selected_cols, values)
                ]
                query = f"select * from {table_name} where {' and '.join(conditions)}"

            if query not in unique_queries:
                all_queries[n].append(query)
                unique_queries.add(query)

        if len(all_queries[n]) < queries_per_n:
            print(
                f"  Warning: only found {len(all_queries[n])} zero-result queries for n={n}"
            )

    return all_queries


def generate_empty_queries_single_column(
    dataset_path_1, dataset_path_2, queries_per_n, table_name="db"
):
    df1 = pd.read_csv(dataset_path_1)
    df2 = pd.read_csv(dataset_path_2)
    col = df1.columns[0]
    columns = df2.columns.tolist()
    all_queries = {}

    def make_condition(col, val):
        return f"{col} = '{val}'" if val != "NULL_VAL" else f"{col} = 'nan'"

    max_attempts = queries_per_n * 1000

    n = 1
    all_queries[n] = []
    print(f"Generating empty queries for n={n}...")

    unique_queries = set()

    attempts = 0
    while len(all_queries[n]) < queries_per_n and attempts < max_attempts:
        attempts += 1

        # Pick a value from any column OTHER than the selected one
        other_cols = [c for c in columns if c != col]
        source_col = random.choice(other_cols)
        val = str(df2[source_col].sample(n=1).iloc[0])

        # Verify it does not appear in the selected column
        if val in set(df1[col].astype(str).tolist()):
            continue

        query = f"select * from {table_name} where {make_condition(col, val)}"
        if query not in unique_queries:
            all_queries[n].append(query)
            unique_queries.add(query)

    if len(all_queries[n]) < queries_per_n:
        print(
            f"  Warning: only found {len(all_queries[n])} zero-result queries for n={n}"
        )

    return all_queries


def generate_empty_queries_for_all_n(
    dataset_name,
    max_m,
    queries_per_n,
    table_name="db",
    seed=0,
):
    set_seed(seed)

    # m = 1 is a special case
    queries = generate_empty_queries_single_column(
        dataset_path_1=f"data/interim/{dataset_name}_first_1.csv",
        dataset_path_2=f"data/interim/{dataset_name}_first_{max_m}.csv",
        queries_per_n=queries_per_n,
        table_name=table_name,
    )
    with open(
        f"data/queries/empty_{dataset_name}_first_1.json",
        "w",
    ) as f:
        json.dump(queries, f)

    for m in range(2, max_m + 1):
        dataset_path = f"data/interim/{dataset_name}_first_{m}.csv"
        queries = generate_empty_queries(
            dataset_path, queries_per_n, table_name=table_name
        )
        with open(
            f"data/queries/empty_{dataset_name}_first_{m}.json",
            "w",
        ) as f:
            json.dump(queries, f)


if __name__ == "__main__":
    cnf = load_config()

    # generate_queries_for_all_n(
    #     dataset_name="dblp", max_m=4, queries_per_n=10, eq="nonequality", seed=cnf.get("seed", 0)
    # )

    # queries = generate_queries(dataset_name, queries_per_n=2)
    # with open("queries.txt", "w") as f:
    #     f.write("\n".join(queries))

    generate_empty_queries_for_all_n(
        dataset_name="movie",
        max_m=15,
        queries_per_n=10,
        table_name="db",
        seed=cnf.get("seed", 0),
    )
