import json

import pandas as pd
import random
from itertools import combinations
import sqlparse


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
    dataset_name, max_m, queries_per_n, eq="nonequality", table_name="db"
):
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
    for l in ref_str:
        frags = l.strip(" ").split(" ")
        if len(frags) == 4:
            frags.pop(0)
        # selections.append((frags[0], frags[2].replace("'", ""), frags[1]))
        selections.append((frags[0], frags[2][1:-1], frags[1]))
    return {"select": project, "table": table, "selections": selections}


if __name__ == "__main__":
    generate_queries_for_all_n(
        dataset_name="movie", max_m=15, queries_per_n=10, eq="equality"
    )

    # queries = generate_queries(dataset_name, queries_per_n=2)
    # with open("queries.txt", "w") as f:
    #     f.write("\n".join(queries))
