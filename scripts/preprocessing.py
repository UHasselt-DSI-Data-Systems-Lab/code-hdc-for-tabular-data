import pandas as pd

def keep_first_n_columns(input_file, output_file, n):
    """
    Keep only the first n columns of a CSV file and save to a new file.
    Used for varying row length for embdi
    """
    # Load the dataframe
    df = pd.read_csv(input_file)
    
    # Slice the dataframe to keep only the first n columns
    # .iloc[:, :n] means "all rows, columns from start to n"
    df_reduced = df.iloc[:, :n]
    
    # Save to a new CSV
    df_reduced.to_csv(output_file, index=False)
    print(f"Success! Saved first {n} columns to {output_file}")


if __name__ == "__main__":
    for k in range(1, 16):
        keep_first_n_columns('data/raw/movie.csv', f'data/raw/movie_first_{k}.csv', k)
