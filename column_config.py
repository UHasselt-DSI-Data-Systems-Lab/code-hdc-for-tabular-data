null_val = "nan"

proj_columns_datasets = {
    "movie": [
        ("id", "id"),
        ("actor_1", f"coalesce(actor_1, '{null_val}')"),
        ("actor_2", f"coalesce(actor_2, '{null_val}')"),
        ("actor_3", f"coalesce(actor_3, '{null_val}')"),
        ("color", f"coalesce(color, '{null_val}')"),
        ("content_rating", f"coalesce(content_rating, '{null_val}')"),
        ("director", f"coalesce(director, '{null_val}')"),
        ("genres", f"coalesce(genres, '{null_val}')"),
        ("original_language", f"coalesce(original_language, '{null_val}')"),
        ("production_companies", f"coalesce(production_companies, '{null_val}')"),
        ("production_countries", f"coalesce(production_countries, '{null_val}')"),
        ("release_date_rounded", f"coalesce(cast(release_date_rounded as varchar), '{null_val}')"),
        ("status", f"coalesce(status, '{null_val}')"),
        ("title", f"coalesce(title, '{null_val}')"),
        ("vote_average", f"coalesce(cast(vote_average as varchar), '{null_val}')"),
        ("year", f"coalesce(cast(year as varchar), '{null_val}')")
    ],
    "dblp": [
        ("id", "id"),
        ("authors", f"coalesce(authors, '{null_val}')"),
        ("title", f"coalesce(title, '{null_val}')"),
        ("venue", f"coalesce(venue, '{null_val}')"),
        ("year", f"coalesce(cast(year as varchar), '{null_val}')")
    ],
    "word2vec": [("id", "id")] + [
        (f"col{i}", f"coalesce(col{i}, '{null_val}')") for i in range(1,16)
    ],
    "fasttext": [("id", "id")] + [
        (f"col{i}", f"coalesce(col{i}, '{null_val}')") for i in range(1,16)
    ]
}