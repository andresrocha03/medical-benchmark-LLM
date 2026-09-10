"""Bounded-memory uniform sampling from local YAGO exports."""
import csv
import gzip
import json
from pathlib import Path
import random
import sqlite3
import tempfile


def reservoir(rows, count, seed):
    rng = random.Random(seed)
    sample = []
    seen = 0
    for row in rows:
        if not row:
            continue
        seen += 1
        if len(sample) < count:
            sample.append(row)
        else:
            index = rng.randrange(seen)
            if index < count:
                sample[index] = row
    if len(sample) < count:
        raise ValueError(f"Requested {count} YAGO rows, but only {seen} usable rows exist.")
    return sample, seen


def sample_yago(path, count=15000, seed=42, input_format="table", drop_columns=()):
    """Table: CSV/TSV header; JSONL: object per entity; triples: head/relation/tail.

    Triple exports are grouped on disk before sampling distinct entities, so
    15K means entity rows, not disconnected individual facts. Raw RDF/Turtle
    must first be exported into these formats with readable labels.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        if input_format == "jsonl":
            def rows():
                for line in source:
                    if line.strip():
                        record = json.loads(line)
                        yield [(key, value) for key, value in record.items()
                               if key not in drop_columns and value is not None]
            return reservoir(rows(), count, seed)
        delimiter = "\t" if ".tsv" in path.suffixes else ","
        reader = csv.DictReader(source, delimiter=delimiter)
        if input_format == "table":
            return reservoir(([(key, value) for key, value in record.items()
                               if key not in drop_columns and value not in (None, "")]
                              for record in reader), count, seed)
        if not {"head", "relation", "tail"} <= set(reader.fieldnames or ()):
            raise ValueError("Triples require a CSV/TSV header: head, relation, tail.")
        with tempfile.TemporaryDirectory(prefix="tarte-yago-") as directory:
            with sqlite3.connect(str(Path(directory) / "facts.db")) as db:
                db.execute("CREATE TABLE facts (head TEXT, relation TEXT, tail TEXT)")
                db.executemany("INSERT INTO facts VALUES (?, ?, ?)",
                               ((r["head"], r["relation"], r["tail"]) for r in reader
                                if r["head"] and r["relation"] and r["tail"]))
                db.execute("CREATE INDEX heads ON facts(head)")
                heads, seen = reservoir((r[0] for r in db.execute(
                    "SELECT DISTINCT head FROM facts ORDER BY head")), count, seed)
                sample = [list(db.execute("SELECT relation, tail FROM facts WHERE head=?",
                                          (head,))) for head in heads]
                return sample, seen
