import pandas as pd
import json, re, ast

INPUT = "repl/data/via.csv"
OUTPUT = "repl/data/pairs.csv"

def normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))

def strip_fences(s: str) -> str:
    s = re.sub(r"^\s*```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s.strip())
    return s

def extract_braced(s: str):
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    core = s[i:j+1]
    # remove trailing commas before } or ]
    core = re.sub(r",(\s*[}\]])", r"\1", core)
    return core

def robust_parse(cell: str):
    if not isinstance(cell, str):
        return None
    s = normalize_quotes(cell)
    s = strip_fences(s)
    core = extract_braced(s)
    if core is None:
        return None

    # Try JSON
    try:
        return json.loads(core)
    except Exception:
        pass

    # Try python-literal dict/list (single quotes etc.)
    try:
        return ast.literal_eval(core)
    except Exception:
        return None

def main():
    df = pd.read_csv(INPUT)

    parsed = df["generation_prompt"].apply(robust_parse)
    failed = int(parsed.isnull().sum())
    print("Total rows:", len(df))
    print("Failed parses:", failed)

    df_ok = df[parsed.notnull()].copy()
    df_ok["parsed"] = parsed[parsed.notnull()].values

    df_ok["action"]  = df_ok["parsed"].apply(lambda x: x.get("Human Action", ""))
    df_ok["attrs"]   = df_ok["parsed"].apply(lambda x: x.get("Feature Attributions", []))
    df_ok["explain"] = df_ok["parsed"].apply(lambda x: x.get("Natural Language Explanation", ""))

    keep = df_ok[["country","topic","value","polarity","action","attrs","explain"]].copy()

    pos = keep[keep["polarity"]=="positive"].rename(columns={
        "action":"action_pos","attrs":"attrs_pos","explain":"explain_pos"
    }).drop(columns=["polarity"])

    neg = keep[keep["polarity"]=="negative"].rename(columns={
        "action":"action_neg","attrs":"attrs_neg","explain":"explain_neg"
    }).drop(columns=["polarity"])

    pairs = pos.merge(neg, on=["country","topic","value"], how="inner")

    print("pairs shape:", pairs.shape)
    print("\nSample:\n", pairs.head(2).to_string(index=False))

    pairs.to_csv(OUTPUT, index=False)
    print("\nWrote:", OUTPUT)

if __name__ == "__main__":
    main()
