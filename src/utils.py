# utils.py

import re
import sqlglot
from difflib import SequenceMatcher
from datetime import datetime
import os, sys 
sys.path.append(os.getcwd())
from src.database import AnalyticsDB

def map_nl_to_columns(question: str, sql: str) -> dict:
    mapping = {}
    try:
        parsed = sqlglot.parse_one(sql, read='duckdb')
        columns_in_query = {c.alias_or_name for c in parsed.find_all(sqlglot.exp.Column)}
        stop_words = {'the', 'a', 'is', 'of', 'in', 'show', 'me', 'list', 'find', 'get', 'calculate'}
        question_words = set(re.findall(r'\b\w+\b', question.lower())) - stop_words
        
        for word in question_words:
            best_match_col = None
            best_score = 0.6
            for col in columns_in_query:
                score = SequenceMatcher(None, word, col.lower().replace('_', ' ')).ratio()
                if score > best_score:
                    best_score = score
                    best_match_col = col
            if best_match_col and best_match_col not in mapping.values():
                 mapping[word] = best_match_col
    except Exception:
        return {}
    return mapping

def validate_date_literals(sql: str) -> tuple[bool, str]:
    date_pattern = r"'(\d{4}-\d{2}-\d{2})'"
    dates = re.findall(date_pattern, sql)
    for date_str in dates:
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            _, month, day = map(int, date_str.split('-'))
            return False, f"Invalid date literal found: '{date_str}'. A date like {month}-{day} is not possible."
    return True, ""

def extract_missing_column_suggestion(error_msg: str, schema_dict: dict[str, list]) -> str:
    match = re.search(r'does not have a column named "(\w+)"', error_msg, re.IGNORECASE)
    if match:
        col_name = match.group(1).lower()
        for table, columns in schema_dict.items():
            if col_name in [c.lower() for c in columns]:
                return f"HINT: The column '{col_name}' exists in the table '{table}'."
    return ""

def get_provenance_for_aggregate(sql: str, db: AnalyticsDB) -> list[str] | None:
    try:
        parsed = sqlglot.parse_one(sql, read='duckdb')
        tables = list(parsed.find_all(sqlglot.exp.Table))
        if not tables: return None

        provenance_cols = [f'"{table.alias_or_name}"."_provenance"' for table in tables]
        parsed.select(*provenance_cols, append=False, copy=False)

        for clause in ['group', 'having', 'order', 'limit', 'qualify']:
            if parsed.args.get(clause):
                parsed.set(clause, None)

        provenance_sql = f"SELECT DISTINCT * FROM ({parsed.sql(dialect='duckdb')})"
        provenance_df = db.query(provenance_sql).df()
        
        all_provenance = set()
        for col_name in provenance_df.columns:
            if '_provenance' in col_name.lower():
                all_provenance.update(provenance_df[col_name].dropna().unique())
        return sorted(list(all_provenance)) if all_provenance else None
    except Exception as e:
        print(f"Could not generate provenance query: {e}")
        return None