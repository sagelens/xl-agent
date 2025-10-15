# agent.py

import re
import sqlglot
import hashlib
from difflib import SequenceMatcher, get_close_matches
from typing import Optional, Tuple
import os, sys 
import ollama
CWD = os.getcwd()
sys.path.append(os.getcwd())
from src.database import DataDictionary, AnalyticsDB
from src.model_runtime import get_duckdb_sql
from src.get_gguf import download_
MODEL_NAME = "duckdb-nsql"

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

USE_OLLAMA = config.get("USE_OLLAMA", False)

if not USE_OLLAMA:
    download_(repo_id="apurv0405/duckdb-nsql-gguf",
    filename="duckdb-nsql.gguf",
    local_dir=CWD + "/models/")


class EnhancedSchemaValidator:
    def __init__(self, schema_dict: dict):
        self.schema_dict = schema_dict
        self.all_columns = set()
        self.table_columns = {}
        for table, cols in schema_dict.items():
            self.table_columns[table] = set(cols)
            self.all_columns.update(cols)
    
    def validate_sql_schema(self, sql: str) -> Tuple[bool, str]:
        try:
            parsed = sqlglot.parse_one(sql, read='duckdb')
            query_aliases = set()
            for select_expression in parsed.find_all(sqlglot.exp.Select):
                for expression in select_expression.expressions:
                    if isinstance(expression, sqlglot.exp.Alias):
                        query_aliases.add(expression.alias)
            
            for col_node in parsed.find_all(sqlglot.exp.Column):
                col_name = col_node.name
                if col_name == '*': continue
                
                if col_name not in self.all_columns and col_name not in query_aliases:
                    suggestions = self._find_similar_columns(col_name)
                    return False, f"Column '{col_name}' does not exist in schema. Did you mean: {suggestions}?"
                
                if col_node.table:
                    table_name = col_node.table
                    if table_name in self.table_columns and col_name not in self.table_columns[table_name]:
                        if col_name in query_aliases: continue
                        correct_table = self._find_column_table(col_name)
                        return False, f"Column '{col_name}' exists in table '{correct_table}', not '{table_name}'"
            
            return True, ""
        except Exception as e:
            return True, ""

    def _find_similar_columns(self, col_name: str, max_results: int = 3) -> list:
        return get_close_matches(col_name, self.all_columns, n=max_results, cutoff=0.6)
    
    def _find_column_table(self, col_name: str) -> str:
        for table, cols in self.table_columns.items():
            if col_name in cols: return table
        return "unknown"

class QueryCache:
    def __init__(self, db: AnalyticsDB):
        self.db = db
        self.cache = {}
        self.materialized_views = {}
        self.hit_count = {}
    
    def get_or_execute(self, sql: str, threshold: int = 2):
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        
        if sql_hash in self.materialized_views:
            view_name = self.materialized_views[sql_hash]
            return self.db.query(f"SELECT * FROM {view_name}").df()
        
        if sql_hash in self.cache:
            self.hit_count[sql_hash] = self.hit_count.get(sql_hash, 0) + 1
            print(f"✓ Cache hit (count: {self.hit_count[sql_hash]})")
            if self.hit_count[sql_hash] >= threshold:
                self._materialize_view(sql, sql_hash)
            return self.cache[sql_hash]
        
        result = self.db.query(sql).df()
        self.cache[sql_hash] = result
        self.hit_count[sql_hash] = 1
        return result
    
    def _materialize_view(self, sql: str, sql_hash: str):
        view_name = f"mv_{sql_hash[:8]}"
        try:
            self.db.con.execute(f"CREATE TABLE {view_name} AS {sql}")
            self.materialized_views[sql_hash] = view_name
            print(f"✓ Materialized view created: {view_name}")
        except Exception as e:
            print(f"⚠️ Could not materialize view: {e}")

class RuleBasedTranslator:
    def __init__(self, schema_dict: dict[str, list], data_dictionary: DataDictionary):
        self.schema_dict = schema_dict
        self.data_dictionary = data_dictionary
        
    def _column_similarity(self, col_name: str, search_term: str) -> float:
        return SequenceMatcher(None, col_name.lower(), search_term.lower()).ratio()
    
    def _find_best_column(self, search_term: str, prefer_numeric: bool = True) -> Optional[Tuple[str, str]]:
        best_match = None
        best_score = 0.0
        for table_name, table_meta in self.data_dictionary.tables.items():
            for col in table_meta['columns']:
                col_name = col['name']
                if col_name == '_provenance': continue
                similarity = self._column_similarity(col_name, search_term)
                if prefer_numeric and col['type'] in ['int64', 'float64']:
                    similarity *= 1.2
                if col['unit'] and any(term in col['unit'].lower() for term in ['$', 'usd', 'dollar', 'currency']):
                    if any(term in search_term.lower() for term in ['revenue', 'spend', 'cost', 'price']):
                        similarity *= 1.1
                if similarity > best_score:
                    best_score = similarity
                    best_match = (table_name, col_name)
        return best_match if best_score > 0.4 else None
    
    def try_translate(self, question: str) -> Optional[str]:
        question_lower = question.lower().strip()
        sum_pattern = r'(?:sum|total)\s+(?:of\s+)?(\w+(?:\s+\w+)?)'
        match = re.search(sum_pattern, question_lower)
        if match:
            column_term = match.group(1).replace(' ', '_')
            result = self._find_best_column(column_term, prefer_numeric=True)
            if result:
                table_name, col_name = result
                return f"SELECT SUM(\"{col_name}\") FROM {table_name};"
        return None # Simplified for brevity, add other rules as needed

class SQLSafetyValidator:
    BANNED_KEYWORDS = {'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update', 'pragma', 'attach', 'detach', 'grant', 'revoke'}
    
    def __init__(self, max_subquery_depth: int = 2):
        self.max_subquery_depth = max_subquery_depth
    
    def validate(self, sql: str) -> Tuple[bool, str]:
        sql_lower = sql.lower().strip()
        first_keyword = sql_lower.split()[0] if sql_lower else ""
        if first_keyword in self.BANNED_KEYWORDS:
            return False, f"Unsafe operation: '{first_keyword}' statements are not allowed. Only SELECT queries are permitted."
        
        try:
            parsed = sqlglot.parse_one(sql, read='duckdb')
            if not isinstance(parsed, sqlglot.exp.Select):
                return False, "Only SELECT statements are allowed."
            depth = self._calculate_subquery_depth(parsed)
            if depth > self.max_subquery_depth:
                return False, f"Query too complex: subquery depth {depth} exceeds limit of {self.max_subquery_depth}."
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"
        
        return True, ""
    
    def _calculate_subquery_depth(self, node, depth: int = 0) -> int:
        max_depth = depth
        for child in node.find_all(sqlglot.exp.Select):
            if child != node:
                max_depth = max(max_depth, self._calculate_subquery_depth(child, depth + 1))
        return max_depth

class DisambiguationHandler:
    def handle_ambiguous_query(self, question: str, candidates: list, confidence: float, semantic_alternatives: list = None):
        if semantic_alternatives:
            print(f"\nDetected semantic ambiguity in query")
            # In a real UI, you would present these options for selection
            for i, alt in enumerate(semantic_alternatives, 1):
                print(f"{i}. {alt['rationale']} -> SQL: {alt['sql'][:100]}...")
            return {"type": "semantic_ambiguity", "options": semantic_alternatives}
        
        if confidence < 0.7:
            print(f"\nQuery interpretation confidence is low ({confidence:.2f})")
            # In a real UI, you would present these for selection
            for i, candidate in enumerate(candidates[:3], 1):
                print(f"{i}. SQL: {candidate['sql'][:100]}... (Score: {candidate['alignment_score']:.2f})")
            return {"type": "low_confidence", "options": candidates}
        return None

class SemanticAmbiguityDetector:
    def __init__(self, schema_dict: dict, data_dictionary: DataDictionary):
        self.schema_dict = schema_dict
    
    def detect_ambiguity(self, question: str, sql: str) -> Tuple[bool, list]:
        question_lower = question.lower()
        alternatives = []
        ambiguous_terms = {
            'revenue': ['gross_revenue', 'net_revenue'],
            'spend': ['planned_spend', 'actual_spend'],
        }
        for term, columns in ambiguous_terms.items():
            if term in question_lower:
                specific_term_found = any(col.replace('_', ' ') in question_lower for col in columns)
                if specific_term_found: continue
                
                available_cols = [col for col in columns if any(col in table_cols for table_cols in self.schema_dict.values())]
                if len(available_cols) > 1:
                    used_col = next((col for col in available_cols if col in sql), None)
                    if used_col:
                        for col in available_cols:
                            if col != used_col:
                                alternatives.append({
                                    'sql': sql.replace(used_col, col),
                                    'rationale': f"Interprets '{term}' as '{col}' instead of '{used_col}'"
                                })
        return len(alternatives) > 0, alternatives

class NLQAgent:
    def __init__(self, num_samples: int = 3, confidence_threshold: float = 0.7):
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold
        self.few_shot_store = {}
        self.last_prompt = None
    
    def translate_with_self_consistency(self, question: str, schema: str, schema_dict: dict, safety_validator: SQLSafetyValidator, schema_validator: EnhancedSchemaValidator) -> Tuple[str | None, float, list]:
        candidates = []
        for i in range(self.num_samples):
            sql = self.translate_to_sql(question, schema)
            if not sql: continue
            is_safe, safety_error = safety_validator.validate(sql)
            if not is_safe: continue
            
            is_valid_schema, schema_error = schema_validator.validate_sql_schema(sql)
            alignment_score = self._calculate_schema_alignment(sql, schema_dict) if is_valid_schema else 0.3
            
            candidates.append({"sql": sql, "alignment_score": alignment_score, "schema_error": schema_error})
        
        if not candidates: return None, 0.0, []
        
        candidates.sort(key=lambda x: x['alignment_score'], reverse=True)
        best_candidate = candidates[0]
        scores = [c['alignment_score'] for c in candidates]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        confidence = best_candidate['alignment_score'] / avg_score if avg_score > 0 else 0.0
        
        return best_candidate['sql'], confidence, candidates
    
    def _calculate_schema_alignment(self, sql: str, schema_dict: dict) -> float:
        try:
            parsed = sqlglot.parse_one(sql, read='duckdb')
            tables_used = {t.alias_or_name for t in parsed.find_all(sqlglot.exp.Table)}
            columns_used = {c.name for c in parsed.find_all(sqlglot.exp.Column) if c.name != '*'}
            valid_tables = sum(1 for t in tables_used if t in schema_dict)
            table_score = valid_tables / len(tables_used) if tables_used else 0
            valid_columns = sum(1 for col in columns_used for table in tables_used if table in schema_dict and col in schema_dict[table])
            column_score = valid_columns / len(columns_used) if columns_used else 1.0
            return 0.4 * table_score + 0.6 * column_score
        except Exception:
            return 0.0

    def add_few_shot_example(self, workbook: str, question: str, sql: str):
        if workbook not in self.few_shot_store:
            self.few_shot_store[workbook] = []
        self.few_shot_store[workbook].append((question, sql))

    def _create_prompt(self, question: str, schema: str) -> str:
        return f"### Instruction\nGenerate a valid DuckDB SQL query.\n### Database Schema\n{schema}\n### Question\n{question}\n### Answer (SQL ONLY)\n"

    def _create_correction_prompt(self, question: str, schema: str, failed_sql: str, error: str, suggestion: str) -> str:
        suggestion_block = f"\n### CRITICAL HINT\n{suggestion}\n" if suggestion else ""
        return f"### URGENT ERROR CORRECTION\nThe SQL query failed. You MUST fix it.\n### The Error\n{error}{suggestion_block}\n### Original Question\n{question}\n### Schema\n{schema}\n### Your FAILED Query\n```sql\n{failed_sql}\n```\nYour Corrected Query (SQL ONLY):"

    def correct_sql(self, question: str, schema: str, failed_sql: str, error: str, suggestion: str) -> str | None:
        prompt = self._create_correction_prompt(question, schema, failed_sql, str(error), suggestion)
        return self._generate_sql(prompt)

    def _generate_sql(self, prompt: str) -> str | None:
        self.last_prompt = prompt
        if USE_OLLAMA:
            response = ollama.generate(model=MODEL_NAME, prompt=prompt)
            sql_raw = response['response'].strip()
        else:
            response = get_duckdb_sql(prompt=prompt)
            sql_raw = response.strip()
        
        match = re.search(r"```(?:sql\n)?(.*?)```", sql_raw, re.DOTALL)
        sql_to_parse = match.group(1).strip() if match else sql_raw
        return sql_to_parse.split(';')[0].strip()

    def translate_to_sql(self, question: str, schema: str) -> str | None:
        prompt = self._create_prompt(question, schema)
        return self._generate_sql(prompt)