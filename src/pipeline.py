# pipeline.py

import sqlglot
import os, sys 
sys.path.append(os.getcwd())
from src.agent import NLQAgent, SQLSafetyValidator, EnhancedSchemaValidator, SemanticAmbiguityDetector, DisambiguationHandler, RuleBasedTranslator, QueryCache
from src.database import AnalyticsDB
from src.utils import validate_date_literals, extract_missing_column_suggestion, get_provenance_for_aggregate, map_nl_to_columns

def generate_and_execute_query(
    question: str,
    agent: NLQAgent,
    db: AnalyticsDB,
    safety_validator: SQLSafetyValidator,
    schema_validator: EnhancedSchemaValidator,
    ambiguity_detector: SemanticAmbiguityDetector,
    disambiguator: DisambiguationHandler,
    rule_translator: RuleBasedTranslator,
    query_cache: QueryCache,
    workbook_name: str
) -> dict:
    response = {
        "question": question, "status": "Failed", "result_rows": None,
        "generated_sql": None, "tables_used": [], "column_mapping": {},
        "provenance": None, "confidence": 0.0,
        "explanation": {"prompt": None, "top_k_candidates": [], "notes": []}
    }

    sql_query_text, confidence, candidates = agent.translate_with_self_consistency(
        question, db.get_schema(), db.get_schema_dict(), safety_validator, schema_validator
    )
    response["confidence"] = confidence
    response["explanation"]["prompt"] = agent.last_prompt
    response["explanation"]["top_k_candidates"] = candidates

    if sql_query_text:
        is_ambiguous, semantic_alts = ambiguity_detector.detect_ambiguity(question, sql_query_text)
        if (confidence < agent.confidence_threshold and candidates) or semantic_alts:
            disambiguator.handle_ambiguous_query(question, candidates, confidence, semantic_alts)

    if not sql_query_text:
        sql_query_text = rule_translator.try_translate(question)
        if sql_query_text: response["confidence"] = 0.95

    if not sql_query_text: return response

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            is_valid, err = validate_date_literals(sql_query_text)
            if not is_valid: raise ValueError(f"Date validation failed: {err}")
            
            is_safe, err = safety_validator.validate(sql_query_text)
            if not is_safe: raise ValueError(f"Safety validation failed: {err}")
            
            validated_sql = sqlglot.parse_one(sql_query_text, read='duckdb').sql(dialect='duckdb', pretty=True)
            response["generated_sql"] = validated_sql
            
            parsed_sql = sqlglot.parse_one(validated_sql, read='duckdb')
            response["tables_used"] = sorted(list({t.alias_or_name for t in parsed_sql.find_all(sqlglot.exp.Table)}))
            response["column_mapping"] = map_nl_to_columns(question, validated_sql)
            
            result_df = query_cache.get_or_execute(validated_sql)
            response["result_rows"] = result_df.to_dict(orient='records')
            response["status"] = "Success"
            
            if not result_df.empty:
                response["provenance"] = get_provenance_for_aggregate(validated_sql, db)
            
            agent.add_few_shot_example(workbook_name, question, validated_sql)
            return response

        except Exception as e:
            error_str = str(e)
            if attempt < max_attempts - 1:
                suggestion = extract_missing_column_suggestion(error_str, db.get_schema_dict())
                corrected_sql = agent.correct_sql(question, db.get_schema(), sql_query_text, error_str, suggestion)
                if corrected_sql:
                    sql_query_text = corrected_sql
                else: break
            else: break
    return response