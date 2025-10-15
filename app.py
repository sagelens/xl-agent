# app.py

import streamlit as st
import tempfile
import os
import pandas as pd
from pprint import pprint
import os, sys 
sys.path.append(os.getcwd())

from src.excel_parser import process_workbook
from src.database import AnalyticsDB
from src.agent import (
    NLQAgent,
    SQLSafetyValidator,
    EnhancedSchemaValidator,
    SemanticAmbiguityDetector,
    DisambiguationHandler,
    RuleBasedTranslator,
    QueryCache
)
from src.pipeline import generate_and_execute_query

@st.cache_resource
def setup_components(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        tmp.write(uploaded_file.getvalue())
        file_path = tmp.name
    try:
        safety_validator = SQLSafetyValidator(max_subquery_depth=3)
        tables, data_dictionary = process_workbook(file_path)
        if not tables:
            st.error("Could not find any tables in the uploaded Excel file.")
            return None

        db = AnalyticsDB(tables, data_dictionary)
        components = {
            "db": db, "agent": NLQAgent(num_samples=3),
            "safety_validator": safety_validator,
            "schema_validator": EnhancedSchemaValidator(db.get_schema_dict()),
            "ambiguity_detector": SemanticAmbiguityDetector(db.get_schema_dict(), data_dictionary),
            "disambiguator": DisambiguationHandler(),
            "rule_translator": RuleBasedTranslator(db.get_schema_dict(), data_dictionary),
            "query_cache": QueryCache(db),
            "workbook_name": uploaded_file.name
        }
        return components
    finally:
        os.unlink(file_path)

def streamlit_app():
    st.set_page_config(layout="wide", page_title="Excel Q&A Bot")
    st.title("Excel Q&A Bot")
    st.markdown("Upload an Excel file and ask questions in plain English.")

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

    if uploaded_file:
        components = setup_components(uploaded_file)
        if components:
            st.success(f"Successfully processed `{uploaded_file.name}`.")
            with st.expander("View Detected Database Schema"):
                st.code(components["db"].get_schema(), language='sql')
            
            user_question = st.text_input("Query:", placeholder="e.g., What is the total net revenue by category?")
            if user_question:
                with st.spinner("Generating SQL and fetching your answer..."):
                    result_report = generate_and_execute_query(question=user_question, **components)
                
                st.subheader("Results")
                if result_report["status"] == "Success" and result_report["result_rows"] is not None:
                    st.dataframe(pd.DataFrame(result_report["result_rows"]))
                else:
                    st.error("Failed to get an answer.")
                
                with st.expander("Show Trust & Explainability Report"):
                    st.markdown(f"**Confidence Score:** `{result_report.get('confidence', 0.0):.2f}`")
                    if result_report.get("generated_sql"):
                        st.markdown("**Generated SQL Query:**")
                        st.code(result_report["generated_sql"], language='sql')
                    if result_report.get("provenance"):
                        st.markdown("**Data Provenance (Source cells):**")
                        st.json(result_report["provenance"])
                    st.json(result_report.get("explanation", {}))

if __name__ == "__main__":
    # To run this app:
    # 1. Save all the files above.
    # 2. Make sure you have the required libraries:
    #    pip install streamlit pandas openpyxl duckdb ollama sqlglot
    # 3. Run from your terminal:
    #    streamlit run app.py
    streamlit_app()