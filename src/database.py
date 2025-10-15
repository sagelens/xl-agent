# database.py

import pandas as pd
from datetime import datetime
import duckdb

class DataDictionary:
    """A structured representation of the database schema and metadata."""
    
    UNIT_CONVERSIONS = {
        'kg': {'to_lbs': 2.20462, 'canonical': 'kg'},
        'lb': {'to_kg': 0.453592, 'canonical': 'kg'},
        'lbs': {'to_kg': 0.453592, 'canonical': 'kg'},
        't': {'to_kg': 1000, 'canonical': 'kg'},
        'ton': {'to_kg': 907.185, 'canonical': 'kg'},
        'g': {'to_kg': 0.001, 'canonical': 'kg'},
    }
    
    def __init__(self):
        self.tables = {}
        self.unit_normalizations = {}
    
    def add_table(self, table_name: str):
        self.tables[table_name] = {"columns": []}

    def add_column(self, table_name: str, col_name: str, col_type: str, unit: str | None, samples: list):
        canonical_unit = self._get_canonical_unit(unit)
        self.tables[table_name]["columns"].append({
            "name": col_name,
            "type": col_type,
            "unit": unit,
            "canonical_unit": canonical_unit,
            "samples": [s for s in samples if pd.notna(s)]
        })
        
        if unit and canonical_unit and unit.lower() != canonical_unit:
            self.unit_normalizations[f"{table_name}.{col_name}"] = {
                "original_unit": unit,
                "canonical_unit": canonical_unit,
                "conversion_factor": self._get_conversion_factor(unit, canonical_unit)
            }
    
    def _get_canonical_unit(self, unit: str | None) -> str | None:
        if not unit:
            return None
        unit_lower = unit.lower().strip()
        return self.UNIT_CONVERSIONS.get(unit_lower, {}).get('canonical')
    
    def _get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        from_unit_lower = from_unit.lower().strip()
        if from_unit_lower in self.UNIT_CONVERSIONS:
            conversions = self.UNIT_CONVERSIONS[from_unit_lower]
            for key, value in conversions.items():
                if 'to_' in key and to_unit in key:
                    return value
        return 1.0

    def generate_prompt(self) -> str:
        prompt_parts = [
            f"-- Current date is {datetime.now().strftime('%Y-%m-%d')}",
            "### Database Schema and Relationships"
        ]
        
        foreign_keys = [
            "-- h1_financials_1.department_id can be joined with department_info_0.dept_id.",
        ]

        business_glossary = [
            "-- 'team members' or 'employees' refers to the 'manager' in the department_info_0 table.",
            "-- 'spend' refers to columns like 'planned_spend' or 'actual_spend'.",
            "-- 'revenue' refers to columns like 'gross_revenue' or 'net_revenue'."
        ]

        for name, meta in self.tables.items():
            table_def = f"CREATE TABLE {name} (\n"
            col_lines = []
            for col in meta['columns']:
                unit_info = f" -- unit: {col['unit']}" if col['unit'] else ""
                canonical_info = f" (canonical: {col.get('canonical_unit')})" if col.get('canonical_unit') and col['canonical_unit'] != col['unit'] else ""
                sample_info = f" -- samples: {col['samples'][:3]}" if col['samples'] else ""
                col_lines.append(f"  \"{col['name']}\" {col['type']},{unit_info}{canonical_info}{sample_info}")
            
            table_def += "\n".join(col_lines)
            table_def += "\n);"
            prompt_parts.append(table_def)

        prompt_parts.append("\n### Business Glossary:")
        prompt_parts.extend(business_glossary)
        prompt_parts.append("\n### Querying Rules:")
        prompt_parts.append("-- 1. **CRITICAL AGGREGATION RULE**: When a question asks for a metric 'by' a category (e.g., 'spend by manager'), you MUST use a GROUP BY clause and an aggregate function (SUM, AVG, COUNT, etc.).")
        prompt_parts.append("-- 2. **CRITICAL JOIN RULE**: To answer the question, only join tables using the explicit relationships defined below.")
        prompt_parts.append("-- 3. **CRITICAL VALIDATION RULE**: Do NOT join columns with different meanings (e.g., never join a department ID with a shipment ID).")
        prompt_parts.append("-- 4. PROVENANCE: Every table has a `_provenance` column. Do NOT use it for calculations.")
        prompt_parts.append("\n### Table Relationships (Foreign Keys):")
        prompt_parts.extend(foreign_keys)
        prompt_parts.append("\n")
        
        if self.unit_normalizations:
            prompt_parts.append("\n-- UNIT NORMALIZATIONS: The following columns have been normalized:")
            for col_path, norm_info in self.unit_normalizations.items():
                prompt_parts.append(f"--   {col_path}: {norm_info['original_unit']} → {norm_info['canonical_unit']} (factor: {norm_info['conversion_factor']})")
        
        return "\n\n".join(prompt_parts)

class AnalyticsDB:
    def __init__(self, tables: dict[str, pd.DataFrame], dictionary: DataDictionary):
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.tables = tables
        self.dictionary = dictionary
        self.schema_dict = {}
        self._materialize()

    def _materialize(self):
        for name, df in self.tables.items():
            self.con.register(name, df)
            self.schema_dict[name] = df.columns.tolist()
        
        self.schema_prompt = self.dictionary.generate_prompt()

    def query(self, sql_query: str): return self.con.execute(sql_query)
    def get_schema(self): return self.schema_prompt
    def get_schema_dict(self): return self.schema_dict