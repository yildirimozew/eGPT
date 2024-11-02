from transformers import pipeline
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

pipe = pipeline("text-generation", model="premai-io/prem-1B-SQL", device=0)

engine = create_engine("sqlite:///text_to_sql.db")
database = SQLDatabase(engine=engine)
execute_query = QuerySQLDataBaseTool(db=database)
tables = database.get_usable_table_names()
table_columns = {}
for table in tables:
    table = Table(table, MetaData(), autoload_with=engine)
    columns = [column.name for column in table.columns]
    table_columns[table.name] = columns

table_columns = str(table_columns)


query = "How many invoices are there?"
prompt = f"Convert this natural language sentence to SQL: {query}. You can use these SQL tables: {tables}. You can use these SQL columns: {table_columns}."
generated_sql = pipe(prompt, max_length=200)[0]['generated_text']
generated_sql = generated_sql.replace(f"Convert this natural language sentence to SQL: {query}. You can use these SQL tables: {tables}. You can use these SQL columns: {table_columns}.", "").strip()
print("generated sql:", generated_sql)
result = execute_query.invoke(generated_sql)
print("SQL execution result:", result)