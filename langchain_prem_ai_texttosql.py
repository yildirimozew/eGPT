from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_ollama import ChatOllama

#pipe = pipeline("text-generation", model="premai-io/prem-1B-SQL", device=0)

llm = ChatOllama(
    model="llama3.1",
)


engine = create_engine("sqlite:///text_to_sql.db")
database = SQLDatabase(engine=engine)
write_query = create_sql_query_chain(llm, database)
execute_query = QuerySQLDataBaseTool(db=database)
tables = database.get_usable_table_names()
table_columns = {}
for table in tables:
    table = Table(table, MetaData(), autoload_with=engine)
    columns = [column.name for column in table.columns]
    table_columns[table.name] = columns

table_columns = str(table_columns)



query = "How many invoices are there?"
prompt = f"Convert this natural language sentence to SQL: {query}. You can use these SQL tables: {tables}. You can use these SQL columns: {table_columns}. Only output a SQL query without braces."
write_query_response = write_query.invoke({"question": f"Convert this natural language sentence to SQL: {query}"})
print(write_query_response)
#generated_sql = llm.invoke(prompt).content
#print(generated_sql)
#generated_sql = generated_sql.replace(f"Do not include anything other than one single SQL query in your answer. Do not write extra natural language sentences. Convert this natural language query to SQL: {query}. You can use these tables: {tables}.", "").strip()
result = execute_query.invoke(write_query_response.split("SQLQuery: ")[-1])
print(result)