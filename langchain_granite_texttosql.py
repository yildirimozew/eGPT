import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate

df = pd.read_csv("./data/customer_maintenance_202410242350.csv", on_bad_lines="warn")
#print(df.shape)
#print(df.columns.tolist())

engine = create_engine("sqlite:///text_to_sql.db")
db = SQLDatabase(engine=engine)

#print("Usable table names: ", db.get_usable_table_names())
#print(db.run("SELECT * FROM text_to_sql WHERE invoice_id = 1685022319400;"))

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db) #bunda bir sıkıntı var
test_response = write_query.invoke({"question": "How many invoices are there in the database"})
print(test_response)
chain = write_query | execute_query


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query} 
SQL Result: {result}
Answer: """
)

def print_and_return(string):
    print(string)
    return string

answer = answer_prompt| print_and_return | llm | StrOutputParser()

def split_query(query):
    query_query = query["query"]
    print("Query query:",query_query)
    splitted_query = query_query.split("SQLQuery:")
    print("Splitted query:",splitted_query[-1])
    return itemgetter(1)(splitted_query[-1])

chain = ( 
    RunnablePassthrough.assign(query=write_query).assign(
        result= split_query | execute_query
    )
    | answer 
)

response = chain.invoke({"question": "How many invoices are there in the database"})
print(response)

