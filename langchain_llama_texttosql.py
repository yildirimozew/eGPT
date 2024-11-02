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

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db) 
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

answer = answer_prompt | llm | StrOutputParser()

def split_query(query):
    query_query = query["query"]
    splitted_query = query_query.split("SQLQuery:")
    print("The SQL query is: \n", splitted_query[-1])
    result = execute_query.invoke(splitted_query[-1])
    print("The SQL execution result is: \n", result)
    return result

chain = ( 
    RunnablePassthrough.assign(query=write_query).assign(
        result= split_query
    )
    | answer 
)

response1 = chain.invoke({"question": "How many invoices are there in the database"})
print("The question was: How many invoices are there in the database, the response is: \n ", response1)
response2 = chain.invoke({"question": "Get me the entry with invoice id 1685022319400"})
print("The question was: Get me the entry with invoice id 1685022319400, the response is: \n ", response2)
response3 = chain.invoke({"question": "Get me the id's of invoices where amount spent is over 20000. Do not limit the number of results"})
print("The question was: Get me the id's of users who have spent over 20000, the response is: \n ", response3)

