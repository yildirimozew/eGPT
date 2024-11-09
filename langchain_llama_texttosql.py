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
import glob
import re

csv_files = glob.glob("./data/*.csv")
engine = create_engine("sqlite:///text_to_sql.db")

for file in csv_files:
    print("Reading file: ", file)
    df = pd.read_csv(file, on_bad_lines="warn")
    table_name = file.split("/")[-1].split(".")[0]
    
    df.to_sql(table_name, con=engine, index=False, if_exists="replace")

db = SQLDatabase(engine=engine, sample_rows_in_table_info=2)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    num_ctx=15000,
)

template = '''Given an input question, create a syntactically correct SQLite query to run. Only return the query. Only return {top_k} results.
Make sure to only use the provided table and column names.
Here are the tables and example rows:

{table_info}.

Question: {input}'''
prompt = PromptTemplate.from_template(template)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db, prompt = prompt, k = 5)

answer_prompt = PromptTemplate.from_template(
"""Given the following user question, corresponding SQL query, and SQL result, answer the user question. Talk like you're giving your answer to the user who asked the question.
    Be sure to include the SQL query and result in your answer. Be formal and concise in your response. If an error occurred, return the error message.

Question: {question}
SQL Query: {query} 
SQL Result: {result}
Answer: """
)

def print_and_return(string):
    print(string)
    return string

answer = answer_prompt | llm | StrOutputParser()
#answer = answer_prompt | write_query | StrOutputParser()

def exec_query(query):
    query_query = query["query"]
    if(query_query == ""):
        return "Query generation error."

    #sql_pattern = r"SELECT\s.+?;|INSERT\s.+?;|UPDATE\s.+?;|DELETE\s.+?;"
    #matches = re.findall(sql_pattern, query_query, re.IGNORECASE | re.DOTALL)
    #seen = set()
    #matches = [x for x in matches if x not in seen and not seen.add(x)]
    #results = []
    #if matches:
    #    print("The regex matched query is: \n", matches)
    #    for match in matches:
    #        print("The SQL query is: \n", match)
    #        result = execute_query.invoke(match)
    #        print("The SQL execution result is: \n", result)
    #        results.append(result)
    #results = " ".join(results)
    result = execute_query.invoke(query_query)
    if(result == ""):
        result = "SQL execution returned no results."
    print("The result is: \n", result)
    return result

chain = ( 
    RunnablePassthrough.assign(query=write_query).assign(
        result = exec_query
    )
    | answer 
)

#response1 = chain.stream({"question": "How many invoices are there in the database"})
#print("The question was: How many invoices are there in the database, the response is: \n ", response1)
#response2 = chain.stream({"question": "Get me the entry with invoice id 1685022319400"})
#print("The question was: Get me the entry with invoice id 1685022319400, the response is: \n ", response2)
#response3 = chain.stream({"question": "Get me the id's of invoices where amount spent is over 20000. Do not limit the number of results. End the query with a semicolon."})
#print("The question was: Get me the id's of users who have spent over 20000, the response is: \n ", response3)
#for chunk in response3:
#    print(chunk, end="", flush=True)

#Return the serial numbers of terminals that have timestamps in 2023
#Return the pdf files from documents that have been created in May
#list the customer id's of top-3 customers who spent most money in august 2023 
#Get me the entry with invoice id 1685022319400
#what is the highest spending customer in the database based on total invoice amounts
#item isimleriyle invoice bağlantısı olmadığı için "what is the most frequently sold item in 2023?" çalışmıyor
while(True):
    input_query = input("\nUser:")
    if input_query == "exit":
        break
    response = chain.stream({"question": input_query})
    for chunk in response:
        print(chunk, end="", flush=True)