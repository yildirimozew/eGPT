import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
import glob
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import translator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
import langdetect

"""what we need our agent to do:
1- decide on whether to query the database
2- decide on whether to give reference to the page the info was retrieved from
3- decide on whether to translate
4- decide on whether to memorize??
other than that we need a session key to retrieve chats"""

#TODO operate multiple llm instances corresponding to multiple users at the same time

#Here will come the document loaders
csv_files = glob.glob("./data/*.csv")
engine = create_engine("sqlite:///text_to_sql.db")

for file in csv_files:
    print("Reading file: ", file)
    df = pd.read_csv(file, on_bad_lines="warn")
    table_name = file.split("/")[-1].split(".")[0]
    
    df.to_sql(table_name, con=engine, index=False, if_exists="replace")

db = SQLDatabase(engine=engine, sample_rows_in_table_info=1)

llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
    num_ctx=8000,
)

execute_query_tool = QuerySQLDataBaseTool(db=db)

#agent tools and langgraph implementation

class State(TypedDict):
    messages: Annotated[list, add_messages]
    is_turkish: bool

graph_builder = StateGraph(State)

def need_query_database(user_prompt: str) -> str: 
    ###TODO fix this, maybe add keyword check like "database" or "get, fetch"
    """Decides whether to query the database or not"""
    template = PromptTemplate.from_template("""Given a user query, tables in the database, decide whether to query the database to answer question or not.
    Answer ONLY one word: Yes/No
    Question: {input}
    Tables: {table_info}
    Answer: Yes/No""")
    prompt = template.invoke({"input": user_prompt["messages"][-1].content, "table_info": db.get_table_info()})
    answer = llm.invoke(prompt)
    return answer.content

def need_translation(user_prompt: str) -> str:
    """Decides whether to translate the prompt to English or not"""
    if langdetect.detect(user_prompt["messages"][-1].content) != "en":
        return "Yes"
    else:
        return "No"

def translate_prompt(state: State) -> str:
    "Translates the given prompt to English"
    translated = translator.translate_to_en_finetuned(state["messages"][-1].content)
    state_msg = HumanMessage(content=translated)
    return {"messages": [RemoveMessage(id=state["messages"][-1].id), state_msg], "is_turkish": True}
    #TODO we need to make the graph bigger and set is_turkish = false in a seperate node

def query_database(state: State) -> str: 
    """Use this when you need to query the database. Takes natural language user request as input and outputs database results"""
    #TODO only link relevant columns here
    template = PromptTemplate.from_template('''Given an input question, create a syntactically correct SQLite query to run. Only return the query. The sql query should only return {top_k} results.
    Make sure to only use the provided table and column names.
    Here are the tables and example rows:

    {table_info}.

    Question: {input}''')
    write_query = create_sql_query_chain(llm, db, prompt = template, k = 5)
    SQL_query = write_query.invoke({"question": state["messages"][-1].content})
    result = execute_query_tool.invoke(SQL_query)
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question. Talk like you're giving your answer to the user who asked the question.
        Be sure to include the SQL query and result in your answer. Be formal and concise in your response. If an error occurred, return the error message.

    Question: {question}
    SQL Query: {query} 
    SQL Result: {result}
    Answer: """
    )

    prompt = answer_prompt.invoke({"question": state["messages"][-1].content, "query": SQL_query, "result": result})
    answer = llm.invoke(prompt)
    if state["is_turkish"] == True:
        translated = translator.translate_to_tr(answer.content)
        print(translated)
    return {"messages": answer}

def chatbot(state: State):
    answer = llm.invoke(state["messages"])
    if state["is_turkish"] == True:
        translated = translator.translate_to_tr(answer.content)
        print(translated)
    return {"messages": answer}
    
def empty_node(state: State):
    return state

#nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("translate_prompt", translate_prompt)
graph_builder.add_node("empty_node", empty_node)
graph_builder.add_node("query_database", query_database)

#edges
graph_builder.add_edge("translate_prompt", "empty_node")
graph_builder.add_edge("query_database", END)
graph_builder.add_edge("chatbot", END)

#conditional edges
graph_builder.add_conditional_edges(START, need_translation, {"Yes": "translate_prompt", "No": "empty_node"})
graph_builder.add_conditional_edges("empty_node", need_query_database, {"Yes": "query_database", "No": "chatbot"})

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    #stream_graph_updates(user_input)
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"messages": [("user", user_input)], "is_turkish": False}, config, stream_mode="values")["messages"][-1].content #TODO bunu humanmessage'a dönüştür
    is_turkish = graph.get_state(config).values["is_turkish"]
    if is_turkish == False: 
        print(result)
