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
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage


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

db = SQLDatabase(engine=engine, sample_rows_in_table_info=3)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    num_ctx=8000,
)

#this loads a text file
#loader = TextLoader("./data/metu.txt")
#data = loader.load()
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#all_splits = text_splitter.split_documents(data)
#local_embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
#vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
#retriever = vectorstore.as_retriever()

execute_query_tool = QuerySQLDataBaseTool(db=db)


def print_and_return(string: str) -> str:
    print(string)
    return string 

#tool initalization
@tool 
def query_database(user_request: str) -> str:
    """Use this when you need to query the database. Takes natural language user request as input and outputs database results"""
    #TODO only link relevant columns here
    template = '''Given an input question, create a syntactically correct SQLite query to run. Only return the query. The sql query should only return {top_k} results.
    Make sure to only use the provided table and column names.
    Here are the tables and example rows:

    {table_info}.

    Question: {input}'''
    prompt = PromptTemplate.from_template(template)
    write_query = create_sql_query_chain(llm, db, prompt = prompt, k = 5)
    SQL_query = write_query.invoke({"question": user_request})
    result = execute_query_tool.invoke(SQL_query)
    return result

@tool
def execute_query(SQLquery: str) -> str:
    """Execute SQL query created by create_query on the database. Returns the result """
    result = execute_query_tool.invoke(SQLquery)
    #can do error correction here
    return result

@tool
def create_query(query: str) -> str:
    """Use this when you need Create SQL Query from user request. Takes natural language user request as input and outputs an SQL query."""
    #TODO only link relevant columns here
    template = '''Given an input question, create a syntactically correct SQLite query to run. Only return the query. The sql query should only return {top_k} results.
    Make sure to only use the provided table and column names.
    Here are the tables and example rows:

    {table_info}.

    Question: {input}'''
    prompt = PromptTemplate.from_template(template)
    write_query = create_sql_query_chain(llm, db, prompt = prompt, k = 5)
    return write_query.invoke({"question": query})

tools = [query_database]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very powerful assistant. You can use several tools to help with your answers. Use the tools only when their description tells that they can help with the certain problem.
            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do, do not use any tool if it is not needed. 
            Action: the action to take, should be one of these: None, {tool_names}
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question""",
        ),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "tool_names": lambda x: " ".join(x.name for x in tools) #TODO fix this
    }
    | prompt | print_and_return
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

#list(agent_executor.stream({"input": "If Ayşe and Mehmet marry, what will be the relationship between Kemal and Mahmut?"}))
#bunu dene

#TODO add memory

query = "If Ayşe and Mehmet marry, what will be the relationship between Kemal and Mahmut?"

system_message = "You are a very powerful and helpful assistant."

memory = MemorySaver()
config = {"configurable": {"thread_id": "test-thread"}}

langgraph_agent_executor = create_react_agent(llm_with_tools, tools, state_modifier=system_message, checkpointer=memory)

#messages = langgraph_agent_executor.invoke({"messages": [("user", query)]}, config=config)
#{
#    "input": query,
#    "output": messages["messages"][-1].content,
#}

RAG_PROMPT = """
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

#print(messages)
#bunu da dene


#agent tools and langgraph implementation

@tool
def need_query_database(user_query: str) -> str:
    """Decides whether to query the database or not"""
    prompt = """Given a user query, tables in the database, decide whether to query the database to answer question or not.
    Question: {input}
    Tables: {table_info}
    Answer: Yes/No"""
    return llm.invoke({"input": user_query, "table_info": db.get_table_info(), "prompt": prompt})


@tool
def need_translation(user_query: str) -> str:
    """Decides whether to translate the query to English or not"""
    prompt = """Given a user query, decide whether to translate it or not. If the query is not in English, you should translate it to English.
    Question: {input}
    Answer: Yes/No"""
    return llm.invoke({"input": user_query, "prompt": prompt})

tools = [need_query_database, need_translation, query_database]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break