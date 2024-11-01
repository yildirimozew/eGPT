import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain.chains import create_sql_query_chain
from langchain_huggingface import HuggingFacePipeline

df = pd.read_csv("./data/customer_maintenance_202410242350.csv", on_bad_lines="warn")
print(df.shape)
print(df.columns.tolist())

engine = create_engine("sqlite:///text_to_sql.db")
#df.to_sql("text_to_sql", engine, index=False)
db = SQLDatabase(engine=engine)

print(db.dialect)
print("Usable table names: ", db.get_usable_table_names())
print(db.run("SELECT * FROM text_to_sql WHERE invoice_id = 1685022319400;"))

llm = ChatOllama(
    model="granite-code:3b",
)

chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "Show me the entry with invoice_id 1685022319400"})
db.run(response)

#llama is really bad at generating sql. 