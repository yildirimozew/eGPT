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

model = ChatOllama(
    model="llama3.2:latest",
)

#this loads ceng491 syllabus, however it cannot parse tables yet, so only ask questions about the text
loader = PyPDFLoader("./data/syllabus.pdf")
pages = []
for page in loader.lazy_load():
    pages.append(page)
data = pages

#this loads subtitles from a statistics video
"""loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=cy8r7WSuT1I", add_video_info=True
)
data = loader.load()"""

#this loads a text file
"""loader = TextLoader("./data/metu.txt")
data = loader.load()"""


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#summarization chain
"""prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

question = "How many masters programs does METU offer?"

docs = vectorstore.similarity_search(question)

print(chain.invoke(docs))"""

#q&a chain

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "How many phd programs does METU offer?"

docs = vectorstore.similarity_search(question)

# Run
print(chain.invoke({"context": docs, "question": question}))
"""

#chat chain with history

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

question = "What is the name of the lake in ODTÜ forest?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "How big is it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])

while(True):
    user_input = input("User: ")
    #hard to implement streaming on chains 
    ai_msg = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    print(ai_msg["answer"])
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ai_msg["answer"]))

    if "exit" == user_input:
        break