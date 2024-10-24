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
    model="llama3.1:latest",
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

#q&a chain

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""


rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "Who are the course coordinators of ceng491?"

docs = vectorstore.similarity_search(question)

# Run
print(chain.invoke({"context": docs, "question": question}))
