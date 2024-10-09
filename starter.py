from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("umarigan/LLama-3-8B-Instruction-tr")
llm = AutoModelForCausalLM.from_pretrained("umarigan/LLama-3-8B-Instruction-tr")

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
#llm = Ollama(model="llama3.2", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)

chat_engine = index.as_chat_engine(chat_mode="react", llm=llm, verbose=True)
while(True):
    prompt = input("Enter your prompt: ")
    if(prompt == "exit"):
        break
    response = chat_engine.chat(prompt)
    print(response)