import langchain
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:latest")

for chunk in llm.stream("The first man on the moon was ..."):
    print(chunk, end="|", flush=True)