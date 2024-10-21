from r2r import R2RClient
import glob

client = R2RClient("http://localhost:7272")
# when using auth, do client.login(...)

print("Please state which chatbot you would like to use:\n1 or 2")
chatbot = input()

document_ids = [t["id"] for t in client.documents_overview()["results"]]

def delete_ingestions(document_ids):
    for document_id in document_ids:
        ingest_deletion_response = client.delete({
                "document_id": {"$eq": document_id}
            }
        )

def new_ingest(file_paths):
    ingest_response = client.ingest_files(
        file_paths=file_paths,
    )
    document_ids = [d["document_id"] for d in ingest_response["results"]]
    return document_ids

file_paths = glob.glob("./ingestion_docs/chatbot" + chatbot + "/*.txt")
print(file_paths)
delete_ingestions(document_ids)
new_ingest(file_paths)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
]

while(True):
    client_input = input()
    if client_input == "exit":
        break
    messages.append({"role": "user", "content": client_input})
    print(messages)
    response = client.agent(
        messages=messages,
        vector_search_settings={"use_hybrid_search": True},
        kg_search_settings={"use_kg_search": False},
        rag_generation_config={
            "stream": False,
            "model": "ollama/llama3",
            "temperature": 0.7,
            "max_tokens": 10
        }
    )
    """full_response = ""
    for chunk in response:
        full_response += chunk
        print(chunk, end="", flush=True) """
    messages.append({"role": "assistant", "content": response})
    print(messages)
    print(response)
