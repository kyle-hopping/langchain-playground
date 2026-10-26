from config import model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_retriever(path="docs/handbook.txt", k=3):
    with open(path) as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([text], metadatas=[{"source": path}])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = InMemoryVectorStore.from_documents(chunks, embeddings)

    return store.as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = build_retriever()

@tool
def search_handbook(query: str) -> str:
    """Search the Nimbus Coffee employee handbook for policies on hours,
    vacation, perks, dress code, equipment, and remote work."""
    return format_docs(retriever.invoke(query))

agent = create_agent(
    model=model,
    tools=[search_handbook],
    system_prompt=(
        "You help Nimbus Coffee employees. Use the handbook tool for any "
        "company policy question. Don't search for general small talk."
    ),
)

for q in [
    "Hi! How's it going?",
    "I just started a remote role. When do I get my stipend, and how much is it?",
]:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for msg in result["messages"]:
        msg.pretty_print()