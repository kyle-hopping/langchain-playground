from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 1. An embedding is a list of numbers that captures meaning
vector = embeddings.embed_query("How much time off do I get?")
print(f"Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

# 2. Store some documents as vectors
docs = [
    Document("Employees earn 1.5 vacation days per month."),
    Document("The espresso machine needs descaling weekly."),
    Document("Staff must wear non-slip shoes."),
    Document("Remote workers get a home office stipend."),
]
store = InMemoryVectorStore.from_documents(docs, embeddings)

# 3. Search by meaning, not keywords
query = "How much time off do I get?"
for doc, score in store.similarity_search_with_score(query, k=2):
    print(f"{score:.3f}  {doc.page_content}")