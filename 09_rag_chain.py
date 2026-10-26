from config import model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the context below. "
     "If the answer isn't in the context, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

for q in [
    "How many vacation days can I carry over?",
    "What do I do if the espresso machine shows E7?",
    "Where is the Latte Art Academy held?",
    "What's the parental leave policy?",   # not in the handbook
]:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q)}\n")