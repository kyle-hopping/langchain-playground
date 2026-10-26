# LangChain Playground

A sandbox for learning [LangChain](https://docs.langchain.com/) (v1.x) through small lessons. Each file introduces a concept and builds on the previous one.

Models run on [Groq's](https://console.groq.com/) free tier, so the lessons cost nothing to run.

## Setup

```bash
# clone and enter the repo
git clone https://github.com/kyle-hopping/langchain-playground.git
cd langchain-playground

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

The install takes a while because `sentence-transformers` (used for the RAG lessons) pulls in PyTorch. The first run of lesson 8 also downloads the embedding model (~90 MB).

Create a `.env` file in the project root with your Groq API key (get one free at console.groq.com, no credit card needed):

```
GROQ_API_KEY=gsk_...
```

`.env` is listed in `.gitignore`, so the key never gets committed.

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Loads `.env` and creates the shared chat model. Change the model or provider here and every lesson picks it up. |
| `01_basics.py` | Send a prompt to a model and print the reply |
| `02_messages.py` | System, human, and AI messages; how conversation history works |
| `03_streaming.py` | Stream the response token by token |
| `04_chains.py` | Prompt templates and chaining components with `\|` |
| `05_structured.py` | Get validated Pydantic objects back instead of free text |
| `06_agent.py` | Give the model tools and let an agent decide when to call them |
| `07_memory.py` | Persist conversation history across turns with a checkpointer |
| `08_embeddings.py` | Turn text into vectors and search a vector store by meaning |
| `09_rag_chain.py` | RAG as a chain: chunk the handbook, retrieve relevant pieces, answer from them |
| `10_rag_agent.py` | RAG as an agent: retrieval becomes a tool the agent calls only when it needs to |
| `docs/handbook.txt` | A made-up employee handbook for "Nimbus Coffee Co.", used as the knowledge source for lessons 9 and 10 |

Run any lesson with the venv active:

```bash
python 01_basics.py
```

## Key takeaways

**1. Basics** – `init_chat_model` is a provider-agnostic factory function. It returns a chat model object that calls the provider's API when you run `.invoke()`.

**2. Messages** – Models are stateless. A "conversation" is just the full message list sent on every call. The system message sets behaviour.

**3. Streaming** – `.stream()` returns a generator. The server sends tokens as they're produced (Server-Sent Events), and each loop iteration receives an `AIMessageChunk`. `end=""` joins the pieces and `flush=True` prints them immediately. Chunks can be added together to rebuild the full message.

**4. Chains** – `prompt | model | parser` pipes components together, and every piece shares the same `.invoke()` / `.stream()` / `.batch()` interface. `StrOutputParser` turns the model's `AIMessage` into a plain string, which keeps chains composable.

**5. Structured output** – `with_structured_output(MyModel)` returns a validated Pydantic object. This is the bridge from chatbot to LLM-inside-real-software.

**6. Agents and tools** – The `@tool` decorator exposes a Python function to the model; the docstring and type hints are what the model sees. `create_agent` runs a loop: the model calls a tool, the result goes back, and it repeats until it can answer. The agent's input and output are its state, `{"messages": [...]}`, and the list grows with every tool call and result.

**7. Memory** – A checkpointer (`InMemorySaver`) stores history per `thread_id`, so each call only needs the new message. A new thread ID starts a fresh conversation.

**8. Embeddings** – An embedding model turns text into a fixed-length vector (384 numbers for `all-MiniLM-L6-v2`) that captures meaning. A vector store keeps those vectors and returns the closest matches to a query, so "time off" finds "vacation days" with no shared keywords. The embedding model runs locally through Hugging Face, so it needs no API key (the first run downloads it).

**9. RAG chain** – Retrieval-augmented generation in four steps: split the document into overlapping chunks (`RecursiveCharacterTextSplitter`), embed them into a vector store, retrieve the top `k` chunks for each question, and paste them into the prompt as context. The system prompt tells the model to answer only from that context and to say "I don't know" otherwise, which is how it handles questions the handbook doesn't cover.

**10. RAG agent** – Wrapping the retriever in a `@tool` lets the agent decide when to search. It skips retrieval for small talk, and it can search more than once for a question with several parts. The tool's docstring tells the model what the handbook covers.

## Small things I learnt along the way

- **Run with the venv's Python.** A `ModuleNotFoundError` usually means the script ran outside the venv. Activate it and check with `which python3`, and select the `.venv` interpreter in VS Code.
- **Models only know their training data.** Ask about something recent and the model may guess or hallucinate. RAG (lessons 8–10) is one fix.
- **Free-tier rate limits.** A `429` error means the per-minute limit was hit. Wait a few seconds and rerun. Or limit tokens!

## Switching providers

Only `config.py` needs to change. For example, to run fully offline with [Ollama](https://ollama.com/):

1. `brew install ollama` and `ollama pull qwen3:8b`
2. `pip install langchain-ollama`
3. In `config.py`, set the model to `init_chat_model("qwen3:8b", model_provider="ollama")`

## Lessons I will later add

- [x] RAG: document loading, chunking, embeddings, and a vector store
- [ ] LangGraph: custom control flow, branching, human-in-the-loop
- [ ] LangSmith: tracing and debugging runs
- [ ] Evaluation: test sets and scoring outputs
- [ ] MCP: connecting agents to external tool servers
