from config import model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "user-1"}}

def chat(text):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    print("AI:", result["messages"][-1].content)

# Each chat is a round trip, so the agent will remember previous chats in the same thread.
# User input -> AI response -> etc.
chat("Hi, I'm Joshua and I'm learning LangChain.")
chat("What's my name?")
chat("What am I learning?")