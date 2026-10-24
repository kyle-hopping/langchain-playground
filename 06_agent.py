from config import model
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Return the number of letters in a word."""
    return len(word)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

agent = create_agent(
    model=model,
    tools=[get_word_length, multiply],
    system_prompt="Use tools for any counting or math.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "How many letters in the phrase 'artificial intelligence', times 2?"}]
})

# The agent takes its whole state as input, and "messages" is a list because the
# agent keeps appending to it during the run (your question, its tool calls,
# the tool results, and the final answer). The output is that same state,
# so result["messages"][-1] is the final answer.
# print(result["messages"][-1].content)
for msg in result["messages"]:
    msg.pretty_print()