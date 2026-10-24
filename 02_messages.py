from config import model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# Create a predefined conversation (history) with system, human, and AI messages
messages = [
    SystemMessage("You are a hiring manager for a Agent Platform Researcher role."),
    HumanMessage("What's a good AI project to stand out as a candidate?"),
    AIMessage("""Build an agent that can autonomously research and summarize the
              latest information in AI and agent platforms."""),
    HumanMessage("""Give me a breakdown on the components of the project and how
                 to implement it using bullet points."""),
]

print(model.invoke(messages).content)
