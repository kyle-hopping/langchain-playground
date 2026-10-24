from config import model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain {topic} to a {level} audience in 4 sentences."),
    ("human", "{question}"),
])

# The | operator pipes components together. It reads as: fill the template, send
# it to the model, extract the string. Every piece shares the
# same .invoke() / .stream() / .batch() interface.

# StrOutputParser() is a convenience that converts the model’s output into a plain string
# Without it, the type is <class 'langchain_core.messages.ai.AIMessage'> and you
# would need to store chain.invoke() in a variable and then access with .content!
chain = prompt | model | StrOutputParser()

# chain = prompt | model
# response = chain.invoke({
#     "topic": "apache kafka",
#     "level": "new-grad",
#     "question": "what are the key components for streaming?",
# })
# print(response.content)

print(chain.invoke({
    "topic": "apache kafka",
    "level": "new-grad",
    "question": "what are the key components for streaming?",
}))

