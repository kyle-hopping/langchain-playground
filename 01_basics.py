from config import model

# Talking to a model!
response = model.invoke("You are a teacher. Explain what LangChain is in one sentence.")

# This holds the content, metadata, and other information about the response.                        
print(response)

# Access the content of the response
print(response.content)