from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# loads OPENAI_API_KEY from .env
load_dotenv()

# Initialize the chat model with specified model name and provider
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq")