from config import model
from pydantic import BaseModel, Field

# Instead of parsing free text, you get a validated Python object back.
class JobPosting(BaseModel):
    title: str
    company: str
    skills: list[str] = Field(description="Technical skills mentioned")
    is_new_grad: bool

structured = model.with_structured_output(JobPosting)

text = """Google is hiring a Backend Engineer I (new grads welcome!).
You'll work with Python, Kafka, and PostgreSQL on our payments platform."""

result = structured.invoke(text)
print(result)
print("--------------------------------")
print(result.title)
print("--------------------------------")
print(result.company)
print("--------------------------------")
print(result.skills)
print("--------------------------------")