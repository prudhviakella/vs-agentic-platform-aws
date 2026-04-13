from pydantic import BaseModel


class ChatRequest(BaseModel):
    message:   str
    thread_id: str
    domain:    str = "pharma"


class ResumeRequest(BaseModel):
    thread_id:   str
    user_answer: str
    domain:      str = "pharma"
