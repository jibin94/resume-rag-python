from pydantic import BaseModel
from typing import List, Optional

class Experience(BaseModel):
    company: str
    position: str
    duration: str
    responsibilities: List[str]

class Education(BaseModel):
    institution: str
    degree: str
    year: str

class ResumeSchema(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    skills: List[str]
    experience: List[Experience]
    education: List[Education]
    summary: Optional[str] = None