# backend/app/models/catalog.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Job(BaseModel):
    id: str
    title: str
    seniority: Optional[str] = None
    stack: List[str] = Field(default_factory=list)
    hard_requirements: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    description: Optional[str] = None

class Candidate(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    stack: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    education: Optional[str] = None
    resume_path: Optional[str] = None
    source: Optional[str] = None  # "applicants" | "prospects"
