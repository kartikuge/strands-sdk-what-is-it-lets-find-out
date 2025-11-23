from pydantic import BaseModel
from typing import List, Optional

class SummaryState(BaseModel):
    text: str
    bullets: Optional[List[str]] = None
    final_summary: Optional[str] = None

    step: str = "plan"  
    # allowed: "plan", "extract", "summarize", "done"