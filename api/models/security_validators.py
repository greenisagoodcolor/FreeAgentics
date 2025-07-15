from pydantic import BaseModel, validator


class SecureInputModel(BaseModel):
    """Base model with common security validations."""

    @validator("*", pre=True)
    def no_sql_injection(cls, v):
        if isinstance(v, str):
            # Check for common SQL injection patterns
            sql_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "UNION", "SELECT"]
            if any(keyword in v.upper() for keyword in sql_keywords):
                # Allow these keywords only in specific fields
                raise ValueError("Invalid input detected")
        return v

    @validator("*", pre=True)
    def no_xss(cls, v):
        if isinstance(v, str):
            # Basic XSS prevention
            if "<script" in v.lower() or "javascript:" in v.lower():
                raise ValueError("Invalid input detected")
        return v
