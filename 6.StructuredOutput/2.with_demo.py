from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Malik Sakib"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5.0,
                        description="A Decimal value respresentationn the cpga value of the student")


new_student = {"age": '32', 'email': 'abc@gmail.com'}
student = Student(**new_student)

print(student)
print(type(student))
print(student.name)
print(student.age)
print(student.email)
print(student.cgpa)
print(student.dict())
print(student.model_dump_json())
