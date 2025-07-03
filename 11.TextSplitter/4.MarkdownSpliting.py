from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
text = """
# 🎓 Student Class in Python

A simple Python class to represent a **Student** with properties and useful methods.

```python
class Student:
    def __init__(self, name, roll_no, grade):
        self.name = name
        self.roll_no = roll_no
        self.grade = grade
        self.marks = {}  # dictionary to store subject: marks

    def add_mark(self, subject, mark):        
        self.marks[subject] = mark

    def get_average(self):
        if not self.marks:
            return 0
        return sum(self.marks.values()) / len(self.marks)

    def get_details(self):
        avg = self.get_average()
        return (
            f"Name: {self.name}\n"
            f"Roll No: {self.roll_no}\n"
            f"Grade: {self.grade}\n"
            f"Average Marks: {avg:.2f}"
        )

    def has_passed(self):
        return all(mark >= 40 for mark in self.marks.values())

# Example usage:
student1 = Student("Sakib Malik", 101, "10th")
student1.add_mark("Math", 85)
student1.add_mark("Science", 78)
student1.add_mark("English", 62)

print(student1.get_details())
print("Passed:", student1.has_passed())
```

## 🔍 Features

- `__init__`: Initializes name, roll number, grade, and marks.
- `add_mark()`: Adds or updates subject marks.
- `get_average()`: Returns average marks.
- `get_details()`: Displays student details.
- `has_passed()`: Checks if all marks are >= 40.

## ✅ Output (Example)

```
Name: Sakib Malik
Roll No: 101
Grade: 10th
Average Marks: 75.00
Passed: True
```

"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=300, chunk_overlap=0)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[1])
