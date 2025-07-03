from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
text = """
class Student:
    def __init__(self, name, roll_no, grade):
        self.name = name
        self.roll_no = roll_no
        self.grade = grade
        self.marks = {}

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
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=0)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[1])
