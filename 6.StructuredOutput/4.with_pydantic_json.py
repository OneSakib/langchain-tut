from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# schema

json_schema = {
    "title": "Review",
    "description": "schema about student",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "write down all the key themes discussed in the review in a list."
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative or positive"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list."
        },
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list."
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["name"]
}


structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke(
    """The hardware is great, but the software feels bloated. There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compare to other brands hoping for a software update to be fix this.
    Review By Sakib Malik""")
print(result)
