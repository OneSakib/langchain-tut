from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# schema


class Review(TypedDict):
    key_themes: Annotated[list[str],
                          "Write down all the key themes discussed in the review in a list."]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str,
                         "Return sentiment of the review either negative or positive"]
    pros: Annotated[Optional[list[str]],
                    "write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]],
                    "write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


structured_model = model.with_structured_output(Review)
result = structured_model.invoke(
    """The hardware is great, but the software feels bloated. There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compare to other brands hoping for a software update to be fix this.
    Review By Sakib Malik""")
print(result)
