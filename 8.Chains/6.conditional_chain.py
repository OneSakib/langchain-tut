from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Give the sentiment of the feedback")


parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive and negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser_2.get_format_instructions()}
)

classifier_chain = prompt_1 | model | parser_2


# branch chain
prompt_2 = PromptTemplate(
    template="Write an appropriate response from the postive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
    template="Write an appropriate response from the negative feedback \n {feedback}",
    input_variables=['feedback']
)
# branch_chain = RunnableBranch(
#     (condition_1, chain_1),
#     (condition_2, chain_2),
#     default_chain
# )
chain_1 = prompt_2 | model | parser
chain_2 = prompt_3 | model | parser
# lambda x: "Could not Find sentiment" --- we can't run  this as a chain so we need to convert it into lambda runnable
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', chain_1),
    (lambda x: x.sentiment == 'negative', chain_2),
    RunnableLambda(lambda x: "Could not Find sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke(
    {"feedback": "This is a beautiful phone"})
print(result)
