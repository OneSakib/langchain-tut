import random
from abc import ABC, abstractmethod
import warnings


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of india',
            'IPL is a Cricket League',
            "AI Standands for Artificial Intelleigence"
        ]
        warnings.warn(
            "old_method() is deprecated and will be removed in a future version. Use new_method() instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return {'response': random.choice(response_list)}

    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of india',
            'IPL is a Cricket League',
            "AI Standands for Artificial Intelleigence"
        ]
        return {'response': random.choice(response_list)}


llm = NakliLLM()

result = llm.invoke("What is the capital of india")

# print(result)


class NakliPromptTempate(Runnable):
    def __init__(self, template, input_variables) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        warnings.warn(
            "old_method() is deprecated and will be removed in a future version. Use new_method() instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return self.template.format(**input_dict)

    def invoke(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTempate(
    template="write a poem about a topic {topic}", input_variables=['topic'])


prompt = template.invoke({'topic': 'India'})

# print("Prompt:", prompt)

result = llm.predict(prompt)
# print("result:", result)


class NakliLLMChain(Runnable):
    def __init__(self, llm, prompt) -> None:
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        warnings.warn(
            "old_method() is deprecated and will be removed in a future version. Use new_method() instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        final_prompt = self.prompt.format(**input_dict)
        result = self.llm.invoke(final_prompt)
        return result['response']

    def invoke(self, input_dict):
        final_prompt = self.prompt.format(**input_dict)
        result = self.llm.invoke(final_prompt)
        return result['response']


chain = NakliLLMChain(llm=llm, prompt=prompt)

result = chain.invoke({'topic': "India"})

# print("Result:", result)


class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def parse(self, data):
        return data

    def invoke(self, data):
        return data['response']


class RunnableConnactor(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_dict):
        for runnable in self.runnable_list:
            input_dict = runnable.invoke(input_dict)
        return input_dict


parser = NakliStrOutputParser()
chain = RunnableConnactor([template, llm, parser])

print("Chain:", chain.invoke({'topic': "india"}))
