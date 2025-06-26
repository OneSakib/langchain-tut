import random


class NakliLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of india',
            'IPL is a Cricket League',
            "AI Standands for Artificial Intelleigence"
        ]
        return {'response': random.choice(response_list)}


llm = NakliLLM()

result = llm.predict("What is the capital of india")

print(result)


class NakliPromptTempate:
    def __init__(self, template, input_variables) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTempate(
    template="write a poem about a topic {topic}", input_variables=['topic'])


prompt = template.format({'topic': 'India'})

print("Prompt:", prompt)

result = llm.predict(prompt)
print("result:", result)


class NakliLLMChain:
    def __init__(self, llm, prompt) -> None:
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(**input_dict)
        result = self.llm.predict(final_prompt)
        return result['response']


chain = NakliLLMChain(llm=llm, prompt=prompt)

result = chain.run({'topic': "India"})

print("Result:", result)
