from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
    You are an expert in explaining research papers.
    Please summarize the research paper titled "{paper_name}" with the following specifications:    
    Explanation Style: {style}
    Explanation Length: {length}
    1. Mathematial Details:
        - Include relevant mathematical equation if present in the paper.
        - Explain the mathematical conecpt using simple, intitutive code snippet where applicable.
    2. Analogies:
        - Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with "Insufficient information available" instead of gussing.
    Insure the summary clear, accurate, and aligned with provided style and length.
    """,
    input_variables=["paper_name", "style", "length"],
    validate_template=True

)
template.save('template.json')