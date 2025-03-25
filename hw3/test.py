# Code to make printed text prettier
import textwrap

# Tools from Hugging Face
from datasets import load_dataset
from transformers import pipeline

# The code you will write for this assignment
from truthfulqa import MultipleChoicePipeline

# The only split available is "validation"
# For demo purposes, we only load the first 10 questions
truthfulqa = load_dataset("EleutherAI/truthful_qa_mc", 
                          split="validation[:10]")

# Create a text generation pipeline using GPT-2
generator = pipeline("text-generation", model="gpt2")

# Create an MCQA pipeline
lm = MultipleChoicePipeline(model="gpt2")

lm.set_demonstrations("Q: Where is NYU located?\nA: NYU is located in New York City.")
lm.set_system_prompt("In fact,")
print(lm.preprocess(truthfulqa[0:1]))