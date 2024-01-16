import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
import os

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# model = genai.GenerativeModel(model_name = "gemini-pro")
model = ChatGoogleGenerativeAI(model="gemini-pro")

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

code_prompt = PromptTemplate(
    template="Write a {language} code that will perform the following task: {task} (simply the code no further explanations)",
    input_variables=['language', 'task']
)

test_prompt = PromptTemplate(
    input_variables=['language', 'code'],
    template="Write a test for the following {language} code : \n {code}"
)

code_chain = LLMChain(
    llm=model,
    prompt=code_prompt,
    output_key="code"
)
test_chain = LLMChain(
    llm=model,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task": args.task
})

print("CODE : ")
print(result["code"])
print("TEST : ")
print(result["test"])