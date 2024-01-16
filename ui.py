import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def main():
    load_dotenv()
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    # model = genai.GenerativeModel(model_name = "gemini-pro")
    model = ChatGoogleGenerativeAI(model="gemini-pro")
    with st.form("Pycode"):
        st.header("Pycode")
        with st.chat_message("assistant"):
            st.write("Hello 👋 This is a Google Gemini-Pro model based tool for converting prompts into code")
        task = 0
        language = st.selectbox('Select your programming language',
                                ("Javascript", "Java", "Python", "C", "C++", "Ruby", "Go", "Rust", "TypeScript"))
        task = st.text_area('Enter the task :')

        prompt2code = st.form_submit_button("Yalla !")
        if prompt2code:
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
                "language": language,
                "task": task
            })

            st.text("Bingo! Let's wish that this is what you asked for !")
            st.write(result["code"])

            st.text("Bonus! Here is how to test whether it works the correct way !")
            st.write(result["test"])


if __name__ == "__main__":
    main()
