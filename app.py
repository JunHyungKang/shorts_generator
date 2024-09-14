import os

import streamlit as st
import streamlit_mermaid as stmd
import tiktoken
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from prompts.base import single_writing_prompt
from writer_agent import WriterAgent

# from your_custom_agent import CustomNovelAgent  # 여러분의 커스텀 에이전트 임포트

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=None,
)

def count_tokens(text: str) -> int:
    encoder = tiktoken.encoding_for_model("gpt-4o")
    return len(encoder.encode(text))

def generate_single_call(prompt):
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template=single_writing_prompt
    )
    chain = prompt_template | llm
    result = chain.invoke({"topic":prompt}).content
    tokens = count_tokens(result)
    return result, tokens

def generate_custom_agent(topic):
    agent = WriterAgent(llm)
    final_book = agent.run(topic)
    tokens = count_tokens(final_book)  # count_tokens 함수는 별도로 구현해야 합니다
    # graph_show = agent.show_graph()
    # return final_book, tokens, graph_show
    return final_book, tokens

st.title("AI 소설 생성 비교")

# 사용자 입력
topic = st.text_input("소설의 주제를 입력하세요:")

if st.button("소설 생성"):
    if topic:
        # 프로그레스 바 표시
        progress_bar = st.progress(0)
        
        # Langchain을 사용한 단일 호출 모델 생성
        progress_bar.progress(25)
        single_call_result, single_call_tokens = generate_single_call(topic)
        
        # 커스텀 에이전트 생성
        progress_bar.progress(75)
        # custom_agent_result, custom_agent_tokens, graph_show= generate_custom_agent(topic)
        custom_agent_result, custom_agent_tokens= generate_custom_agent(topic)
        # stmd.st_mermaid(graph_show)
        
        progress_bar.progress(100)
        
        # 결과 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Langchain 단일 호출 모델 결과")
            st.write(f"총 토큰 수: {single_call_tokens}")
            st.write(single_call_result)
        
        with col2:
            st.subheader("커스텀 에이전트 결과")
            st.write(f"총 토큰 수: {custom_agent_tokens}")
            st.write(custom_agent_result)
            st.write('아직 미구현')
        
        # 프로그레스 바 제거
        progress_bar.empty()
    else:
        st.warning("주제를 입력해주세요.")

# 스타일링을 위한 CSS
st.markdown("""
<style>
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)