import os
import pandas as pd
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents.agent_types import AgentType


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.set_page_config(page_title="엑셀 기반 챗봇 💬", page_icon="💬")
st.title("엑셀 기반 챗봇 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None


# 이전 대화 내용을 출력
def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 대화 기록을 저장
def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 탭 구분
main1, main2 = st.tabs(["파일", "대화내용"])


# 파일 업로드 기능
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # csv, excel 파일인 경우
    extension = file_path.split(".")[-1]

    df = None
    # CSV 파일인 경우
    if extension == "csv":
        df = pd.read_csv(file_path)
    # Excel 파일인 경우
    elif extension == "xlsx" or extension == "xls":
        df = pd.read_excel(file_path)
    # 데이터프레임을 반환합니다.
    return df


with st.sidebar:
    # 대화내용을 초기화 하기 위한 버튼
    clear_btn = st.button("대화내용 초기화")

    # 파일 업로드
    file = st.file_uploader(
        "파일 업로드",
        type=["csv", "xls", "xlsx"],
    )

# 사용자가 파일을 업로드를 한다면..
if file:
    # Excel 을 처리해 주는 도구를 생성합니다.
    df = embed_file(file)
    main1.dataframe(df)
    st.session_state["df"] = df


if clear_btn:
    retriever = st.session_state["messages"].clear()

# 이전 대화를 출력
print_history()

# 메시지 출력을 위한 빈 창을 만든다.
msg_container = st.empty()


def create_agent(df, callback):
    # 에이전트 생성
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",  # 모델은 반드시 gpt-4o-mini 를 사용해야 오류가 안납니다.
            streaming=True,
            callbacks=[callback],
        ),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
    )
    return agent


# 사용자가 대화를 입력 하면..
if user_input := st.chat_input():
    # 파일 업로드가 되어 있다면..
    if st.session_state["df"] is not None:
        # 사용자의 메시지를 출력을 합니다.
        st.chat_message("user").write(user_input)

        # 답변을 스트리밍으로 받기 위한 설정
        class StreamCallback(BaseCallbackHandler):
            def set_container(self, container):
                self.container = container
                self.ai_answer = ""

            def on_llm_new_token(self, token: str, **kwargs):
                self.ai_answer += token
                self.container.markdown(self.ai_answer)

        with st.chat_message("assistant"):
            # 메시지를 실시간 출력할 빈 창을 하나 만든다.
            chat_container = st.empty()
            # 빈 창을 셋팅을 한다.
            callback = StreamCallback()
            callback.set_container(chat_container)

            # 스트리밍 출력을 위한 콜백을 agent 만들때 넣어 준다.
            agent = create_agent(st.session_state["df"], callback=callback)

            # 질의
            ai_answer = agent.invoke({"input": user_input})

        # 대화 기록을 저장
        add_history("user", user_input)
        add_history("ai", ai_answer["output"])
    else:
        msg_container.error("파일을 업로드 해주세요(CSV, Excel)")
