import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal
import os


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.set_page_config(page_title="이미지 인식 기반 챗봇 💬", page_icon="💬", layout="wide")
st.title("이미지 인식 기반 챗봇 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

main1, main2 = st.tabs(["이미지", "대화내용"])


def print_history():
    for msg in st.session_state["messages"]:
        main2.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


@st.cache_resource(show_spinner="업로드한 이미지를 처리중입니다...")
def image_processing(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 체인을 생성합니다.
def create_chain(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    llm = ChatOpenAI(model=model_name, temperature=0)

    # 멀티모달 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 파일로 부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)
    return answer


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")
    # LLM 선택
    model_selection = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"])

    # 이미지 업로드
    file = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"],
    )

    system_prompt = st.text_area(
        "시스템 프롬프트",
        "당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.",
        height=200,
    )

if clear_btn:
    st.session_state["messages"].clear()

print_history()


if file:
    image_filepath = image_processing(file)
    main1.image(image_filepath, use_column_width=True)

msg_container = main2.empty()

if user_input := st.chat_input():
    # 이미지 파일이 업로드 되었을 경우
    if file:
        # 이미지 처리
        image_filepath = image_processing(file)
        # 멀티모달 체인에 이미지를 분석하는 요청
        answer = create_chain(
            image_filepath, system_prompt, user_input, model_name=model_selection
        )

        main2.chat_message("user").write(user_input)

        # 요청사항을 처리
        with main2.chat_message("assistant"):
            chat_container = st.empty()
            ai_answer = ""
            for chunk in answer:
                ai_answer += chunk.content
                chat_container.markdown(ai_answer)

        add_history("user", user_input)
        add_history("ai", ai_answer)
    else:
        msg_container.error("이미지 파일을 업로드해주세요.")
