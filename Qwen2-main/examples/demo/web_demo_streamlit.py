import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


st.set_page_config(page_title="😴SleepingFace-Chat😴")
st.title("😴SleepingFace-Chat😴")


@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen2/LLaMA-Factory-main/outputs",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "/root/autodl-tmp/Qwen2/LLaMA-Factory-main/outputs"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/Qwen2/LLaMA-Factory-main/outputs",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是SleepingFace小助手，很高兴为您服务😴")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '😴' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def generate_response(model, tokenizer, messages):
    # Prepare the input text by concatenating all messages
    input_text = ""
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        input_text += f"{role}: {message['content']}\n"

    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # Generate a response
    outputs = model.generate(inputs, max_length=512, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's response part
    response = response.split("Assistant:")[-1].strip()
    return response


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='😴'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            response = generate_response(model, tokenizer, messages)
            placeholder.markdown(response)

        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
