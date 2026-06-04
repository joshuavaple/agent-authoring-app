# streamlit_app.py
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.title("FastAPI Demo")

# ── POST: Capitalize ──────────────────────────────────────────────
st.header("POST /transform")
post_input = st.text_input("Input to capitalize", key="post_input")
if st.button("Send POST"):
    if post_input:
        res = requests.post(f"{API_BASE}/transform", json={"user_input": post_input})
        if res.ok:
            st.success(f"Result: {res.json()['result']}")
        else:
            st.error(f"Error {res.status_code}: {res.text}")
    else:
        st.warning("Enter some text first.")

st.divider()

# ── GET: Lookup ───────────────────────────────────────────────────
st.header("GET /lookup")
get_input = st.text_input("Key to look up (try: alpha, beta, gamma)", key="get_input")
if st.button("Send GET"):
    if get_input:
        res = requests.get(f"{API_BASE}/lookup", params={"user_input": get_input})
        if res.ok:
            data = res.json()
            st.success(f"Value: {data['value']}")
        else:
            st.error(f"Error {res.status_code}: {res.json().get('detail', res.text)}")
    else:
        st.warning("Enter a key first.")