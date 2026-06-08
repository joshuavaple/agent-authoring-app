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

st.divider()

# ── GET: Search Experiments ───────────────────────────────────────
st.header("GET /search_experiments")
view_type_label = st.selectbox("View type", options=["ACTIVE_ONLY", "DELETED_ONLY", "ALL"], key="view_type")
max_results = st.number_input("Max results", min_value=1, value=5, step=1, key="max_results")
if st.button("Search experiments"):
    params = {"view_type": view_type_label, "max_results": int(max_results)}
    res = requests.get(f"{API_BASE}/search_experiments", params=params)
    if res.ok:
        experiments = res.json() # list of dicts parse from res.content
        if experiments:
            st.success(f"Found {len(experiments)} experiment(s)")
            # experiments = [exp.pop("tags", None) for exp in experiments]
            st.dataframe(experiments)
        else:
            st.info("No experiments found.")
    else:
        st.error(f"Error {res.status_code}: {res.text}")