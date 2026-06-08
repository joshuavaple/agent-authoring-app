#!/usr/bin/env bash
uvicorn main:app --reload &
trap "kill $!" EXIT
streamlit run ui.py
