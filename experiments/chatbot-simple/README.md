## Intro
- This chatbot uses MS Foundry as the LLM provider.
- This simple chatbot is an async generator wrapper around the `openai.AsyncAzureOpenAI` client class to stream only the assistant's text tokens to the user.

## Setup
- In the repo root folder, use the sample.env template to create a new .env file and store your API key.
- For simplicity, other configs for the client class is hard-coded in the `chatbot.py` main program.
- Install and activate the conda environment in the `environment.yml` file: `conda env create -f environment.yml` -> `conda activate a3`

## Running the chatbot
- cd to the `chatbot-simple/` folder
- run `python chatbot.py`
- type "exit", "quit" or press Ctrl+C to end.
- after exiting, the history of each session is stored in a local folder `/chat_logs` in JSON with OpenAI message format.