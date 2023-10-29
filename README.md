### Chatbot
Dashboard to conversationally retrieve information from user-provided document using an LLM.

### Installation
1. clone repo <br>
`git clone https://github.com/mikeolubode/kagglex.git` <br>

2. create a new environment <br>
`python -m venv .venv`

3. activate new environment <br>
`source .venv/bin/activate`

4. install dependencies<br>
`pip install -r requirements.txt`

5. set up the following environment variables
EMAIL, GOOGLE_PALM_API_KEY and CONFLUENCE_API_TOKEN

### Usage
Run application `chainlit run src/app.py -w`