import chainlit as cl
import requests
import os

confluence_url = "https://mikesofts.atlassian.net/"
email = os.environ['confluence_user']
api_token = os.environ['confluence_api_token']

# Set headers for authentication
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# Provide your email and API token for basic authentication
auth = (email, api_token)
# print(auth)
@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    endpoint = f"{confluence_url}/wiki/rest/api/content/{message}"
    response = requests.get(endpoint, headers=headers, auth=auth)
    response_json = response.json()
    # Send a response back to the user
    await cl.Message(
        content=f"Hey buddy! For this id: {message} the title is {response_json['title']}",
    ).send()