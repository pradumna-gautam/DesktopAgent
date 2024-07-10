import base64
import requests

def encode_image(image_path):
    """
    Encode an image to base64 format for API usage.
    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: File at {image_path} not found.")
        return None

def inference_chat(chat, api_token):    
    """
    Send a chat message to OpenAI's GPT-4 model and get the response.
    :param chat: List of tuples containing role and content for the chat.
    :param api_token: API token for authentication.
    :return: Response from the model.
    """
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    data = {
        "model": 'gpt-4-vision-preview',
        "messages": [],
        "max_tokens": 2048,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res.raise_for_status()  # Raises an HTTPError for bad responses
            res = res.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print("Network Error:")
            print(e)
            continue
        else:
            break
    
    return res

