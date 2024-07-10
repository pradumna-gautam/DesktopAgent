import copy
from DesktopAgent.api import encode_image  

def init_chat(instruction):
    """
    Initialize chat history with the given instruction.
    :param instruction: The instruction to initialize the chat.
    :return: Initialized chat history.
    """
    operation_history = []
    system_prompt = "You are a helpful desktop operating assistant. You need to help me operate the desktop to complete my instruction.\n"
    system_prompt += f"My instruction is: {instruction}"
    operation_history.append(["user", [{"type": "text", "text": system_prompt}]])
    operation_history.append(["assistant", [{"type": "text", "text": "Sure. How can I help you?"}]])
    return operation_history

def add_response(role, prompt, chat_history, image=None):
    """
    Add a response to the chat history.
    :param role: The role of the responder (user or assistant).
    :param prompt: The text prompt of the response.
    :param chat_history: The current chat history.
    :param image: Optional image to include in the response.
    :return: Updated chat history.
    """
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
                "type": "text", 
                "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history

def add_multiimage_response(role, prompt, chat_history, images):
    """
    Add a multi-image response to the chat history.
    :param role: The role of the responder (user or assistant).
    :param prompt: The text prompt of the response.
    :param chat_history: The current chat history.
    :param images: List of images to include in the response.
    :return: Updated chat history.
    """
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
            "type": "text", 
            "text": prompt
        },
    ]
    for image in images:
        base64_image = encode_image(image)
        this_content = {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(this_content)

    new_chat_history.append([role, content])
    return new_chat_history

def print_status(chat_history):
    """
    Print the current status of the chat history.
    :param chat_history: The current chat history.
    """
    print("*" * 100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>" * (len(chat[1]) - 1))
    print("*" * 100)
