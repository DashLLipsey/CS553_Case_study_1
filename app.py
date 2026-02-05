import gradio as gr
from huggingface_hub import InferenceClient
import os

pipe = None
stop_inference = False

# Fancy styling
fancy_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""
# Creating example inputs and outputs for few shot learning
EXAMPLE_INPUT_1 = 'Make me lyrics and chords for a song in the style of Simon and Garfunkel about sitting through a computer science lecture'
EXAMPLE_OUTPUT_1 = """
'Fluorescent Afternoon'
Verse:
G - Em - C - G - G - Em - C - D
Morning light through dusty panes  
Coffee cooling in my hand 
The screen glows blue with ancient code 
Only half of it is planned 
The professor clears his throat 
Like a ritual we all know 
I underline a word called theory 
But I don't know where it goes 
Chorus:
C - G - Em - C - C - G - D
Hello, pointers, my old friends 
I've come to misread you again 
Because a slide was softly creeping 
Left its syntax while I was sleeping 
And the thought that formed inside my brain 
Was interrupted once again 
By the hum of fluorescent afternoon 
"""
EXAMPLE_INPUT_2 = 'Make me lyrics and chords for a song in the style of Travis Scott about someone driving to school'
EXAMPLE_OUTPUT_2 = """
'Late Bell (AM Drive)'
Hook:
Fm - Db - Ab - Eb - Fm - Db - Ab - Eb
I'm riding to school with the sun in my eyes 
Radio low but the bass still cries 
Running these lights, yeah I'm losing my time 
Late bell ringing but I'm still gonna slide 
Windows down, let the cold air bite 
Thoughts too loud in the early light 
I'm not awake but I'm still alive 
On the way to class, yeah I'm still gonna ride 
Verse:
Fm - Db - Ab - Eb - Fm - Db - Ab - Eb
Seat lean back, backpack on the floor 
Same street dreams that I had before 
Teachers talk but my mind elsewhere 
Trying find a future in the traffic glare 
Gas light on, but I'm pushing my luck 
Need more sleep, need way more trust 
Clock keep yelling that I'm behind 
But my soul moving faster than the hands of time 
"""
EXAMPLE_INPUT_3 = 'Make me chords and lyrics for a song in the style of Nirvana about Charlie Kirk'
EXAMPLE_OUTPUT_3 = """
'Campus Static' 
Verse:
Em - G - A - C - Em - G - A - C
T-shirt slogans, megaphone grin 
Selling answers in a paper-thin skin 
Talks real loud, says he's saving my soul 
But he's reading from a script he was sold 
Dorm room rage, hotel stage 
Same old war in a different age 
Says “think free” but it sounds rehearsed 
Like a bad idea wearing a tie and a curse 
Pre-chorus:
A - C - A - C
You say it's simple 
Like I'm dumb 
If I don't clap 
You say I've lost 
Chorus:
C - A - Em - G - C - A - Em - G
I don't need you 
Talking at me 
Like I'm broken 
Like I'm empty 
You don't scare me 
You just bore me 
Selling fear like 
It's conformity 
"""

def respond(
    message,                         # Input
    history: list[dict[str, str]],   # list or previous messages
    system_message,                  # Chatbot "instructions"
    max_tokens,                      # 3 parameters, length, randomness, diversity
    temperature,
    top_p,
    hf_token: gr.OAuthToken,         # API access token
    use_local_model: bool,           # Use local machine or remote machine
):
    
    global pipe                      # Cache the local pipeline

    # Build messages from history and prepend few-shot examples (Examples listed at the top)
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": EXAMPLE_INPUT_1},
                {"role": "assistant", "content": EXAMPLE_OUTPUT_1},
                {"role": "user", "content": EXAMPLE_INPUT_2},
                {"role": "assistant", "content": EXAMPLE_OUTPUT_2},
                {"role": "user", "content": EXAMPLE_INPUT_3},
                {"role": "assistant", "content": EXAMPLE_OUTPUT_3}
                ]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local_model:
        print("[MODE] local")
        # Using local machine use the transformers pipeline to get model
        from transformers import pipeline
        import torch
        if pipe is None:
            '''pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-3.2-1B-Instruct"  # Is this the model that we want?
            )'''
            pipe = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct"
            )

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        # Just output the answer, remove the prompt
        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()
    else:
        print("[MODE] api")
        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return
        # Use Hugging Face client and define model
        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response

    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a professional Songwriter and Lyricist." \
        " Your goal is to write lyrics that have a strong rhythm, clear structure, and creative rhymes." \
        " Follow these rules:" \
        "1. Always label your sections (e.g., [Verse 1], [Chorus], [Bridge])." \
        "2. Adapt your vocabulary to the requested genre and/or artist and song topic (e.g., use slang for Hip Hop, emotional imagery for Pop)." \
        "3. Always mention the chords in each section (e.g., G - Em - C)." \
        "4. Always include the name of the song.",
        label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
)


with gr.Blocks(css=fancy_css) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'> 🎵 Song Generator Chatbot🎵</h1>")
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()


