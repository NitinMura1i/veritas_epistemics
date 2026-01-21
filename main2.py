import os
from dotenv import load_dotenv
import gradio as gr

from xai_sdk import Client

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)

# Dark theme
dark_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#0f0f0f",
    body_background_fill_dark="#0f0f0f",
    block_background_fill="#1a1a1a",
    block_background_fill_dark="#1a1a1a",
    block_title_text_color="#ffffff",
    block_label_text_color="#e0e0e0",
    input_background_fill="#111111",
    input_background_fill_dark="#111111",
    input_border_color="#444444",
    input_border_color_dark="#444444",
    input_border_color_focus="#6366f1",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    body_text_size="md",
)

with gr.Blocks(theme=dark_theme, title="Veritas Epistemics - Truth-Seeking Article Generator") as demo:
    # Centered title image
    gr.Image(
        value="veritas_title.png",
        show_label=False,
        interactive=False,
        container=False,
        height=120,
        width=None,
        elem_id="veritas-title",
        show_download_button=False,
        show_fullscreen_button=False,
    )

    # CSS: center title + clean chat-style input with arrow
    gr.HTML("""
    <style>
        #veritas-title {
            text-align: center !important;
            margin: 30px auto -50px auto !important;
            padding: 0 !important;
        }
        #veritas-title img {
            display: block !important;
            margin: 0 auto !important;
            max-width: 90% !important;
        }
        footer {
            display: none !important;  /* hide Gradio footer */
        }
        .chat-input-container {
            position: relative;
            max-width: 800px !important;
            margin: 0 auto 20px auto !important;
        }
        .chat-input-container .gr-textbox {
            border-radius: 12px !important;
            padding: 14px 70px 14px 18px !important;
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }
        .chat-input-container .gr-textbox textarea {
            resize: none !important;
            font-size: 1.1rem !important;
            line-height: 1.5 !important;
        }
        .chat-input-container .gr-textbox textarea::placeholder {
            color: #aaaaaa !important;
            opacity: 0.8 !important;
            font-size: 1rem !important;
        }
        .send-btn {
            position: absolute !important;
            bottom: 3px !important;
            right: 3px !important;
            width: 36px !important;
            height: 36px !important;
            border-radius: 8px !important;
            background-color: #6366f1 !important;
            color: white !important;
            border: none !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.6rem !important;
            z-index: 10 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        }
        .send-btn:hover {
            background-color: #4f46e5 !important;
        }
        /* Central article styling */
        .central-article {
            border-radius: 12px !important;
            background-color: #1a1a1a !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
            padding: 20px !important;
            font-family: monospace !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            min-height: 400px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
            margin: 0px auto 40px auto !important;
            max-width: 800px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        }
        .central-article::placeholder {
            color: #777777 !important;
            opacity: 0.7 !important;
            font-style: italic !important;
            text-align: center !important;
        }
        /* Toolbar nuke (your existing rule kept) */
        .icon-button-wrapper.top-panel.hide-top-corner,
        .icon-button-wrapper.top-panel,
        .icon-button-wrapper,
        .top-panel,
        .hide-top-corner,
        div[class*="icon-button-wrapper"],
        div[class*="top-panel"],
        div[class*="hide-top-corner"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        .gr-image-container:has(div.icon-button-wrapper),
        .gr-image-container:has(.top-panel),
        .gr-image-container:has(.hide-top-corner) {
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }
    </style>
    """)

    # Chat-style input with send arrow in corner
    with gr.Column(elem_classes=["chat-input-container"]):
        topic_input = gr.Textbox(
            placeholder="Enter a topic of your choice. e.g. Machine Learning, Astrology, Global Warming",
            lines=1,
            container=False,
            elem_id="topic-input-box"
        )

        send_arrow = gr.Button(
            value="➤",
            variant="primary",
            elem_id="send-arrow-btn",
            elem_classes=["send-btn"],
            size="sm"
        )

    # Central article section (always visible, large rectangle with default text)
    gr.Textbox(
        value="This is where your article will appear. Iterate it in order to get as close to the truth as you can!",
        lines=25,
        max_lines=80,
        interactive=False,
        show_copy_button=False,  # no copy for placeholder
        container=False,
        elem_classes=["central-article"],
        label=""  # no label to keep it clean
    )

# Launch without footer
demo.launch(show_api=False)
