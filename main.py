import os
from dotenv import load_dotenv
import wikipediaapi
import gradio as gr

from xai_sdk.chat import user, system
from xai_sdk import Client

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)

wiki = wikipediaapi.Wikipedia(
    user_agent="Veritas_Epistemics/0.1 (nitinmurali.03@gmail.com; contact via GitHub)",
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


def get_wikipedia_context(topic):
    page = wiki.page(topic)
    if not page.exists():
        print(f"Warning: No Wikipedia page found for '{topic}'")
        return ""

    context = f"Title: {page.title}\nURL: {page.fullurl}\n\nSummary:\n{page.summary[:800]}\n"

    if page.sections:
        context += f"\n\n({page.sections[0].title}):\n{page.sections[0].text[:400]}"

    return context.strip()


def generate_pipeline(topic):
    if not topic.strip():
        return "Please enter a topic!", "", "", "", ""

    yield "Retrieving Wikipedia context...", "", "", "", ""

    wiki_context = get_wikipedia_context(topic)

    yield "Generating article...", wiki_context, "", "", ""

    chat = client.chat.create(model="grok-4")

    chat.append(system(
        "You are a truthful, accurate encyclopedia writer focused on maximum truth-seeking. "
        "Cite sources where possible and avoid speculation."))

    chat.append(user(f"""
You are writing an encyclopedia article on '{topic}'.
Here is grounding context from Wikipedia to ensure factual accuracy:
{wiki_context}

Write a clear, factual 300-word encyclopedia-style article.
Be precise, include key facts, and ground your content in the provided Wikipedia context where possible.

For every factual claim or key piece of information, include an inline citation in the form [1], [2], etc.
At the end of the article, list the references (e.g., [1] Wikipedia article on Machine Learning: https://en.wikipedia.org/wiki/Machine_learning).
Use [1] for the Wikipedia URL, and [2], [3], etc. for any additional sources you reference.
"""))

    response = chat.sample()
    article = response.content.strip()

    yield "Critiquing for accuracy...", wiki_context, article, "", ""

    critique_chat = client.chat.create(model="grok-4")

    critique_chat.append(system(
        "You are a rigorous, no-nonsense fact-checker with broad scientific and general knowledge. "
        "Your job is to scrutinize the provided article for factual accuracy, potential hallucinations, "
        "overstatements, missing important context, unsubstantiated claims, or logical inconsistencies. "
        "Be extremely harsh but fair and evidence-based in your feedback. "
        "List any issues clearly with explanations and references to specific parts of the article. "
        "If the article is fully accurate and well-balanced, state that explicitly and praise its strengths. "
        "Do not add new information or speculate - only critique what's there."
    ))
    critique_chat.append(user(f"""
Critique this encyclopedia-style article strictly for factual accuracy:

{article}

Flag anything wrong, questionable, or incomplete. Provide reasoning for each point.
"""))

    critique_response = critique_chat.sample()
    critique = critique_response.content.strip()

    yield "Refining article...", wiki_context, article, critique, ""

    refine_chat = client.chat.create(model="grok-4")

    refine_chat.append(system(
        "You are an expert encyclopedia editor focused on maximum truth-seeking. "
        "Use the provided critique to revise the article: fix any flagged issues, add qualifiers for precision, include missing minor context if suggested, improve balance, and enhance citations where helpful. "
        "Even if the critique finds few issues, always make at least one small improvement (e.g., add a qualifier like 'loosely inspired', mention a key limitation, or include an additional reference). "
        "Keep the article factual, concise (~300 words), and output only the revised article."
    ))
    refine_chat.append(user(f"""
Original article:
{article}

Critique with issues to address:
{critique}

Revise the article accordingly for improved factual accuracy and completeness.
"""))

    refine_response = refine_chat.sample()
    refined_article = refine_response.content.strip()

    yield "Done!", wiki_context, article, critique, refined_article


def run_multi_agent_debate(article, wiki_context):
    # Agent 1: Defender
    defender_chat = client.chat.create(model="grok-4")
    defender_chat.append(system(
        "You are the Defender agent: strongly support and justify the key claims in the article. "
        "Provide additional evidence, reasoning, and citations from the grounding context or known facts."
    ))
    defender_chat.append(
        user(f"Defend this article:\n{article}\nGrounding context:\n{wiki_context}"))
    defender_response = defender_chat.sample().content.strip()

    # Agent 2: Challenger
    challenger_chat = client.chat.create(model="grok-4")
    challenger_chat.append(system(
        "You are the Challenger agent: critically attack weaknesses, contradictions, overstatements, "
        "or missing context in the article. Be aggressive but evidence-based."
    ))
    challenger_chat.append(
        user(f"Challenge this article:\n{article}\nGrounding context:\n{wiki_context}"))
    challenger_response = challenger_chat.sample().content.strip()

    # Agent 3: Arbiter (final refinement)
    arbiter_chat = client.chat.create(model="grok-4")
    arbiter_chat.append(system(
        "You are the Arbiter agent: read the original article, Defender's support, and Challenger's criticism. "
        "Resolve conflicts, strengthen weak points, add qualifiers, improve balance, and produce the final improved version. "
        "Keep it ~300 words, factual, with citations. Output only the final article."
    ))
    arbiter_chat.append(user(f"""
Original article:
{article}

Defender's arguments:
{defender_response}

Challenger's criticisms:
{challenger_response}

Grounding context:
{wiki_context}

Produce the final refined article.
"""))
    final_response = arbiter_chat.sample().content.strip()

    return defender_response, challenger_response, final_response


def run_debate_only(article, wiki_context, critique):
    if not article.strip():
        return "No article to debate yet! Run 'Generate Article' first.", ""

    defender, challenger, final = run_multi_agent_debate(article, wiki_context)

    transcript = f"**Defender:**\n{defender}\n\n**Challenger:**\n{challenger}"

    return final, transcript


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
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    body_text_size="md",
)

with gr.Blocks(theme=dark_theme, title="Veritas Epistemics - Truth-Seeking Article Generator") as demo:
    gr.Image(
        value="veritas_title.png",
        show_label=False,
        interactive=False,
        container=False,
        height=100,
        width=300,
        elem_id="veritas-title",
        show_download_button=False,
        show_fullscreen_button=False,  # ← add this
    )

    gr.HTML("""
    <style>
        /* Center and tighten title image */
        #veritas-title {
            text-align: center !important;
            margin: 15px auto 5px auto !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        #veritas-title img {
            display: block !important;
            margin: 0 auto !important;
            padding: 0 !important;
            border: none !important;
            box-shadow: none !important;
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
        }

        /* Nuke Gradio image toolbar / controls completely */
        .gr-image .gr-image-toolbar,
        .gr-image .toolbar,
        .gr-image-toolbar,
        .gr-image-container .toolbar,
        .gr-box .gr-image-controls,
        .gr-image-controls,
        .gr-image .gr-image-header,
        .gr-image-header {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            min-height: 0 !important;
            min-width: 0 !important;
            border: none !important;
        }

        /* Force parent containers to collapse */
        .gr-image-container,
        .gr-image,
        .gr-box:has(.gr-image) {
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            overflow: hidden !important;
            background: transparent !important;
        }

        /* Your other styles (chat input, send btn, etc.) */
        .chat-input-container {
            position: relative;
            max-width: 700px !important;
            margin: 10px auto 20px auto !important;
        }
        .chat-input-container .gr-textbox {
            border-radius: 12px !important;
            padding: 12px 70px 12px 16px !important;  /* increased right padding from 60px → 70px */
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
        }
        .chat-input-container .gr-textbox textarea {
            resize: none !important;
            font-size: 1rem !important;
            padding-right: 60px !important;
        }
        .send-btn {
            position: absolute !important;
            bottom: 2px !important;
            right: 2px !important;
            width: 37px !important;
            height: 37px !important;
            border-radius: 6px !important;
            background-color: #5170ff !important;
            color: white !important;
            border: none !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.5rem !important;
            z-index: 20 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        }
        .send-btn:hover {
            background-color: #4f46e5 !important;
        }
        #topic-input-box::placeholder {
            font-size: 0.7rem !important;
            color: #aaaaaa !important;
            opacity: 0.8 !important;
        }
        #topic-input-box {
            font-size: 1rem !important;
        }
            .icon-button-wrapper.top-panel.hide-top-corner,
    .icon-button-wrapper,
    .top-panel,
    .hide-top-corner,
    .gr-image .icon-button-wrapper,
    .gr-image .top-panel {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
        min-height: 0 !important;
     min-width: 0 !important;
     border: none !important;
     background: transparent !important;
    }

    /* Force the image container to have no reserved space */
    .gr-image-container:has(.icon-button-wrapper),
    .gr-image .gr-image-header,
    .gr-image-header {
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }

    /* Extra safety: clip any overflow */
    #veritas-title {
        overflow: hidden !important;
        clip-path: inset(0 0 0 0) !important;
    }
    </style>
    """)

    with gr.Column(scale=1, min_width=500, elem_classes=["chat-input-container"]):
        topic_input = gr.Textbox(
            label="Truth-Seeking Article Generator",
            placeholder="Enter a topic of your choice. e.g., Machine Learning, Blockchain Technology, Global Warming",
            lines=1,
            container=False,
            elem_id="topic-input-box",
            elem_classes=["chat-input"]
        )

        send_arrow = gr.Button(
            value="➤",
            variant="primary",
            elem_id="send-arrow-btn",
            elem_classes=["send-btn"],
            size="sm"
        )

    status = gr.Markdown("Ready...", elem_id="status")

    with gr.Tabs():
        with gr.TabItem("Initial Generation"):
            with gr.Row(equal_height=True):
                wiki_output = gr.Textbox(
                    label="Wikipedia Context",
                    lines=15,
                    max_lines=50,
                    interactive=False,
                    show_copy_button=True
                )
                article_output = gr.Textbox(
                    label="Original Generated Article",
                    lines=15,
                    max_lines=50,
                    interactive=False,
                    show_copy_button=True
                )

        with gr.TabItem("Critique & Refinement"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("**Original**")
                    original_ref = gr.Textbox(
                        value=article_output.value,
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Critique**")
                    critique_output = gr.Textbox(
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Refined**")
                    refined_output_basic = gr.Textbox(
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )

        with gr.TabItem("Advanced Debate"):
            debate_btn = gr.Button(
                "Run Multi-Agent Debate & Refine", variant="secondary")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("**Pre-Debate Refined**")
                    pre_debate_refined = gr.Textbox(
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Debate Transcript**")
                    debate_transcript = gr.Textbox(
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Post-Debate Final**")
                    final_refined = gr.Textbox(
                        lines=20,
                        max_lines=60,
                        interactive=False,
                        show_copy_button=True
                    )

    # Button connections
    # generate_btn.click(
    #     fn=generate_pipeline,
    #     inputs=topic_input,
    #     outputs=[status, wiki_output, article_output,
    #              critique_output, refined_output_basic]
    # )

    debate_btn.click(
        fn=run_debate_only,
        inputs=[article_output, wiki_output, critique_output],
        outputs=[debate_transcript, final_refined]
    )

    send_arrow.click(
        fn=generate_pipeline,
        inputs=topic_input,
        outputs=[status, wiki_output, article_output,
                 critique_output, refined_output_basic]
    )

if __name__ == "__main__":
    demo.launch()
