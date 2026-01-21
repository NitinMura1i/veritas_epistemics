import os
from dotenv import load_dotenv
import wikipediaapi
import gradio as gr
import pandas as pd
import json

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


def generate_synthetic_eval(refined_article):
    if not refined_article.strip():
        return "No refined article yet! Generate and refine first.", None, None, None

    # Step 1: Generate synthetic Q&A pairs
    synth_prompt = f"""
From this refined article:
{refined_article}

Generate exactly 5 high-quality synthetic evaluation Q&A pairs for factual testing.
For each:
- "question": A clear, diverse question about the article.
- "golden_answer": Direct, accurate answer from the article.
- "distractors": List of 2 plausible but wrong answers (hallucinations or common misconceptions).
- "difficulty": "easy", "medium", or "hard"

Output as valid JSON list only, no extra text.
Example format:
[{{"question": "...", "golden_answer": "...", "distractors": ["wrong1", "wrong2"], "difficulty": "medium"}}, ...]
"""

    synth_chat = client.chat.create(model="grok-4")
    synth_chat.append(system(
        "You are an expert synthetic data generator for LLM evaluation. Output only valid JSON."))
    synth_chat.append(user(synth_prompt))
    synth_response = synth_chat.sample().content.strip()

    # Parse JSON
    try:
        qa_pairs = json.loads(synth_response)
    except:
        qa_pairs = []

    # Step 2: Evaluate factual accuracy
    eval_prompt = f"""
Evaluate the refined article for factual accuracy, bias, and hallucination risk (0-10 scale).
Article: {refined_article}
Provide:
- overall_score: number 0-10
- reasoning: brief explanation
- bias_flags: list of any detected biases (or empty list)
- hallucination_risk: "low", "medium", "high"

Output as JSON only.
"""

    eval_chat = client.chat.create(model="grok-4")
    eval_chat.append(
        system("You are a strict factual evaluator. Output only valid JSON."))
    eval_chat.append(user(eval_prompt))
    eval_response = eval_chat.sample().content.strip()

    try:
        eval_data = json.loads(eval_response)
    except:
        eval_data = {"overall_score": 0, "reasoning": "Eval failed",
                     "bias_flags": [], "hallucination_risk": "unknown"}

    # Prepare table data
    df_data = []
    for pair in qa_pairs:
        df_data.append({
            "Question": pair.get("question", ""),
            "Golden Answer": pair.get("golden_answer", ""),
            "Distractors": ", ".join(pair.get("distractors", [])),
            "Difficulty": pair.get("difficulty", "")
        })

    df = pd.DataFrame(df_data)

    # Score text
    score_text = f"**Factual Accuracy Score: {eval_data.get('overall_score', 'N/A')}/10**\n\n" \
        f"**Reasoning:** {eval_data.get('reasoning', 'N/A')}\n\n" \
        f"**Bias Flags:** {', '.join(eval_data.get('bias_flags', [])) or 'None'}\n" \
        f"**Hallucination Risk:** {eval_data.get('hallucination_risk', 'N/A')}"

    # JSON for download
    json_data = {
        "qa_pairs": qa_pairs,
        "evaluation": eval_data,
        "article": refined_article
    }
    json_str = json.dumps(json_data, indent=2)

    return score_text, df, json_str


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
        show_fullscreen_button=False,
    )

    gr.HTML("""
    <style>
        #veritas-title {
            text-align: center;
            margin: 15px auto 5px auto !important;
        }
        #veritas-title img {
            display: block;
            margin: 0 auto;
        }
        .chat-input-container {
            position: relative;
            max-width: 700px !important;
            margin: 10px auto 20px auto !important;
        }
        .chat-input-container .gr-textbox {
            border-radius: 12px !important;
            padding: 12px 70px 12px 16px !important;
            background-color: #111111 !important;
            color: #e5e7eb !important;
            border: 1px solid #444444 !important;
        }
        .chat-input-container .gr-textbox textarea {
            resize: none !important;
            font-size: 1.1rem !important;
        }
        .send-btn {
            position: absolute !important;
            bottom: 8px !important;
            right: 8px !important;
            width: 40px !important;
            height: 40px !important;
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
        #topic-input-box::placeholder {
            font-size: 0.7rem !important;
            color: #aaaaaa !important;
            opacity: 0.8 !important;
        }
        #topic-input-box {
            font-size: 1rem !important;
        }
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
            background: transparent !important;
        }
        #veritas-title .gr-image,
        #veritas-title img {
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            box-shadow: none !important;
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
                        value="This is where your truth-seeking article will be displayed.\n\n" +
                              "Enter a topic above and click the arrow to generate the initial version.\n" +
                              "Then use Critique, Debate, or Synthetic Eval to iteratively sculpt it closer to maximum truth.",
                        lines=25,
                        max_lines=80,
                        interactive=False,
                        show_copy_button=True,
                        label="Current Sculpted Article",
                        elem_classes=["center-article"]
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

        with gr.TabItem("Synthetic Data & Eval"):
            eval_btn = gr.Button(
                "Generate Synthetic Eval Data", variant="secondary")
            eval_status = gr.Markdown(
                "Click to generate evaluation data from refined article.")
            eval_score = gr.Markdown()
            eval_table = gr.Dataframe(interactive=False)
            eval_download = gr.File(
                label="Download JSON", file_types=[".json"])

    # Button connections
    send_arrow.click(
        fn=generate_pipeline,
        inputs=topic_input,
        outputs=[status, wiki_output, article_output,
                 critique_output, refined_output_basic]
    )

    debate_btn.click(
        fn=run_debate_only,
        inputs=[article_output, wiki_output, critique_output],
        outputs=[debate_transcript, final_refined]
    )

    eval_btn.click(
        fn=generate_synthetic_eval,
        inputs=refined_output_basic,
        outputs=[eval_score, eval_table, eval_download]
    )

if __name__ == "__main__":
    demo.launch()
