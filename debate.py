# debate.py
# Multi-agent debate functionality

import os
from datetime import datetime

from config import client
from helpers import generate_edit_log
from xai_sdk.chat import user, system


def run_multi_agent_debate(current_article_clean: str, original_article: str):
    """Run multi-agent debate on current article.

    Args:
        current_article_clean: The current article content (without headers)
        original_article: The original article content for display

    Yields: (left, center, right, filepath, state_updates)

    state_updates is None during streaming, and on final yield contains:
    {
        'current_article_clean': str,  # Updated article content
        'article_history_entry': str   # Entry to append to article_history
    }
    """
    if not current_article_clean:
        error_msg = "⚠️ No article to debate. Please generate an article first."
        return error_msg, "", "", None, None

    # Store original for comparison later
    original_for_diff = current_article_clean

    # Center panel: original article (will transition to revised at end)
    center_display = "📄 ORIGINAL ARTICLE\n"
    center_display += "=" * 42 + "\n\n"
    center_display += original_article

    # Left panel: debate transcript (progressive updates)
    debate_transcript = "🎭 DEBATE TRANSCRIPT\n"
    debate_transcript += "=" * 42 + "\n\n"
    debate_transcript += "⏳ Initializing debate agents...\n\n"

    # Right panel: edit log placeholder
    edit_log_placeholder = "📝 EDIT LOG\n"
    edit_log_placeholder += "=" * 42 + "\n\n"
    edit_log_placeholder += "Changes will appear here after the debate concludes.\n\n"
    edit_log_placeholder += "The arbiter will synthesize the debate into a revised article, "
    edit_log_placeholder += "and this panel will show exactly what was changed."

    yield debate_transcript, center_display, edit_log_placeholder, None, None

    # Agent 1: Defender
    debate_transcript = "🎭 DEBATE TRANSCRIPT\n"
    debate_transcript += "=" * 42 + "\n\n"
    debate_transcript += "🟢 DEFENDER AGENT\n"
    debate_transcript += "-" * 42 + "\n"
    debate_transcript += "⏳ Analyzing article for strengths...\n\n"
    yield debate_transcript, center_display, edit_log_placeholder, None, None

    try:
        defender_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        defender_chat.append(system(
            "You are the Defender agent in an epistemic debate. Your role is to identify the strongest "
            "epistemic qualities of the article: appropriate certainty language, balanced framing, "
            "acknowledgment of limitations, and clear communication. Argue for what the article does well "
            "in terms of epistemic integrity. Be rigorous but fair. Provide a concise defense (~150 words)."
        ))
        defender_chat.append(user(
            f"Defend the epistemic strengths of this article:\n\n{current_article_clean}"))

        # Stream Defender response
        defender_response = ""
        debate_base = "🎭 DEBATE TRANSCRIPT\n" + "=" * \
            42 + "\n\n🟢 DEFENDER AGENT\n" + "-" * 42 + "\n"

        for response, chunk in defender_chat.stream():
            if chunk.content:
                defender_response += chunk.content
                streaming_transcript = debate_base + defender_response
                yield streaming_transcript, center_display, edit_log_placeholder, None, None

        debate_transcript = debate_base + f"{defender_response}\n\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None, None

        # Agent 2: Challenger
        debate_transcript += "🔴 CHALLENGER AGENT\n"
        debate_transcript += "-" * 42 + "\n"
        debate_transcript += "⏳ Analyzing article for weaknesses...\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None, None

        challenger_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        challenger_chat.append(system(
            "You are the Challenger agent in an epistemic debate. Your role is to critically examine "
            "the article for epistemic weaknesses: overstatements, unwarranted certainty, missing caveats, "
            "one-sided framing, lack of epistemic humility, or unclear communication. Focus on how claims "
            "are presented, not their factual accuracy. Be aggressive but fair. Point out specific issues. "
            "Provide a concise critique (~150 words)."
        ))
        challenger_chat.append(user(
            f"Challenge the epistemic quality of this article:\n\n{current_article_clean}"))

        # Stream Challenger response
        challenger_response = ""
        challenger_base = debate_transcript[:-
                                            len("⏳ Analyzing article for weaknesses...\n\n")]

        for response, chunk in challenger_chat.stream():
            if chunk.content:
                challenger_response += chunk.content
                streaming_transcript = challenger_base + challenger_response
                yield streaming_transcript, center_display, edit_log_placeholder, None, None

        debate_transcript = challenger_base + f"{challenger_response}\n\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None, None

        # Agent 3: Arbiter (produces revised article)
        debate_transcript += "⚖️ ARBITER AGENT\n"
        debate_transcript += "-" * 42 + "\n"
        debate_transcript += "⏳ Synthesizing debate and producing revised article...\n\n"
        yield debate_transcript, center_display, edit_log_placeholder, None, None

        arbiter_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        arbiter_chat.append(system(
            "You are the Arbiter agent. Read the original article, the Defender's support, and the Challenger's criticisms. "
            "Produce a revised article that:\n"
            "- Strengthens valid points raised by the Defender\n"
            "- Addresses valid criticisms from the Challenger\n"
            "- Adds qualifiers and nuance where needed\n"
            "- Maintains factual accuracy and epistemic integrity\n"
            "- Keeps the same structure and ~300 word length\n"
            "Output ONLY the final revised article in markdown format, with inline citations and a Sources section."
        ))
        arbiter_chat.append(user(f"""Original article:
{current_article_clean}

Defender's arguments:
{defender_response}

Challenger's criticisms:
{challenger_response}

Produce the final revised article."""))

        # Stream Arbiter's revised article to CENTER panel (replacing original)
        revised_article = ""
        revised_header = "📝 REVISED ARTICLE\n" + "=" * 42 + "\n\n"

        # Show "generating" in edit log
        generating_edit_log = "📝 EDIT LOG\n"
        generating_edit_log += "=" * 42 + "\n\n"
        generating_edit_log += "Generating revised article...\n\n"
        generating_edit_log += "Edit log will be computed once revision is complete."

        for response, chunk in arbiter_chat.stream():
            if chunk.content:
                revised_article += chunk.content
                streaming_center = revised_header + revised_article
                yield debate_transcript, streaming_center, generating_edit_log, None, None

        # Generate the edit log by comparing original and revised
        edit_log = generate_edit_log(original_for_diff, revised_article)

        # Add completion message to debate transcript
        debate_transcript += "✅ Debate complete! Revised article generated.\n"

        # Center panel: final revised article
        final_center = revised_header + revised_article

        # Build the article history entry
        final_article = f"📝 REVISED ARTICLE (Post-Debate)\n"
        final_article += "=" * 42 + "\n\n"
        final_article += revised_article

        # Save revised article to file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(revised_article)
        except Exception as e:
            print(f"Error saving revised article: {e}")
            filepath = None

        # Build state updates for main.py to apply
        state_updates = {
            'current_article_clean': revised_article,
            'article_history_entry': final_article
        }

        yield debate_transcript, final_center, edit_log, filepath, state_updates

    except Exception as e:
        error_transcript = debate_transcript + \
            f"\n\n❌ Error during debate: {str(e)}\n\nPlease try again."
        yield error_transcript, center_display, edit_log_placeholder, None, None
