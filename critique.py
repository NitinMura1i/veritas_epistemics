# critique.py
# Self-critique functionality

import os
from datetime import datetime

from config import client
from helpers import generate_edit_log
from xai_sdk.chat import user, system


def run_self_critique(current_article_clean: str, original_article: str):
    """Run self-critique with chain-of-thought analysis streaming.

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
        error_msg = "⚠️ No article to critique. Please generate an article first."
        return error_msg, "", "", None, None

    # Store original for comparison later
    original_for_diff = current_article_clean

    # Center panel: original article (will transition to refined at end)
    center_display = "📄 ORIGINAL ARTICLE\n"
    center_display += "=" * 44 + "\n\n"
    center_display += original_article

    # Left panel: critique analysis (streaming)
    critique_analysis = "💭 SELF-CRITIQUE ANALYSIS\n"
    critique_analysis += "=" * 44 + "\n\n"
    critique_analysis += "🔎 Analyzing article for epistemic issues...\n\n"

    # Right panel: edit log placeholder
    edit_log_placeholder = "📝 EDIT LOG\n"
    edit_log_placeholder += "=" * 44 + "\n\n"
    edit_log_placeholder += "Changes will appear here after critique completes.\n\n"
    edit_log_placeholder += "The refined article will be compared to the original, "
    edit_log_placeholder += "and this panel will show exactly what was changed."

    yield critique_analysis, center_display, edit_log_placeholder, None, None

    try:
        # Step 1: Generate critique analysis with streaming
        critique_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        critique_chat.append(system(
            "You are a professional epistemic analyst conducting a critical evaluation of the article. "
            "Provide a structured analysis identifying weaknesses in epistemic quality.\n\n"
            "CONSTRAINTS:\n"
            "- Maximum 300 words total\n"
            "- Identify 2-3 most critical issues\n"
            "- Use formal, professional language\n"
            "- Be direct and specific in identifying problems\n\n"
            "Format your analysis as numbered sections (1, 2, 3...) addressing the most critical issues found. "
            "Consider evaluating:\n"
            "- Source reliability and diversification\n"
            "- Claims requiring qualification or additional context\n"
            "- Potential biases or one-sided framing\n"
            "- Missing caveats or epistemic humility\n"
            "- Overstatements or unwarranted certainty\n\n"
            "Use clear, declarative statements rather than casual observations. "
            "Number your sections sequentially (1, 2, 3) based on the issues you identify."
        ))
        critique_chat.append(user(
            f"Analyze this article for epistemic issues:\n\n{current_article_clean}"
        ))

        # Stream the critique analysis
        full_critique = ""
        for response, chunk in critique_chat.stream():
            if chunk.content:
                full_critique += chunk.content
                critique_analysis_display = "💭 SELF-CRITIQUE ANALYSIS\n"
                critique_analysis_display += "=" * 44 + "\n\n"
                critique_analysis_display += full_critique
                yield critique_analysis_display, center_display, edit_log_placeholder, None, None

        # Step 2: Generate refined article based on critique
        critique_analysis_display = "💭 SELF-CRITIQUE ANALYSIS\n"
        critique_analysis_display += "=" * 44 + "\n\n"
        critique_analysis_display += full_critique + "\n\n"
        critique_analysis_display += "✓ Analysis complete. Generating refined article...\n"
        yield critique_analysis_display, center_display, edit_log_placeholder, None, None

        refinement_chat = client.chat.create(model="grok-4-1-fast-reasoning")
        refinement_chat.append(system(
            "You are refining an article based on epistemic critique. Your task:\n"
            "- Address all valid concerns raised in the critique\n"
            "- Add appropriate qualifiers and nuance\n"
            "- Strengthen unsupported claims with evidence or soften them\n"
            "- Maintain factual accuracy and cite sources\n"
            "- Keep the same structure and ~300 word length\n"
            "Output ONLY the final refined article in markdown format with inline citations and Sources section."
        ))
        refinement_chat.append(user(
            f"Original article:\n{current_article_clean}\n\n"
            f"Critique analysis:\n{full_critique}\n\n"
            f"Produce the refined article addressing these concerns."
        ))

        # Show generating status in edit log
        generating_edit_log = "📝 EDIT LOG\n"
        generating_edit_log += "=" * 44 + "\n\n"
        generating_edit_log += "Generating refined article...\n\n"
        generating_edit_log += "Edit log will be computed once refinement is complete."

        # Stream refined article to CENTER panel (replacing original)
        refined_header = "📝 REFINED ARTICLE\n" + "=" * 44 + "\n\n"
        refined_article = ""
        for response, chunk in refinement_chat.stream():
            if chunk.content:
                refined_article += chunk.content
                streaming_center = refined_header + refined_article
                yield critique_analysis_display, streaming_center, generating_edit_log, None, None

        # Generate the edit log by comparing original and revised
        edit_log = generate_edit_log(original_for_diff, refined_article)

        # Build the article history entry
        final_article = f"📝 REFINED ARTICLE (Post-Critique)\n"
        final_article += "=" * 44 + "\n\n"
        final_article += f"__Epistemically refined through self-critique__\n\n---\n\n{refined_article}"

        # Save refined article to file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"critique_{timestamp}.md"
        filepath = os.path.abspath(filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(refined_article)
        except Exception as e:
            print(f"Error saving refined article: {e}")
            filepath = None

        # Build state updates for main.py to apply
        state_updates = {
            'current_article_clean': refined_article,
            'article_history_entry': final_article
        }

        # Final yield with complete versions
        final_critique_display = "💭 SELF-CRITIQUE ANALYSIS\n"
        final_critique_display += "=" * 44 + "\n\n"
        final_critique_display += full_critique + "\n\n"
        final_critique_display += "✅ Critique complete! Refined article generated."

        final_center = refined_header + refined_article
        yield final_critique_display, final_center, edit_log, filepath, state_updates

    except Exception as e:
        error_analysis = critique_analysis + \
            f"\n\n❌ Error during critique: {str(e)}\n\nPlease try again."
        yield error_analysis, center_display, edit_log_placeholder, None, None
