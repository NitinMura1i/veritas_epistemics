# helpers.py
# UI helper functions and utilities

import re


def generate_edit_log(original: str, revised: str) -> str:
    """Generate a structured edit log comparing original and revised text.

    Uses section-aware sentence comparison to show what changed.
    """

    def truncate(text, max_len=140):
        """Truncate text to max_len with ellipsis."""
        text = text.replace('\n', ' ').strip()
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."

    def count_sources(text):
        """Count the number of sources in the text."""
        source_patterns = re.findall(r'\[[\^]?\d+\]:', text)
        if source_patterns:
            return len(source_patterns)
        if "## Sources" in text:
            sources_section = text.split("## Sources")[-1]
            lines = [l.strip() for l in sources_section.strip().split(
                '\n') if l.strip() and l.strip() != '## Sources']
            return len(lines)
        return 0

    def split_into_sections(text):
        """Split text into sections by ## headers."""
        sections = {}
        current_header = "intro"
        current_content = []

        for line in text.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections[current_header] = '\n'.join(current_content)
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_header] = '\n'.join(current_content)

        return sections

    def split_sentences(text):
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    # Count sources for summary
    original_source_count = count_sources(original)
    revised_source_count = count_sources(revised)

    # Split into sections
    orig_sections = split_into_sections(original)
    rev_sections = split_into_sections(revised)

    # Compare sentences within each section (preserve order from original)
    changes = []
    all_headers = list(orig_sections.keys())
    for h in rev_sections.keys():
        if h not in all_headers:
            all_headers.append(h)

    for header in all_headers:
        # Skip sources section
        if 'Sources' in header or 'sources' in header:
            continue

        orig_text = orig_sections.get(header, "")
        rev_text = rev_sections.get(header, "")

        orig_sentences = split_sentences(orig_text)
        rev_sentences = split_sentences(rev_text)

        # Compare by position within this section
        max_len = max(len(orig_sentences), len(rev_sentences))
        for i in range(max_len):
            orig_sent = orig_sentences[i] if i < len(orig_sentences) else ""
            rev_sent = rev_sentences[i] if i < len(rev_sentences) else ""

            # Skip if identical or both empty
            if orig_sent == rev_sent:
                continue
            # Skip if one is empty
            if not orig_sent or not rev_sent:
                continue

            changes.append((orig_sent, rev_sent))

    # Build edit log
    edit_log = "📝 EDIT LOG\n"
    edit_log += "=" * 44 + "\n\n"

    if changes:
        for orig, rev in changes:
            edit_log += "                 (Original)\n"
            edit_log += f"{truncate(orig)}\n\n"
            edit_log += "                    ↓\n\n"
            edit_log += "                 (Revised)\n"
            edit_log += f"{truncate(rev)}\n\n"
            edit_log += "-" * 44 + "\n\n"

    # Summary
    if changes:
        edit_log += f"Total: {len(changes)} edits\n"
    else:
        edit_log += "No significant changes detected.\n"

    # Note source changes if there are any
    if revised_source_count != original_source_count:
        edit_log += f"Sources expanded from {original_source_count} to {revised_source_count}\n"

    return edit_log
