# article.py
# Article generation functions

import os
import re
import time
import requests
from typing import Dict, Optional
from datetime import datetime
from autocorrect import Speller

from config import client
from xai_sdk.chat import user, system
from xai_sdk.tools import web_search


def fetch_wikipedia(topic: str) -> Optional[Dict[str, str]]:
    """Fetch Wikipedia article content for a given topic."""
    try:
        # Spell-check the topic before searching
        spell = Speller(lang='en')
        corrected_topic = spell(topic)

        # Wikipedia API search endpoint with required headers
        search_url = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "Veritas-Epistemics/1.0 (Educational Project; claude-code@anthropic.com)"
        }

        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": corrected_topic,
            "srlimit": 1
        }
        search_response = requests.get(
            search_url, params=search_params, headers=headers, timeout=10)
        search_data = search_response.json()

        if not search_data.get("query", {}).get("search"):
            return None

        # Get the page title
        page_title = search_data["query"]["search"][0]["title"]

        # Fetch page content
        content_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": True
        }
        content_response = requests.get(
            search_url, params=content_params, headers=headers, timeout=10)
        content_data = content_response.json()

        pages = content_data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        extract = pages[page_id].get("extract", "")

        return {
            "title": page_title,
            "content": extract,
            "url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        }
    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")
        return None


def fetch_content_from_url(url: str) -> Optional[Dict[str, str]]:
    """Fetch content from a URL. Supports Wikipedia URLs via API."""
    try:
        if "wikipedia.org/wiki/" in url:
            from urllib.parse import unquote
            title = url.split("/wiki/")[-1].replace("_", " ")
            title = unquote(title)

            api_url = "https://en.wikipedia.org/w/api.php"
            headers = {
                "User-Agent": "Veritas-Epistemics/1.0 (Educational Project)"}
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True
            }
            response = requests.get(
                api_url, params=params, headers=headers, timeout=10)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page_id = list(pages.keys())[0]

            if page_id == "-1":
                return {"title": title, "content": "", "url": url}

            extract = pages[page_id].get("extract", "")
            actual_title = pages[page_id].get("title", title)

            return {"title": actual_title, "content": extract, "url": url}
        else:
            return {"title": url, "content": "", "url": url}
    except Exception as e:
        print(f"Error fetching content from URL: {e}")
        return {"title": url, "content": "", "url": url}


def generate_initial_article(topic: str):
    """Generate initial article with Wikipedia-first optimization (web search fallback).

    Yields: (article_display, status_log, sources_display, filepath, state_updates)

    state_updates is None during streaming, and on final yield contains:
    {
        'article_history_entry': str,  # Entry to append to article_history
        'current_sources': list,       # Updated current_sources
        'current_article_clean': str,  # Clean article content
        'current_grounding_context': str,  # Source context for debates
        'original_article': str        # Original article content
    }
    """
    # Status in left panel
    status_log = "🔍 PROCESS LOG\n"
    status_log += "=" * 42 + "\n\n"
    status_log += "🚀 Starting article generation...\n\n"

    # Show loading message in center panel
    article_placeholder = "📝 YOUR ARTICLE\n" + "=" * 42 + \
        "\n\nYour generated article will appear shortly..."

    # Show loading message in right panel
    loading_message = "📚 SOURCE MATERIAL\n" + "=" * 42 + \
        "\n\nSearching for sources...\n\nSources will appear here when article generation is complete."

    yield article_placeholder, status_log, loading_message, None, None

    # Initialize variables
    source_notes = []
    sources_list = []
    source_context = ""
    streaming_sources_display = ""
    use_wikipedia = False
    wiki_data = None

    # Step 1: Try Wikipedia first (fast path)
    status_log += "🔎 Searching for sources...\n\n"
    yield article_placeholder, status_log, loading_message, None, None

    wiki_data = fetch_wikipedia(topic)  # This API call IS the natural delay

    if wiki_data:
        # Check match quality - calculate word overlap
        topic_words = set(topic.lower().split())
        title_words = set(wiki_data['title'].lower().split())

        # Remove common words that don't indicate topic match
        stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at',
                      'to', 'for', 'and', 'or', 'is', 'are', 'was', 'were'}
        topic_words = topic_words - stop_words
        title_words = title_words - stop_words

        if topic_words:
            overlap = len(topic_words & title_words)
            match_ratio = overlap / len(topic_words)
        else:
            match_ratio = 0

        # Also check for exact match (case-insensitive)
        exact_match = topic.lower().strip(
        ) == wiki_data['title'].lower().strip()

        if exact_match or match_ratio >= 0.7:
            use_wikipedia = True
            status_log += f"🎯 Found: '{wiki_data['title']}'\n\n"
        else:
            status_log += f"⚠️ No exact match found\n\n"
    else:
        status_log += "⚠️ No direct source found\n\n"

    yield article_placeholder, status_log, loading_message, None, None

    # Path A: Use Wikipedia (fast path)
    if use_wikipedia and wiki_data:
        # Set up source display
        streaming_sources_display = "📚 SOURCE MATERIAL\n"
        streaming_sources_display += "=" * 42 + "\n\n"
        streaming_sources_display += f"**{wiki_data['title']}**\n\n"
        streaming_sources_display += f"Source: {wiki_data['url']}\n\n"
        streaming_sources_display += "---\n\n"
        streaming_sources_display += f"{wiki_data['content'][:2500]}"

        source_context = f"Wikipedia Article: {wiki_data['title']}\n\n{wiki_data['content'][:3000]}"
        source_notes.append(
            f"[Wikipedia: {wiki_data['title']}]({wiki_data['url']})")
        sources_list = [{"name": wiki_data['title'], "type": "Wikipedia",
                        "url": wiki_data['url'], "content": wiki_data['content']}]

        status_log += "📝 Generating article...\n\n"
        yield article_placeholder, status_log, streaming_sources_display, None, None

        try:
            wiki_chat = client.chat.create(model="grok-4-1-fast")

            wiki_chat.append(system(
                "You are an expert knowledge synthesizer focused on epistemic integrity. "
                "Write factual, well-sourced articles with inline citations. "
                "Be clear about certainty levels and avoid speculation."
            ))

            wiki_prompt = f"""Write a comprehensive, factual article about "{topic}".

Use the following Wikipedia content as your primary source:

{source_context}

Requirements:
1. Write in encyclopedic style (factual, neutral, well-structured)
2. Include inline citations like [1] when referencing facts
3. Add a "## Sources" section at the end listing all references
4. Be clear about certainty levels - use phrases like "evidence suggests", "widely accepted", etc.
5. Length: EXACTLY 300 words (excluding the Sources section)
6. Structure: Introduction, 2-3 main sections, conclusion, sources

Format the article in clean markdown."""

            wiki_chat.append(user(wiki_prompt))

            article_content = ""
            for response, chunk in wiki_chat.stream():
                if chunk.content:
                    article_content += chunk.content

                    streaming_article = "📝 YOUR ARTICLE\n"
                    streaming_article += "=" * 42 + "\n\n"
                    streaming_article += article_content

                    yield streaming_article, status_log, streaming_sources_display, None, None

        except Exception as e:
            status_log += f"❌ Error generating article: {str(e)}\n\n"
            error_article = "📝 YOUR ARTICLE\n" + "=" * 42 + \
                "\n\n❌ Error generating article. Please try again."
            yield error_article, status_log, streaming_sources_display, None, None
            return

    # Path B: Use web search (slower path for specific/niche topics)
    else:
        status_log += "🌐 Using web search for specific content...\n\n"
        yield article_placeholder, status_log, loading_message, None, None

        try:
            chat = client.chat.create(
                model="grok-4-1-fast",
                include=["inline_citations"],
                tools=[web_search()],
            )

            chat.append(system(
                "You are an expert knowledge synthesizer focused on epistemic integrity. "
                "Search the web for ONE authoritative source about the topic, then write a factual, "
                "well-sourced article with inline citations. Use exactly 1 high-quality source. "
                "Be clear about certainty levels and avoid speculation."
            ))

            prompt = f"""Search the web for authoritative information about "{topic}", then write a comprehensive, factual article.

Requirements:
1. Search for and use exactly 1 high-quality, authoritative source (prefer academic sites, reputable news, official sources)
2. Write in encyclopedic style (factual, neutral, well-structured)
3. Include inline citations referencing your source
4. Add a "## Sources" section at the end listing the reference with URL
5. Be clear about certainty levels - use phrases like "evidence suggests", "widely accepted", etc.
6. Length: EXACTLY 300 words (excluding the Sources section)
7. Structure: Introduction, 2-3 main sections, conclusion, sources

Format the article in clean markdown."""

            chat.append(user(prompt))

            status_log += "📡 Searching web sources...\n\n"
            yield article_placeholder, status_log, loading_message, None, None

            article_content = ""
            final_response = None
            first_chunk = True
            start_time = time.time()
            update_messages = [
                (5, "🔍 Finding relevant sources..."),
                (10, "📖 Reading source content..."),
                (15, "✨ Preparing article structure..."),
            ]
            next_update_idx = 0

            for response, chunk in chat.stream():
                final_response = response

                # Time-based updates while waiting for content
                elapsed = time.time() - start_time
                if first_chunk and next_update_idx < len(update_messages):
                    threshold, message = update_messages[next_update_idx]
                    if elapsed >= threshold:
                        status_log += f"{message}\n\n"
                        yield article_placeholder, status_log, loading_message, None, None
                        next_update_idx += 1

                if chunk.content:
                    if first_chunk:
                        status_log += "📝 Writing article...\n\n"
                        first_chunk = False

                    article_content += chunk.content
                    streaming_article = "📝 YOUR ARTICLE\n"
                    streaming_article += "=" * 42 + "\n\n"
                    streaming_article += article_content
                    yield streaming_article, status_log, loading_message, None, None

            # Extract cited URLs
            cited_urls = re.findall(
                r'\]\((https?://[^\)]+)\)', article_content)
            seen = set()
            unique_cited_urls = []
            for url in cited_urls:
                if url not in seen:
                    seen.add(url)
                    unique_cited_urls.append(url)

            citations = unique_cited_urls[:1] if unique_cited_urls else []
            if not citations and final_response and hasattr(final_response, 'citations') and final_response.citations:
                citations = list(final_response.citations)[:1]

            if citations:
                status_log += "📄 Processing source...\n\n"
                yield streaming_article, status_log, loading_message, None, None

                url = citations[0]
                source_data = fetch_content_from_url(url)

                streaming_sources_display = "📚 WEB SOURCE\n"
                streaming_sources_display += "=" * 42 + "\n\n"

                if source_data and source_data.get("content"):
                    content = source_data["content"]
                    words = content.split()
                    truncated_content = " ".join(words[:250])
                    if len(words) > 250:
                        truncated_content += "..."

                    streaming_sources_display += f"Source: {url}\n\n"
                    streaming_sources_display += "-" * 42 + "\n\n"
                    streaming_sources_display += truncated_content

                    source_notes.append(f"[{source_data['title']}]({url})")
                    sources_list.append({
                        "name": source_data["title"],
                        "type": "Web",
                        "url": url,
                        "content": content
                    })
                    source_context = f"Web Source: {source_data['title']}\n\n{content[:3000]}"
                else:
                    streaming_sources_display += f"**{url}**\n\n"
                    streaming_sources_display += "(Content preview not available for this source)"
                    source_notes.append(f"[Source]({url})")
                    sources_list.append(
                        {"name": url, "type": "Web", "url": url})
                    source_context = f"Web Source: {url}"
            else:
                status_log += "⚠️ No citations found in response\n\n"
                streaming_sources_display = "📚 SOURCE MATERIAL\n" + "=" * 42 + \
                    "\n\n⚠️ No sources found.\n\nArticle generated from general knowledge."

        except Exception as e:
            status_log += f"❌ Web search failed: {str(e)}\n\n"
            error_article = "📝 YOUR ARTICLE\n" + "=" * 42 + \
                "\n\n❌ Error generating article. Please try again."
            yield error_article, status_log, loading_message, None, None
            return

    # Add header at the top
    final_article = "📝 YOUR ARTICLE\n"
    final_article += "=" * 42 + "\n\n"
    final_article += article_content

    # Update status log to show completion
    status_log += "✅ Article generation complete!\n\n"

    # Save article to file for download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[<>:"/\\|?*]', '', topic)
    safe_topic = safe_topic.replace(' ', '_')[:50]
    filename = f"article_{safe_topic}_{timestamp}.md"
    filepath = os.path.abspath(filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_content)
    except Exception as e:
        print(f"Error saving article: {e}")
        filepath = None

    # Build state updates for main.py to apply
    state_updates = {
        'article_history_entry': final_article,
        'current_sources': sources_list,
        'current_article_clean': article_content,
        'current_grounding_context': source_context,
        'original_article': article_content
    }

    # Yield final result with filepath and state updates
    yield final_article, status_log, streaming_sources_display, filepath, state_updates
