import os
from dotenv import load_dotenv
import wikipediaapi

from xai_sdk.chat import user, system
from xai_sdk import Client

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)

wiki = wikipediaapi.Wikipedia(
    user_agent="Veritas_Epistemics/0.1 (nitinmurali.03@gmail.com; contact vie GitHub)",
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
        context += f"\n\n ({page.sections[0].title}):\n{page.sections[0].text[:400]}"

    return context.strip()


topic = "Sacred Geometry"

wiki_context = get_wikipedia_context(topic)
print("\nRetrieved Wikipedia Context (for grounding):\n")
preview = wiki_context[:500] + \
    "..." if len(wiki_context) > 500 else wiki_context
print(preview)

chat = client.chat.create(model="grok-4")

chat.append(system(
    "You are a truthful, accurate encyclopedia writer focused on maximum truth-seeking. "
    "Cite sources where possible and avoid speculation."))

chat.append(user(f"""
                 You are writing an encyclopedia article on '{topic}'.
                 
                 Here is grounding context from Wikipedia to ensure factual accuracy:
                 {wiki_context}

                 Write a clear, factual 300-word encyclopedia-style article on '{topic}'.
                 Be precise, include key facts, and note any major sources or references.
                 Ground your content in the provided Wikipedia context where possible.
                 
                 For every factual claim or key piece of information, include an inline citation in the form [1], [2], etc. 
                 At the end of the article, list the references (e.g., [1] Wikipedia article on Machine Learning: https://en.wikipedia.org/wiki/Machine_learning). 
                 Use [1] for the Wikipedia URL, and [2], [3], etc. for any additional sources you reference.
                 """))

response = chat.sample()
article = response.content.strip()

print("Generated Article:\n")
print(article)

# ---------CRITIQUE SECTION---------

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

print("\n" + "="*60)
print("Critique / Self-Check (Epistemics Step 1):")
print("="*60)
print(critique)

# ---------SELF-REFINEMENT LOOP ---------

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

print("\n" + "="*60)
print("Refined Article (After Self-Critique):")
print("="*60)
print(refined_article)
