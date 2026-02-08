# synthetic.py
# Synthetic data generation with Ray parallel processing

import os
import re
import json
import datetime
import ray
from xai_sdk import Client
from xai_sdk.chat import user, system

from config import api_key, client


# ============================================================================
# RAY PARALLEL PROCESSING FOR SYNTHETIC DATA GENERATION
# ============================================================================

@ray.remote
def generate_single_example(api_key: str, topic: str, example_id: int, quality: str, target_flaw: str, word_range: str):
    """
    Ray remote function to generate and label a single synthetic example.
    Runs in parallel with other examples for faster batch generation.

    Args:
        api_key: xAI API key (needed because Ray runs in separate process)
        topic: The topic to generate an article about
        example_id: The ID number for this example
        quality: Target quality tier (excellent, good, fair, poor, terrible)
        target_flaw: The type of epistemic flaw to include
        word_range: Word count range for the article

    Returns:
        dict: Generated article with labels and metadata
    """
    import datetime
    import re
    from xai_sdk import Client
    from xai_sdk.chat import user, system

    # Create client inside Ray worker (can't share client across processes)
    worker_client = Client(api_key=api_key, timeout=3600)

    # Build quality instruction based on target quality
    if quality == "excellent":
        quality_instruction = """Generate an epistemically EXCELLENT article (should score 9-10 on all dimensions):
- Use precise hedging language ("suggests", "indicates", "may", "appears to") for uncertain claims
- Include specific citations with sources for ALL major claims (e.g., "According to [Source, Year]...")
- Present multiple perspectives on any debatable points
- Explicitly distinguish between established facts and ongoing research/speculation
- Include caveats about data limitations or methodological constraints
- Use measured, qualified language throughout - no absolute statements without ironclad evidence
- Include a Sources/References section with specific attributions"""
    elif quality == "good":
        quality_instruction = """Generate a GOOD quality article (target 6-8 scores):
- Include SPECIFIC citations for most claims (e.g., "According to [Source, Year]...")
- Use appropriate hedging language for most statements ("suggests", "indicates", "research shows")
- Allow only 1-2 minor lapses: one claim without citation OR one slightly promotional phrase
- Maintain neutral, informative tone overall
- Present facts objectively without one-sided framing
- Include a proper Sources section with multiple references"""
    elif quality == "fair":
        quality_instruction = """Generate a FAIR quality article (target 4-6 scores):
- Include SOME specific citations, but leave several significant claims unsourced
- Mix hedging with definitive statements
- Attempt balance but lean slightly positive/promotional
- Include basic sourcing attempts even if vague
- Should feel like a decent Wikipedia article with noticeable gaps"""
    elif quality == "poor":
        quality_instruction = """Generate an article with significant epistemic problems:
- Many claims lack sources or citations
- Overconfident language throughout ("X is", "Y proves", etc.)
- Missing important qualifiers and hedges
- One-sided framing on debatable topics
- Presents speculation as fact in multiple places"""
    else:  # terrible
        quality_instruction = """Generate an article with severe epistemic failures:
- Almost no citations or sources provided
- Extreme overconfidence and absolutist language
- No hedging or uncertainty acknowledgment
- Heavily biased framing
- Speculation presented as definitive fact"""

    # Generate the article
    generator_chat = worker_client.chat.create(model="grok-4-1-fast-reasoning")
    generator_chat.append(system(
        f"You are generating a synthetic training example for an epistemic quality classifier.\n\n"
        f"{quality_instruction}\n\n"
        f"Write a {word_range} word article about the given topic with the specified epistemic characteristics. "
        f"Output ONLY the article text in markdown format. Include a brief Sources section if appropriate."
    ))
    generator_chat.append(user(f"Generate an article about: {topic}"))

    generated_article = ""
    for response, chunk in generator_chat.stream():
        if chunk.content:
            generated_article += chunk.content

    # Label the article
    labeler_chat = worker_client.chat.create(model="grok-4-1-fast-reasoning")
    labeler_chat.append(system(
        "You are an epistemic quality labeler. Analyze the article and provide scores (0-10) for:\n"
        "- source_quality: How well claims are sourced and cited\n"
        "- certainty_appropriateness: Whether certainty language matches evidence strength\n"
        "- bias_level: How balanced vs biased the framing is (10 = very balanced, 0 = very biased)\n"
        "- completeness: Whether important caveats and limitations are mentioned\n\n"
        "Format your response as:\n"
        "SOURCE_QUALITY: [0-10]\n"
        "CERTAINTY: [0-10]\n"
        "BIAS: [0-10]\n"
        "COMPLETENESS: [0-10]\n"
        "FLAWS: [comma-separated list of specific issues]"
    ))
    labeler_chat.append(user(f"Label this article:\n\n{generated_article}"))

    label_response = ""
    for response, chunk in labeler_chat.stream():
        if chunk.content:
            label_response += chunk.content

    # Parse labels
    source_quality = int(re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response).group(
        1)) if re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response) else 5
    certainty = int(re.search(r'CERTAINTY:\s*(\d+)', label_response).group(1)
                    ) if re.search(r'CERTAINTY:\s*(\d+)', label_response) else 5
    bias = int(re.search(r'BIAS:\s*(\d+)', label_response).group(1)
               ) if re.search(r'BIAS:\s*(\d+)', label_response) else 5
    completeness = int(re.search(r'COMPLETENESS:\s*(\d+)', label_response).group(1)
                       ) if re.search(r'COMPLETENESS:\s*(\d+)', label_response) else 5
    flaws_match = re.search(r'FLAWS:\s*(.+)', label_response)
    flaws = flaws_match.group(1).strip() if flaws_match else "none identified"

    return {
        "id": example_id,
        "topic": topic,
        "article": generated_article,
        "target_quality": quality,
        "target_flaw": target_flaw,
        "labels": {
            "source_quality": source_quality,
            "certainty_appropriateness": certainty,
            "bias_balance": bias,
            "completeness": completeness
        },
        "identified_flaws": flaws,
        "generated_at": datetime.datetime.now().isoformat()
    }


def run_synthetic_data_generation(topic, num_examples, quality_dist, flaw_type, article_length):
    """Generate synthetic training data with controlled epistemic properties."""

    # Validate topic is not empty
    if not topic or topic.strip() == "":
        error_msg = "⚠️ ERROR: Please enter a topic before generating synthetic data."
        yield error_msg, "", "", None
        return

    # Comprehensive topic validation
    topic_clean = topic.strip()

    # Check minimum length
    if len(topic_clean) < 2:
        error_msg = "⚠️ ERROR: Topic must be at least 2 characters long.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Check for alphabetic content (reject pure emojis, punctuation, numbers)
    alphabetic_chars = re.findall(r'[a-zA-Z]', topic_clean)
    if len(alphabetic_chars) < 2:
        error_msg = "⚠️ ERROR: Topic must contain at least 2 letters.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Check for variety (reject single repeated character like "aaaa" or "kkkk")
    if len(set(alphabetic_chars)) < 2:
        error_msg = "⚠️ ERROR: Topic must contain more than one unique letter.\n\nPlease enter a meaningful topic (e.g., 'AI', 'Climate Change', 'Quantum Computing')."
        yield error_msg, "", "", None
        return

    # Convert num_examples from string to int
    num_examples = int(num_examples)

    # Map article length to word count range
    if article_length.startswith("Brief"):
        word_range = "75-100"
    elif article_length.startswith("Standard"):
        word_range = "175-200"
    elif article_length.startswith("Long"):
        word_range = "275-300"
    else:
        word_range = "175-200"  # Default to Standard

    # Convert quality distribution selection to quality tiers list
    if quality_dist == "All":
        quality_tiers = ["excellent", "good", "fair", "poor", "terrible"]
    elif quality_dist == "Excellent":
        quality_tiers = ["excellent"]
    elif quality_dist == "Good":
        quality_tiers = ["good"]
    elif quality_dist == "Poor":
        quality_tiers = ["poor"]
    elif quality_dist == "Terrible":
        quality_tiers = ["terrible"]
    else:
        quality_tiers = ["excellent", "good", "fair", "poor", "terrible"]

    # Left panel: Generation log
    log = "🏭 SYNTHETIC DATA GENERATION\n"
    log += "=" * 44 + "\n\n"
    log += f"📊 Configuration:\n"
    log += f"- Target topic: {topic}\n"
    log += f"- Number of examples: {num_examples}\n"
    log += f"- Quality distribution: {quality_dist}\n"
    log += f"- Flaw type: {flaw_type}\n"
    log += f"- Article length: {article_length} ({word_range} words)\n\n"
    log += "⏳ Starting generation process...\n\n"

    yield "", log, "", None

    # Storage for generated data
    synthetic_dataset = []

    # Center panel: Article preview (will update with each generation)
    center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"

    # Right panel: Metadata accumulation
    metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"

    try:
        # ================================================================
        # PARALLEL PROCESSING WITH RAY (for 3+ examples)
        # Ray is initialized at app startup for cleaner operation
        # ================================================================
        if num_examples >= 3:
            log += f"⚡ Using parallel processing for {num_examples} examples...\n\n"
            yield center_preview, log, metadata_display, None

            # Prepare all example configurations
            example_configs = []
            for i in range(num_examples):
                quality = quality_tiers[i % len(quality_tiers)]

                if quality == "excellent":
                    target_flaw = "none"
                elif flaw_type == "Auto":
                    flaw_map = {
                        "good": "minor_overcertainty",
                        "fair": "missing_citations",
                        "poor": "multiple_flaws",
                        "terrible": "severe_bias_and_overstatements"
                    }
                    target_flaw = flaw_map.get(quality, "none")
                elif flaw_type == "Citations":
                    target_flaw = "missing_citations"
                elif flaw_type == "Certainty":
                    target_flaw = "overcertainty"
                elif flaw_type == "Bias":
                    target_flaw = "biased_framing"
                elif flaw_type == "Multiple":
                    target_flaw = "multiple_flaws"
                else:
                    target_flaw = "auto"

                example_configs.append((i + 1, quality, target_flaw))

            # Launch all examples in parallel using Ray
            log += f"🚀 Launching {num_examples} parallel generation tasks...\n\n"

            # Show friendly messages in center and right panels while generating
            center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
            center_preview += "**Your synthetic data is being generated...**\n\n"
            center_preview += f"Generating {num_examples} examples in parallel.\n"
            center_preview += "This may take a moment. Results will appear here shortly.\n"

            metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"
            metadata_display += "**Awaiting results...**\n\n"
            metadata_display += "Quality scores and flaw analysis\n"
            metadata_display += "will appear here once complete.\n"

            yield center_preview, log, metadata_display, None

            futures = [
                generate_single_example.remote(
                    api_key, topic, ex_id, quality, flaw, word_range)
                for ex_id, quality, flaw in example_configs
            ]

            # Wait for all to complete and collect results
            log += f"⏳ Generating all examples simultaneously...\n\n"
            yield center_preview, log, metadata_display, None

            synthetic_dataset = ray.get(futures)

            # Sort by ID to maintain order
            synthetic_dataset.sort(key=lambda x: x["id"])

            log += f"✅ All {num_examples} examples generated successfully!\n\n"

            # Build final displays
            center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
            for entry in synthetic_dataset:
                center_preview += f"**Example {entry['id']}/{num_examples}** (Quality: {entry['target_quality'].upper()})\n"
                center_preview += "-" * 44 + "\n\n"
                center_preview += entry['article'] + "\n\n"
                center_preview += "=" * 44 + "\n\n"

            metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"
            for entry in synthetic_dataset:
                metadata_display += f"**Example {entry['id']}** - {entry['target_quality'].upper()}\n"
                metadata_display += f"Source Quality: {entry['labels']['source_quality']}/10\n"
                metadata_display += f"Certainty Appropriateness: {entry['labels']['certainty_appropriateness']}/10\n"
                metadata_display += f"Bias Balance: {entry['labels']['bias_balance']}/10\n"
                metadata_display += f"Completeness: {entry['labels']['completeness']}/10\n"
                metadata_display += f"Flaws: {entry['identified_flaws']}\n\n"

            yield center_preview, log, metadata_display, None

        # ================================================================
        # SEQUENTIAL PROCESSING (for 1-2 examples, with live streaming)
        # ================================================================
        else:
            for i in range(num_examples):
                # Select quality tier and specific flaw for this example
                quality = quality_tiers[i % len(quality_tiers)]

                # Determine target flaw based on flaw_type selection
                # Excellent quality always has no flaws, regardless of flaw_type
                if quality == "excellent":
                    target_flaw = "none"
                elif flaw_type == "Auto":
                    # Map quality to specific epistemic flaws automatically
                    flaw_map = {
                        "good": "minor_overcertainty",
                        "fair": "missing_citations",
                        "poor": "multiple_flaws",
                        "terrible": "severe_bias_and_overstatements"
                    }
                    target_flaw = flaw_map.get(quality, "none")
                elif flaw_type == "Citations":
                    target_flaw = "missing_citations"
                elif flaw_type == "Certainty":
                    target_flaw = "overcertainty"
                elif flaw_type == "Bias":
                    target_flaw = "biased_framing"
                elif flaw_type == "Multiple":
                    target_flaw = "multiple_flaws"
                else:
                    target_flaw = "auto"

                # Update log
                log += f"📝 Generating example {i+1}/{num_examples}...\n"
                log += f"   Quality tier: {quality}\n"
                log += f"   Target flaw: {target_flaw}\n\n"
                yield center_preview, log, metadata_display, None

                # Generate article with controlled epistemic quality
                generator_chat = client.chat.create(
                    model="grok-4-1-fast-reasoning")

                # Craft prompt based on target quality and specific flaw type
                if quality == "excellent":
                    # Excellent articles are always truly excellent, ignore flaw_type
                    quality_instruction = """Generate an epistemically EXCELLENT article (should score 9-10 on all dimensions):
- Use precise hedging language ("suggests", "indicates", "may", "appears to") for uncertain claims
- Include specific citations with sources for ALL major claims (e.g., "According to [Source, Year]...")
- Present multiple perspectives on any debatable points
- Explicitly distinguish between established facts and ongoing research/speculation
- Include caveats about data limitations or methodological constraints
- Use measured, qualified language throughout - no absolute statements without ironclad evidence
- Include a Sources/References section with specific attributions"""
                elif quality == "good":
                    quality_instruction = """Generate a GOOD quality article (target 6-8 scores):
- Include SPECIFIC citations for most claims (e.g., "According to [Source, Year]...")
- Use appropriate hedging language for most statements ("suggests", "indicates", "research shows")
- Allow only 1-2 minor lapses: one claim without citation OR one slightly promotional phrase ("vibrant", "boasts")
- Maintain neutral, informative tone overall - avoid multiple promotional words
- Present facts objectively without one-sided framing
- Include a proper Sources section with multiple references
- The article should feel solid and informative, just not quite perfect"""
                elif quality == "fair":
                    quality_instruction = """Generate a FAIR quality article (target 4-6 scores):
- Include SOME specific citations, but leave several significant claims unsourced
- Mix hedging with definitive statements - use "may" and "suggests" sometimes, but also use "is" and "proves" for uncertain claims
- Attempt balance but lean slightly positive/promotional without being one-sided propaganda
- Include basic sourcing attempts ("studies show", "researchers say") even if vague
- Include a Sources section but it can be incomplete or vague
- Should feel like a decent Wikipedia article with noticeable gaps"""
                elif quality == "poor":
                    quality_instruction = """Generate an article with significant epistemic problems:
- Many claims lack sources or citations
- Overconfident language throughout ("X is", "Y proves", etc.)
- Missing important qualifiers and hedges
- One-sided framing on debatable topics
- Presents speculation as fact in multiple places"""
                else:  # terrible
                    quality_instruction = """Generate an article with severe epistemic failures:
- Almost no citations or sources provided
- Extreme overconfidence and absolutist language
- No hedging or uncertainty acknowledgment
- Heavily biased framing
- Speculation presented as definitive fact
- Makes strong causal claims without evidence"""

                # Add flaw-specific emphasis only for non-excellent quality tiers
                if quality != "excellent" and flaw_type != "Auto":
                    if flaw_type == "Citations":
                        quality_instruction += "\n\nIMPORTANT: Focus especially on citation/sourcing issues. Make claims without proper attribution or sources."
                    elif flaw_type == "Certainty":
                        quality_instruction += "\n\nIMPORTANT: Focus especially on certainty language issues. Use overconfident language and avoid appropriate hedging."
                    elif flaw_type == "Bias":
                        quality_instruction += "\n\nIMPORTANT: Focus especially on biased framing. Present one-sided perspectives and miss important counterarguments."
                    elif flaw_type == "Multiple":
                        quality_instruction += "\n\nIMPORTANT: Include multiple types of epistemic flaws (citations, certainty, bias) throughout the article."

                generator_chat.append(system(
                    f"You are generating a synthetic training example for an epistemic quality classifier.\n\n"
                    f"{quality_instruction}\n\n"
                    f"Write a {word_range} word article about the given topic with the specified epistemic characteristics. "
                    f"Output ONLY the article text in markdown format. Include a brief Sources section if appropriate."
                ))
                generator_chat.append(user(f"Generate an article about: {topic}"))

                # Generate article with streaming updates
                generated_article = ""
                for response, chunk in generator_chat.stream():
                    if chunk.content:
                        generated_article += chunk.content

                        # Rebuild center preview with streaming article
                        temp_center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
                        # Show all previously completed articles
                        for entry in synthetic_dataset:
                            temp_center_preview += f"**Example {entry['id']}/{num_examples}** (Quality: {entry['target_quality'].upper()})\n"
                            temp_center_preview += "-" * 44 + "\n\n"
                            temp_center_preview += entry['article'] + "\n\n"
                            temp_center_preview += "=" * 44 + "\n\n"
                        # Show currently streaming article
                        temp_center_preview += f"**Example {i+1}/{num_examples}** (Quality: {quality.upper()}) - GENERATING...\n"
                        temp_center_preview += "-" * 44 + "\n\n"
                        temp_center_preview += generated_article + \
                            "▌"  # Add cursor to show it's streaming

                        yield temp_center_preview, log, metadata_display, None

                # Label the article with epistemic scores
                labeler_chat = client.chat.create(model="grok-4-1-fast-reasoning")
                labeler_chat.append(system(
                    "You are an epistemic quality labeler. Analyze the article and provide scores (0-10) for:\n"
                    "- source_quality: How well claims are sourced and cited\n"
                    "- certainty_appropriateness: Whether certainty language matches evidence strength\n"
                    "- bias_level: How balanced vs biased the framing is (10 = very balanced, 0 = very biased)\n"
                    "- completeness: Whether important caveats and limitations are mentioned\n\n"
                    "Evaluate objectively using the full 0-10 scale:\n"
                    "- 9-10: Near-perfect epistemic quality\n"
                    "- 7-8: Strong quality with minor issues\n"
                    "- 5-6: Adequate with noticeable problems\n"
                    "- 3-4: Significant issues but some structure\n"
                    "- 1-2: Severe problems, minimal value\n"
                    "- 0: Complete epistemic failure\n\n"
                    "Also identify specific epistemic flaws present.\n\n"
                    "Format your response as:\n"
                    "SOURCE_QUALITY: [0-10]\n"
                    "CERTAINTY: [0-10]\n"
                    "BIAS: [0-10]\n"
                    "COMPLETENESS: [0-10]\n"
                    "FLAWS: [comma-separated list of specific issues]"
                ))
                labeler_chat.append(
                    user(f"Label this article:\n\n{generated_article}"))

                # Get labels
                label_response = ""
                for response, chunk in labeler_chat.stream():
                    if chunk.content:
                        label_response += chunk.content

                # Parse labels
                source_quality = int(re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response).group(
                    1)) if re.search(r'SOURCE_QUALITY:\s*(\d+)', label_response) else 5
                certainty = int(re.search(r'CERTAINTY:\s*(\d+)', label_response).group(1)
                                ) if re.search(r'CERTAINTY:\s*(\d+)', label_response) else 5
                bias = int(re.search(r'BIAS:\s*(\d+)', label_response).group(1)
                           ) if re.search(r'BIAS:\s*(\d+)', label_response) else 5
                completeness = int(re.search(r'COMPLETENESS:\s*(\d+)', label_response).group(
                    1)) if re.search(r'COMPLETENESS:\s*(\d+)', label_response) else 5
                flaws_match = re.search(r'FLAWS:\s*(.+)', label_response)
                flaws = flaws_match.group(1).strip(
                ) if flaws_match else "none identified"

                # Store in dataset
                data_entry = {
                    "id": i + 1,
                    "topic": topic,
                    "article": generated_article,
                    "target_quality": quality,
                    "target_flaw": target_flaw,
                    "labels": {
                        "source_quality": source_quality,
                        "certainty_appropriateness": certainty,
                        "bias_balance": bias,
                        "completeness": completeness
                    },
                    "identified_flaws": flaws,
                    "generated_at": datetime.datetime.now().isoformat()
                }
                synthetic_dataset.append(data_entry)

                # Update center preview with all articles
                center_preview = "📝 GENERATED ARTICLES\n" + "=" * 44 + "\n\n"
                for entry in synthetic_dataset:
                    center_preview += f"**Example {entry['id']}/{num_examples}** (Quality: {entry['target_quality'].upper()})\n"
                    center_preview += "-" * 44 + "\n\n"
                    center_preview += entry['article'] + "\n\n"
                    center_preview += "=" * 44 + "\n\n"

                # Update metadata display
                metadata_display = "📋 DATASET METADATA\n" + "=" * 44 + "\n\n"
                for entry in synthetic_dataset:
                    metadata_display += f"**Example {entry['id']}** - {entry['target_quality'].upper()}\n"
                    metadata_display += f"Source Quality: {entry['labels']['source_quality']}/10\n"
                    metadata_display += f"Certainty Appropriateness: {entry['labels']['certainty_appropriateness']}/10\n"
                    metadata_display += f"Bias Balance: {entry['labels']['bias_balance']}/10\n"
                    metadata_display += f"Completeness: {entry['labels']['completeness']}/10\n"
                    metadata_display += f"Flaws: {entry['identified_flaws']}\n\n"

                # Update log
                log += f"✅ Example {i+1} complete\n"
                log += f"   Scores: {source_quality}, {certainty}, {bias}, {completeness}\n\n"

                yield center_preview, log, metadata_display, None

        # Export to JSONL
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize topic for filename (remove invalid characters)
        safe_topic = re.sub(r'[<>:"/\\|?*]', '', topic)  # Remove invalid chars
        # Replace spaces with underscores
        safe_topic = safe_topic.replace(' ', '_')
        # Limit length to avoid overly long filenames
        safe_topic = safe_topic[:50]
        filename = f"synthetic_data_{safe_topic}_{timestamp}.jsonl"

        # Convert to absolute path for reliable downloads in deployed environments
        absolute_filepath = os.path.abspath(filename)

        with open(absolute_filepath, 'w', encoding='utf-8') as f:
            for entry in synthetic_dataset:
                f.write(json.dumps(entry) + '\n')

        # Final log update
        log += "\n" + "=" * 44 + "\n"
        log += f"✅ Generation complete!\n\n"
        log += f"📁 Exported to: {filename}\n"
        log += f"📊 Total examples: {num_examples}\n"
        log += f"💾 Format: JSONL (for ML training)\n\n"
        log += "Dataset can be used for:\n"
        log += "- Training epistemic classifiers\n"
        log += "- Evaluating quality detection systems\n"
        log += "- Fine-tuning article generators\n"

        # Final metadata summary
        metadata_display += "\n" + "=" * 44 + "\n"
        metadata_display += f"**📊 DATASET SUMMARY**\n\n"
        metadata_display += f"Total examples: {num_examples}\n"
        metadata_display += f"Topic: {topic}\n"
        metadata_display += f"Quality distribution:\n"
        for tier in quality_tiers:
            count = sum(
                1 for e in synthetic_dataset if e['target_quality'] == tier)
            metadata_display += f"  - {tier}: {count}\n"
        metadata_display += f"\n📁 Saved to: {filename}\n"

        yield center_preview, log, metadata_display, absolute_filepath

    except Exception as e:
        error_log = log + \
            f"\n\n❌ Error during generation: {str(e)}\n\nPlease try again."
        yield center_preview, error_log, metadata_display, None
