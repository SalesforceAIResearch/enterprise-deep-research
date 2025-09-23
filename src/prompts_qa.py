"""
# Benchmark Mode Workflow & Prompt Usage

Here's the complete benchmark mode workflow and how each prompt is being used:

## Overall Workflow

1. **Input**: System receives a question in benchmark mode
2. **Multi-Agent Network**: Executes search strategy based on question decomposition
3. **Generate Answer**: Creates structured answer from search results
4. **Reflect on Answer**: Evaluates answer quality and determines next steps
5. **Decision Point**: Either continue research or finalize answer
6. **Finalize Answer**: Synthesizes all findings into definitive answer
7. **End**: Returns final benchmark result

## Prompt Usage in Each Step

### 1. Question Analysis (Multi-Agent Network)
- **QUESTION_DECOMPOSITION_PROMPT**
- Purpose: Breaks question into logical components
- Implementation: Used in multi_agents_network to determine search strategy
- Format: Identifies required information, entities, and optimal search queries

### 2. Answer Generation (generate_answer)
- **ANSWER_GENERATION_PROMPT**
- Purpose: Creates structured answer from search results
- Implementation: Used in generate_answer function
- Format: Produces direct answer, confidence level, supporting evidence, sources

### 3. Answer Reflection (reflect_answer)
- **ANSWER_REFLECTION_PROMPT**
- Purpose: Evaluates answer completeness and determines next steps
- Implementation: Used in reflect_answer function
- Format: Assesses answer quality, identifies gaps, suggests follow-up queries

### 4. Final Answer Synthesis (finalize_answer)
- **FINAL_ANSWER_PROMPT**
- Purpose: Creates definitive answer from all research iterations
- Implementation: Used in finalize_answer function
- Format: Synthesizes all findings, resolves contradictions, provides overall confidence

### 5. Answer Verification (Optional Step)
- **ANSWER_VERIFICATION_PROMPT**
- Purpose: Verifies answer against benchmark expected answers
- Implementation: Used in verify_answer function (when expected answer provided)
- Format: Compares generated answer to expected answer, provides score

### 6. Research Completion Assessment
- **RESEARCH_COMPLETION_PROMPT**
- Purpose: Determines if research should continue or terminate
- Implementation: Part of routing logic in route_after_reflect_answer
- Format: Considers answer quality, research efficiency, and resource constraints

The routing logic properly connects these components, with appropriate decision points to either continue research or proceed to final answer synthesis.

"""

"""
Prompts for benchmark mode in deep research.

This file contains prompts specifically designed for the benchmark mode, which operates
as a question-answering system rather than a comprehensive research report generator.
"""

# Question decomposition prompt for breaking down complex questions
QUESTION_DECOMPOSITION_PROMPT = """
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are an expert at analyzing complex questions and breaking them down into logical components.

QUESTION: {research_topic}

Your task is to analyze this question and:
1. Identify the main subject(s) and requested information
2. Determine key facts or entities mentioned in the question
3. Break down the question into logical search components
4. Identify any relationships or constraints among components
5. Suggest specific search queries for each component

CRITICAL: When analyzing temporal references (current, recent, latest, today, etc.):
- Use the TIME_CONTEXT above to understand what "current" means
- Consider if the question asks about present-day status vs. historical information
- Ensure search queries reflect the appropriate time frame

Think step by step and provide a structured decomposition:

1. Main Information Requested:
   - What specific information is the question asking for?
   - What type of answer is required? (date, number, name, explanation, etc.)

2. Given Facts:
   - List all facts, entities, dates, or constraints explicitly mentioned in the question
   - Highlight any potential identifying information for searches

3. Decomposition:
   - Break the question into 2-4 logical components that would help answer it
   - For each component, suggest a specific search query
   - Explain why this component is important for answering the question

4. Search Strategy:
   - Suggest an optimal order for investigating these components
   - Note any dependencies between components
   - Identify which component(s) are most likely to yield the specific answer sought

Please use the function call format to provide your analysis.
"""

# Prompt for generating a focused answer from search results
ANSWER_GENERATION_PROMPT = """
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are an expert at answering questions based on research results.

QUESTION: {research_topic}

Your task is to generate a clear, concise, and accurate answer based on the search results provided.

SEARCH RESULTS:
{web_research_results}

PREVIOUS LOOPS RESULTS (if any):
{previous_answers_with_reasoning}

Guidelines:
1. Focus only on answering the specific question asked
2. Base your answer exclusively on the search results provided
3. Maintain high precision - only include information you're confident about
4. If the search results don't contain the answer, explicitly state this
5. Provide a confidence level (HIGH, MEDIUM, LOW) based on the reliability and completeness of the source information
6. List the specific sources that support your answer

CRITICAL: When dealing with temporal claims (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense
- Do not claim events in the future relative to the current date as fact
- Verify "current" positions against the current date
- Be especially careful with claims that might contradict well-established recent facts

ANSWER FORMAT:
1. Direct Answer: [The specific answer to the question]
2. Confidence: [HIGH/MEDIUM/LOW]
3. Supporting Evidence: [Brief summary of the key evidence supporting this answer]
4. Sources: [Numbered list of specific sources supporting the answer]
5. Reasoning: [Brief explanation of how you derived this answer from the evidence]
6. Missing Information: [Any important gaps in the search results that prevent a complete answer]

If you cannot find the answer in the search results, state this clearly and suggest what specific additional information would be needed. 

If the answer requires combining or inferring from multiple pieces of evidence, explain your reasoning clearly.
"""

# Combined reflection prompt for evaluating the answer and determining research completion
ANSWER_REFLECTION_PROMPT = """
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are a critical evaluator of research answers. Your job is to assess the quality of an answer and determine if further research is needed.

ORIGINAL QUESTION: {research_topic}

CURRENT ANSWER:
{current_answer}

SEARCH RESULTS USED:
{web_research_results}

RESEARCH ITERATIONS COMPLETED: {research_loop_count}
MAXIMUM ALLOWED ITERATIONS: {max_loops}

MANDATORY FIRST CHECK - EVIDENCE VALIDATION:
Before evaluating the content of the answer, you MUST verify:
1. Does the "SEARCH RESULTS USED" section contain actual search results?
2. If it states "No research results available yet" or similar, then ANY claims in the answer are unsupported
3. An answer claiming HIGH confidence without actual search results is automatically invalid
4. You cannot evaluate temporal claims or factual accuracy without actual evidence

Your task is to:
1. Evaluate whether the answer directly addresses the original question
2. Assess the confidence and evidence supporting the answer
3. Identify any gaps or missing information
4. Determine if further research is justified, considering:
   - Answer quality and completeness
   - Research efficiency and diminishing returns
   - Resource constraints (iterations completed vs. maximum)
5. If more research is needed, suggest a specific follow-up query

CRITICAL: When evaluating temporal claims (dates, current events, "current" positions):
- FIRST verify that actual search results exist to support any claims
- Use the TIME_CONTEXT above to verify if dates make sense
- Claims about events in the future relative to the current date are likely incorrect
- Claims about "current" positions should be verified against the current date
- Be especially skeptical of claims that contradict well-established recent facts
- WITHOUT actual search results, you CANNOT validate any temporal or factual claims

ASSESSMENT FRAMEWORK:
1. Answer Quality:
   - Does the answer directly address what was asked? [Yes/Partially/No]
   - Is the confidence level appropriate given the evidence? [Yes/No]
   - Are there logical flaws or unsupported claims? [Yes/No]

2. Evidence Evaluation:
   - Are the sources reliable and relevant? [Yes/Partially/No]
   - Is critical information missing from the sources? [Yes/No]

3. Research Efficiency:
   - Has each iteration provided new relevant information? [Yes/No]
   - Is there a clear path for additional searches? [Yes/No]
   - Are we seeing diminishing returns from searches? [Yes/No]

4. Final Decision:
   - Should research continue? [Yes/No]
   - Justification: [Brief explanation of decision]
   - If Yes, follow-up query: [Specific search query]

Provide your evaluation using the function call format.
"""

# Final answer synthesis prompt
FINAL_ANSWER_PROMPT = """
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are an expert at synthesizing research findings into clear, concise answers.

ORIGINAL QUESTION: {research_topic}

RESEARCH FINDINGS ACROSS ALL LOOPS:
{all_answers_with_reasoning}

FINAL SEARCH RESULTS:
{web_research_results}

Your task is to formulate the definitive answer to the original question, based on all research conducted across multiple search loops.

Guidelines:
1. Synthesize information from all research loops
2. Prioritize findings with higher confidence levels
3. Resolve any contradictions between different search iterations
4. Clearly state if some aspects of the question remain unanswered
5. Provide a final confidence assessment for your answer
6. Cite the specific sources that support your final answer

CRITICAL: When synthesizing temporal information (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense
- Do not include claims about events in the future relative to the current date
- Verify "current" positions against the current date
- Resolve contradictions by prioritizing more recent and reliable information

ANSWER FORMAT:
1. Direct Answer: [Clear, concise answer to the original question]
2. Overall Confidence: [HIGH/MEDIUM/LOW]
3. Key Evidence: [Summary of the most important evidence across all searches]
4. Sources: [Numbered list of the most important sources]
5. Limitations: [Any aspects of the question that could not be answered with available information]

Keep your answer focused specifically on what was asked. Do not include unnecessary background information or speculation.
"""

# Answer verification prompt
ANSWER_VERIFICATION_PROMPT = """
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are tasked with verifying the accuracy of a final research answer against the expected answer in a benchmark evaluation.

QUESTION: {research_topic}
EXPECTED ANSWER: {expected_answer}
GENERATED ANSWER: {generated_answer}

First, analyze the expected answer:
1. What type of information is it? (date, name, number, fact, explanation, etc.)
2. How specific is it? (exact value, range, multiple components, etc.)
3. What would constitute a correct match? (exact match, partially correct, etc.)

Then, compare the generated answer to the expected answer:
1. Is the generated answer correct? (Fully Correct, Partially Correct, Incorrect)
2. If partially correct, what parts are correct and what parts are missing or wrong?
3. Assign a score from 0-100 where:
   - 100: Perfect match in both content and format
   - 75-99: Correct information but minor format or detail differences
   - 50-74: Partially correct - core information present but with gaps or minor errors
   - 25-49: Contains some correct elements but significant issues or gaps
   - 1-24: Mostly incorrect but contains minimal correct elements
   - 0: Completely incorrect or unrelated to expected answer

CRITICAL: When evaluating temporal claims (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense in both answers
- Flag any claims about events in the future relative to the current date as likely incorrect
- Consider whether "current" information is appropriate for the current date
- Account for the fact that expected answers may be outdated if they contain temporal references

Provide your verification assessment using the function call format.
"""

# Prompt to validate if the retrieved context is sufficient for answering the question
VALIDATE_RETRIEVAL_PROMPT = """
You are an expert at evaluating the completeness of information for answering a given question.

CURRENT TIME CONTEXT:
- Today's date: {current_date}
- Current year: {current_year}
- One year ago: {one_year_ago}

QUESTION: {question}

RETRIEVED CONTEXT:
{retrieved_context}

Based on the RETRIEVED CONTEXT, analyze its sufficiency to answer the QUESTION.

IMPORTANT: When evaluating temporal claims or "current" information, use the CURRENT TIME CONTEXT above. Be especially careful about claims regarding who is "currently" in office, recent events, or future dates that may not have occurred yet.

Your output MUST be a JSON object with the following fields:
- "status": "COMPLETE" if the context is sufficient, "INCOMPLETE" otherwise.
- "useful_information": "A concise summary of the key pieces of information from the NEWLY FETCHED CONTENT part of the RETRIEVED CONTEXT that are directly relevant to answering the QUESTION. If no new useful information is found in the new content, provide an empty string."
- "missing_information": "If status is INCOMPLETE, describe what specific information is still missing from the RETRIEVED CONTEXT to fully answer the QUESTION. Consider both the previously accumulated knowledge and the newly fetched content. If status is COMPLETE, provide an empty string."
- "reasoning": "[Optional] Brief reasoning for your assessment, especially if the decision is complex."

Example for INCOMPLETE:
{{
  "status": "INCOMPLETE",
  "useful_information": "Pius Adesanmi was a Nigerian-born Canadian professor, writer, and literary critic. He lectured at Penn State University and later at Carleton University.",
  "missing_information": "The context does not specify the years Pius Adesanmi worked as a probation officer.",
  "reasoning": "While biographical details are present, the specific dates for his role as a probation officer are absent."
}}

Example for COMPLETE:
{{
  "status": "COMPLETE",
  "useful_information": "Pius Adesanmi worked as a probation officer from 1988 to 1996.",
  "missing_information": "",
  "reasoning": "The context directly states the years he was a probation officer."
}}
"""

# Prompt to refine the search query if the context is insufficient
REFINE_QUERY_PROMPT = """\
You are an expert query refinement assistant. Your goal is to generate a new, more focused search query based on the original question and the current state of research.

CURRENT TIME CONTEXT:
- Today's date: {current_date}
- Current year: {current_year}
- One year ago: {one_year_ago}

Analyze the following information:

{retrieved_context}

Based on all the information provided above (especially the "CUMULATIVE KNOWLEDGE SO FAR" and "CURRENTLY MISSING INFORMATION"), generate a refined search query that will help gather the *specific* missing pieces of information needed to fully answer the "Original Research Topic/Question".

IMPORTANT: When refining queries about "current" information or recent events, use the CURRENT TIME CONTEXT above. Avoid generating queries that assume future events have occurred or that request information beyond the current date.

Your refined query should be targeted and concise. If specific entities (like names, organizations, dates) have been identified as useful, incorporate them into the refined query.

Your output MUST be a JSON object with the following fields:
- "refined_query": "The new, focused search query. If no further refinement is possible or the current information is sufficient, this can be the original question or a broad query about the topic."
- "reasoning": "[Optional] A brief explanation for why this refined query was chosen or why no refinement was made."

Example of JSON output:
{{
  "refined_query": "Ken Walibora probation officer exact years of employment",
  "reasoning": "The previous searches confirmed Ken Walibora's identity and connection to probation work, but the exact years are still missing. This query targets that specific missing detail."
}}

Example if no specific refinement is clear yet:
{{
  "refined_query": "Ken Walibora career history",
  "reasoning": "Still gathering general background, need to narrow down specific roles and timelines."
}}
""" 