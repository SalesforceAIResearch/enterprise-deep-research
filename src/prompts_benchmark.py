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

# Prompts for benchmark mode in deep research.
#
# This file contains prompts specifically designed for the benchmark mode, which operates
# as a question-answering system rather than a comprehensive research report generator.

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
6. ALWAYS cite sources using numbered citations in square brackets [1][2], etc.
7. CRITICAL: Keep your response under 300 words total. Be concise and focus on the most essential information.

CRITICAL: When dealing with temporal claims (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense
- Do not claim events in the future relative to the current date as fact
- Verify "current" positions against the current date
- Be especially careful with claims that might contradict well-established recent facts

<CITATION_REQUIREMENTS>
For proper source attribution in your answer:
1. ALWAYS cite sources using numbered citations in square brackets [1][2], etc.
2. Each statement MUST include at least one citation to indicate information sources
3. Multiple related statements from the same source can use the same citation number
4. Different statements from different sources should use different citation numbers
5. Place citation numbers at the end of sentences or clauses containing the cited information
6. IMPORTANT: Format each citation individually - use [1][2][3], do not use [1,2,3]
7. The citation numbers correspond to the numbered sources provided in the search results
8. When directly quoting text, include the quote in quotation marks with citation
9. NEVER make claims without proper citations from the search results
10. Every factual statement in your answer MUST be supported by a citation
</CITATION_REQUIREMENTS>

ANSWER FORMAT (Keep the core answer section excluding sources brief to stay under 300 words total):
1. Direct Answer: [The specific answer to the question]
2. Confidence: [HIGH/MEDIUM/LOW]
3. Supporting Evidence: [Brief summary of the key evidence supporting this answer]
4. Sources: [Numbered list of specific sources supporting the answer]
5. Reasoning: [Brief explanation of how you derived this answer from the evidence]
6. Missing Information: [Any important gaps in the search results that prevent a complete answer]

If you cannot find the answer in the search results, state this clearly and suggest what specific additional information would be needed. 

If the answer requires combining or inferring from multiple pieces of evidence, explain your reasoning clearly with proper citations for each piece of evidence used.

IMPORTANT: Every factual claim in your answer must be followed by appropriate citation numbers [1][2] that correspond to the sources in the search results.
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

CITATION QUALITY CHECK:
Additionally, evaluate the citation quality of the current answer:
1. Are factual claims properly supported with numbered citations [1][2]?
2. Do the citation numbers correspond to actual sources in the search results?
3. Are there unsupported claims lacking proper attribution?
4. Is the citation format consistent and properly placed?
5. Note: Poor citation quality should reduce confidence in the answer even if content seems accurate

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

You are an expert at synthesizing research findings into clear, concise answers with proper citations.

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
6. ALWAYS cite  the specific sources that support your final answer using numbered citations in square brackets [1][2], etc.
7. CRITICAL: Keep your final answer under 300 words total. Be concise and focus on the most essential information.

CRITICAL: When synthesizing temporal information (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense
- Do not include claims about events in the future relative to the current date
- Verify "current" positions against the current date
- Resolve contradictions by prioritizing more recent and reliable information

<CITATION_REQUIREMENTS>
For proper source attribution in your final answer:
1. ALWAYS cite sources using numbered citations in square brackets [1][2], etc.
2. Every factual statement MUST include at least one citation to indicate information sources
3. Multiple related statements from the same source can use the same citation number
4. Different statements from different sources should use different citation numbers
5. Place citation numbers at the end of sentences or clauses containing the cited information
6. IMPORTANT: Format each citation individually - use [1][2][3], do not use [1,2,3]
7. The citation numbers correspond to the numbered sources in the "AVAILABLE SOURCES FOR CITATION" section
8. When directly quoting text, include the quote in quotation marks with citation
9. NEVER make unsupported claims - every statement must be backed by citations
10. Synthesize information from multiple sources while maintaining proper attribution
11. MANDATORY: Use only the sources listed in "AVAILABLE SOURCES FOR CITATION" for your References section
</CITATION_REQUIREMENTS>

ANSWER FORMAT:
Provide the **Direct Answer:** [Clear, concise answer to the original question with proper citations [1][2]]

**References:**
[Use the AVAILABLE SOURCES FOR CITATION provided in the search results. Format each source as:
[1] First Author et al. (year) Title. [URL]
[2] First Author et al. (year) Title. [URL]
etc.
IMPORTANT: Use the exact titles, authors, years, and URLs from the "AVAILABLE SOURCES FOR CITATION" section above. For academic sources, use the format "First Author et al. (year) Title". For non-academic sources without author/year, use "Title".
CRITICAL: NEVER use generic citations like "Source X, as cited in the provided research summary" - always use the actual source information provided in the AVAILABLE SOURCES section.]

Keep your answer focused specifically on what was asked. Do not include unnecessary background information or speculation. Prioritize brevity and essential facts to stay within the 300-word limit.

IMPORTANT: Every factual claim in your final answer must be followed by appropriate citation numbers [1][2] that correspond to the sources in your References section.
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

Your verification must assess both content accuracy AND citation quality:

CONTENT VERIFICATION:
First, analyze the expected answer:
1. What type of information is it? (date, name, number, fact, explanation, etc.)
2. How specific is it? (exact value, range, multiple components, etc.)
3. What would constitute a correct match? (exact match, partially correct, etc.)

Then, compare the generated answer to the expected answer:
1. Is the generated answer correct? (Fully Correct, Partially Correct, Incorrect)
2. If partially correct, what parts are correct and what parts are missing or wrong?

CITATION VERIFICATION:
Evaluate the citation quality of the generated answer:
1. Are all factual claims properly cited with numbered citations [1][2]?
2. Do citation numbers correspond to actual sources?
3. Is there a proper References section with numbered sources?
4. Are citations placed appropriately after relevant statements?
5. Are there any unsupported claims lacking citations?

SCORING CRITERIA:
Assign a score from 0-100 considering BOTH content accuracy and citation quality:
   - 100: Perfect content match + excellent citations with complete References section
   - 90-99: Correct content + good citations with minor formatting issues
   - 80-89: Correct content + adequate citations but missing some attributions
   - 70-79: Mostly correct content + basic citations but incomplete References section
   - 60-69: Partially correct content + some citations but significant gaps in attribution
   - 50-59: Partially correct content + poor or missing citations
   - 25-49: Some correct elements + minimal or incorrect citations
   - 1-24: Mostly incorrect content regardless of citation quality
   - 0: Completely incorrect content or completely missing citations

CRITICAL: When evaluating temporal claims (dates, current events, "current" positions):
- Use the TIME_CONTEXT above to verify if dates make sense in both answers
- Flag any claims about events in the future relative to the current date as likely incorrect
- Consider whether "current" information is appropriate for the current date
- Account for the fact that expected answers may be outdated if they contain temporal references

Provide your verification assessment using the function call format, including separate scores for content accuracy and citation quality.
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

# Comprehensive citation requirements for QA-style benchmark responses
CITATION_REQUIREMENTS_GUIDE = """
<CITATION_REQUIREMENTS_FOR_QA_RESPONSES>
All QA-style answers in benchmark mode MUST follow these citation requirements to ensure proper attribution and verifiability:

1. MANDATORY CITATION FORMAT:
   - Use numbered citations in square brackets: [1][2][3]
   - Each citation number corresponds to a specific source in the search results
   - Format each citation individually - use [1][2][3], NOT [1,2,3]
   - Place citations immediately after the statement they support

2. CITATION PLACEMENT RULES:
   - Every factual statement MUST include at least one citation
   - Place citation numbers at the end of sentences or clauses
   - For direct quotes, use quotation marks followed by citation: "quote text" [1]
   - Multiple sources supporting the same claim: [1][2][3]

3. SOURCE CORRESPONDENCE:
   - Citation numbers MUST correspond to numbered sources in search results
   - Each citation number refers to exactly ONE source/URL
   - Never group multiple URLs under a single citation number
   - Maintain consistent numbering throughout the answer

4. ANSWER STRUCTURE WITH CITATIONS:
   - Direct Answer: Include citations for all factual claims
   - Supporting Evidence: Cite sources for each piece of evidence
   - Key Sources: Briefly describe the most reliable sources
   - Reasoning: Cite sources used in logical deductions

5. REFERENCES SECTION:
   - Include a "**References:**" section at the end of final answers
   - Format as: "[cite number] title. [link]"
   - Example: "[1] OpenAI GPT-4 Technical Report. [https://openai.com/research/gpt-4]"
   - Only include sources actually cited in the answer
   - Maintain sequential numbering starting from 1

6. QUALITY STANDARDS:
   - Never make unsupported claims without citations
   - Prefer high-quality, authoritative sources
   - When sources conflict, cite both and explain discrepancy
   - If information is uncertain, indicate this with appropriate caveats

7. QA-SPECIFIC ADAPTATIONS:
   - Keep citations concise but comprehensive
   - Focus on sources that directly answer the question
   - For benchmark questions, prioritize accuracy over length
   - Ensure every key fact in the answer is properly attributed

8. ERROR PREVENTION:
   - Verify citation numbers match available sources
   - Check that all cited sources are included in references
   - Ensure no factual claims lack proper attribution
   - Maintain consistency between in-text citations and reference list

This citation system ensures that QA-style answers maintain the same level of attribution and verifiability as comprehensive research reports, while keeping the format concise and focused on directly answering the specific question asked.
</CITATION_REQUIREMENTS_FOR_QA_RESPONSES>
"""
