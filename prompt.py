import re

def get_answer_points_prompt(question, answer):
    answer_points_promt = f'''# Code Repository Comprehension Evaluation Framework Generator

## Task Description
Act as a technical documentation specialist to transform raw LLM answers into standardized evaluation frameworks containing:
1. Refined canonical answer
2. Quantifiable scoring dimensions
For automated assessment of model performance on codebase understanding tasks.

## Input Requirements
- Question(Codebase comprehension question with specific technical details):
<question>
{question}
</question>
- Raw Answer(Technical response requiring evaluation):
<answer>
{answer}
</answer>

## Processing Specifications

### Canonical Answer Generation
1. **Completeness**: Preserve all technical details and key conclusions
2. **Professionalism**: Restructure as technical documentation (logical flow/accurate terminology/no redundancy)
3. **Format**: Use paragraph text only - prohibit lists/code blocks

### Scoring Dimension Extraction
1. **Independence**: Each dimension must represent atomic technical elements
2. **Readability**: Each dimension description must be a Complete Setence, not Noun Phrase
3. **Priority**: Ordered by descending technical criticality
4. **Quantification**: Total score is strictly 10 points
5. **Verifiability**: Each dimension must map to specific canonical answer content

## Output Format
Strictly adhere to this structure (no free-form text):

### Canonical Answer
[Technically refined answer paragraph]

### Evaluation Dimensions (Total: 10 points)
- 1. [Technical element 1 description] (x points): [Exact canonical answer excerpt]
- 2. [Technical element 2 description] (x points): [Exact canonical answer excerpt]
- 3. [Technical element 3 description] (x points): [Exact canonical answer excerpt](if exist)
- 4. [Technical element 4 description] (x points): [Exact canonical answer excerpt](if exist)
...
'''
    return answer_points_promt


def get_rewrite_answer_prompt(answer, points):
    rewrite_prompt = f'''Task: Rewrite the provided Standard Answer into a numbered list of answer item, where each item directly corresponds to one Scoring Criteria point. Follow these rules:

Order: Address scoring criteria in the exact order listed, starting with the highest-weighted point.
Structure: Begin each numbered item with the exact criterion description from the scoring criteria, followed by a colon (:).
Content: After the colon, write only the part of the standard answer that satisfies this scoring point.
Format: Use markdown with no free-form text.
Example:

1. No, cannot be removed (3 points): The include/cutlass/gemm directory cannot be relocated without disrupting critical system functionality
2. Correct identification of core GEMM component dependencies: The include/cutlass/gemm directory serves as the architectural backbone for CUTLASS's GEMM operations. This path contains essential components such as the core Gemm template class in device/gemm.h, which defines fundamental matrix operation interfaces, and specialized implementations like the SM100-optimized CollectiveMma structure in collective/sm100_blockscaled_mma_warpspecialized.hpp.
Input:

Standard Answer:
<answer>
{answer}
</answer>
Scoring Criteria:
<points>
{points}
</points>
Output Format:

1. [Criterion 1 description] ([points] points): [Explanation]  
2. [Criterion 2 description] ([points] points): [Explanation]  
...  '''
    return rewrite_prompt


def get_evaluation_prompt(question, groundtruth_answer, points: list[str], candidate_answer) -> str:

    def get_point_value(point):
        pattern = r'\(([\d\.]+) point(s)?\)'
        match = re.search(pattern, point)
        if match:
            return match.group(1)
        else:
            print('in get_evaluation_prompt::get point_value unmatched!\n' + point)
            exit(1)

    points_desc = ''
    output_format = ''
    for i, point in enumerate(points):
        points_desc += point.strip() + '\n'
        if i == 0:
            output_format += f'Criterion {i + 1}:\n   - [Score: x/{get_point_value(point)}]\n   - Rationale: [Explanation of how the answer addresses this point]\n\n'
        else:
            output_format += f'Criterion {i + 1}:\n   - [Score: x/{get_point_value(point)}]\n   - Rationale: [Explanation]\n\n'

    evaluation_prompt = f'''Task: Evaluate a candidate answer against the scoring criteria and canonical answer for a question.
Question:<question>{question}</question>

Canonical Answer:
<canonical_answer>
{groundtruth_answer}
</canonical_answer>

Scoring Criteria:
<points>
{points_desc.strip()}
</points>
Total: 10 points

Candidate Answer:
<candidate_answer>
{candidate_answer}
</candidate_answer>

Evaluation Process

Step 1: Compare the candidate answer to the canonical answer reference.
Step 2: For each scoring criterion:
Check if the candidate answer explicitly addresses the criterion.
Award full points if the answer covers the criterion accurately and completely.
Award partial/no points for incomplete, incorrect, or missing coverage.
Step 3: Provide a rationale for each scoring decision.

Donnot output the process above, only output the evaluation result.
## Output Format
Strictly adhere to this structure (no free-form text):

Candidate Answer Score: [Total]/10

{output_format.strip()}
'''
    return evaluation_prompt