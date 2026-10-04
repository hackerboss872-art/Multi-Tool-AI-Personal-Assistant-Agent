"""
Job Search Agent using LangChain and Ollama (Qwen 2.5 3B)

Workflow:
1. Retrieve candidate information from resume.pdf using Resume RAG.
2. Search for job opportunities using the existing job search tool.
3. Use local Qwen 2.5 3B to compare the resume with the job results.
4. Generate a structured career report.

Report sections:
A. Matching job opportunities
B. Skills from the resume that match
C. Skills that may be required or missing
D. A practical preparation plan
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from tools import get_resume_data, get_job_recommendation


# ==========================================
# 1. LOCAL QWEN MODEL
# ==========================================

def get_job_agent_model():
    """
    Create the local Qwen 2.5 3B model.

    No cloud LLM is used.
    Tools are used separately in Step 1 and Step 2.
    """

    return ChatOllama(
        model="qwen2.5:3b",
        temperature=0.2
    )


# ==========================================
# 2. JOB SEARCH AGENT
# ==========================================

def run_agent(job_title: str, salary: str) -> str:
    """
    Run the complete Job Search Agent workflow.

    Steps:
    1. Retrieve resume information.
    2. Search for relevant jobs.
    3. Compare resume and jobs using Qwen.
    """

    # --------------------------------------
    # Convert salary to integer
    # --------------------------------------

    try:
        salary_min = int(
            str(salary)
            .replace(",", "")
            .strip()
        )
    except ValueError:
        salary_min = 400000

    print(
        f"--> Step 1: Retrieving resume data "
        f"for '{job_title}'..."
    )

    # --------------------------------------
    # STEP 1: Resume RAG
    # --------------------------------------

    try:

        resume_data = get_resume_data.invoke(
            {
                "query": (
                    f"{job_title} "
                    "skills experience education projects"
                )
            }
        )

    except Exception as error:

        resume_data = (
            f"Could not retrieve resume data: {error}"
        )

    # Keep the context manageable for Qwen
    resume_data = str(resume_data)[:6000]

    print(
        f"--> Step 2: Searching live job listings "
        f"for '{job_title}' "
        f"(Min Salary: {salary_min} INR)..."
    )

    # --------------------------------------
    # STEP 2: Job Search
    # --------------------------------------

    try:

        job_data = get_job_recommendation.invoke(
            {
                "what": job_title,
                "salary_min": salary_min
            }
        )

    except Exception as error:

        job_data = (
            f"Could not retrieve job listings: {error}"
        )

    # Keep the context manageable for Qwen
    job_data = str(job_data)[:6000]

    # --------------------------------------
    # STEP 3: QWEN COMPARISON
    # --------------------------------------

    print(
        "--> Step 3: Comparing resume skills "
        "with job listings using Qwen 2.5 3B..."
    )

    system_prompt = """
You are an expert AI Job Search and Career Advisor.

Your task is to compare a candidate's resume information
with the provided job search results.

IMPORTANT RULES:

1. Use ONLY the information provided in the
   resume data and job search results.

2. Do NOT invent:
   - companies
   - job listings
   - salaries
   - candidate skills
   - candidate experience
   - education
   - job requirements

3. If something is not available in the provided data,
   clearly say that it is not available.

4. Do not claim that a candidate is definitely qualified
   for a job. Explain the available matching information.

5. Keep the answer clear and practical.

Your final response MUST contain exactly these
four sections:

A. Matching job opportunities

B. Skills from the resume that match

C. Skills that may be required or missing

D. A practical preparation plan
"""

    user_prompt = f"""
TARGET JOB:
{job_title}

MINIMUM SALARY:
{salary_min} INR

----------------------------------------
RESUME INFORMATION
----------------------------------------

{resume_data}

----------------------------------------
JOB SEARCH RESULTS
----------------------------------------

{job_data}

----------------------------------------

Now compare the resume information with the
job search results.

Return the final answer using exactly these
four sections:

A. Matching job opportunities

B. Skills from the resume that match

C. Skills that may be required or missing

D. A practical preparation plan
"""

    # Create local Qwen model
    llm = get_job_agent_model()

    # Create messages
    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_prompt
        )
    ]

    # Generate final report
    try:

        response = llm.invoke(messages)

        return response.content

    except Exception as error:

        return (
            "Could not generate the final job analysis.\n"
            f"Error: {error}"
        )


# ==========================================
# 3. TERMINAL TEST
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("JOB SEARCH AGENT - TERMINAL TEST")
    print("=" * 60)

    test_job_title = "Python Developer"
    test_salary = "400000"

    print(
        f"Target Job Title : {test_job_title}"
    )

    print(
        f"Minimum Salary   : {test_salary} INR"
    )

    print("=" * 60)
    print()

    # Run the Job Search Agent
    report = run_agent(
        job_title=test_job_title,
        salary=test_salary
    )

    # Display final report
    print()
    print("=" * 60)
    print("FINAL JOB SEARCH AGENT REPORT")
    print("=" * 60)
    print()

    print(report)

    print()
    print("=" * 60)
    print("JOB SEARCH AGENT FINISHED")
    print("=" * 60)