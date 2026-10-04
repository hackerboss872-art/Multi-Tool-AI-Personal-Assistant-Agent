from langchain_core.tools import tool
from ddgs import DDGS
from rag import process_pdf

# -------------------------------------------------
# Reuse Chroma vectorstore from rag.py
# -------------------------------------------------
vectorstore = process_pdf()


# -------------------------------------------------
# Resume RAG Tool
# -------------------------------------------------

@tool
def get_resume_data(query: str) -> str:
    """
    Search and retrieve relevant information from the candidate's resume.

    Use this tool when the user asks questions about candidate resume details,
    work experience, skills, education, projects, contact info, or background.
    """
    if vectorstore is None:
        return "Resume vector database is not available."

    try:
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in the resume."

        clean_contents = [doc.page_content.strip() for doc in docs if doc.page_content]
        return "\n\n".join(clean_contents)
    except Exception as e:
        return f"Error retrieving resume data: {e}"


# -------------------------------------------------
# Web Search Tool
# -------------------------------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the live web using DuckDuckGo.

    Use this tool when the user asks for current,
    recent, or internet-based information.
    """

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No search results found."

        output = []

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "")

            output.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   URL: {url}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search failed: {e}"


# -------------------------------------------------
# Job Recommendation Tool
# -------------------------------------------------

@tool
def get_job_recommendation(what: str, salary_min: int) -> str:
    """
    Search live job listings using a job-related keyword
    and minimum annual salary in Indian Rupees.

    Example:
    what = "python"
    salary_min = 400000
    """

    query = (
        f"{what} jobs India "
        f"salary {salary_min} INR"
    )

    try:
        results = DDGS().text(
            query,
            max_results=8
        )

        if not results:
            return "No job listings found."

        output = []

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "")

            output.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   URL: {url}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Job search failed: {e}"


# -------------------------------------------------
# List of available tools
# -------------------------------------------------

tools = [
    web_search,
    get_job_recommendation,
    get_resume_data
]


if __name__ == "__main__":
    print("Tools loaded successfully!")
    print()
    print("Available tools:")
    print("- web_search")
    print("- get_job_recommendation")
    print("- get_resume_data")