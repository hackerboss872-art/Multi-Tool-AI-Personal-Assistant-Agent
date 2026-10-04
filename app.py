"""
AI Job Search Agent - Gradio Web Interface

Powered by:
- Local LLM: Qwen 2.5 3B via Ollama
- Resume RAG: Chroma vectorstore (rag.py / tools.py)
- Search Tools: DuckDuckGo live job search (tools.py)
"""

import os
import sys
from urllib.parse import urlparse
import gradio as gr

# Import existing project modules safely
try:
    from tools import vectorstore, get_resume_data, get_job_recommendation
except Exception as import_err:
    print(f"Warning importing tools: {import_err}")
    vectorstore = None
    get_resume_data = None
    get_job_recommendation = None


# ==========================================
# 1. RESUME RAG RETRIEVAL
# ==========================================

def retrieve_resume_summary(job_title: str) -> dict:
    """
    Retrieves candidate information from resume.pdf using the existing RAG vectorstore.
    """
    if vectorstore is None or get_resume_data is None:
        return {
            "skills": "Not found in resume.",
            "experience": "Not found in resume.",
            "projects": "Not found in resume.",
            "education": "Not found in resume.",
            "full_context": "Not found in resume."
        }

    try:
        # Perform targeted queries to extract resume details
        skills_raw = get_resume_data.invoke("technical skills programming languages frameworks tools database")
        exp_raw = get_resume_data.invoke("work experience job history roles responsibilities internships")
        projects_raw = get_resume_data.invoke("projects software built applications portfolio development")
        edu_raw = get_resume_data.invoke("education degree university college qualifications degree GPA")
        full_context = get_resume_data.invoke(job_title)

        def clean_val(val):
            if not val or "not found" in str(val).lower() or "error" in str(val).lower():
                return "Not found in resume."
            return str(val).strip()

        return {
            "skills": clean_val(skills_raw),
            "experience": clean_val(exp_raw),
            "projects": clean_val(projects_raw),
            "education": clean_val(edu_raw),
            "full_context": clean_val(full_context)
        }
    except Exception as err:
        return {
            "skills": f"Error loading resume data: {err}",
            "experience": "Not found in resume.",
            "projects": "Not found in resume.",
            "education": "Not found in resume.",
            "full_context": "Not found in resume."
        }


# ==========================================
# 2. JOB SEARCH & PARSING
# ==========================================

def search_and_parse_jobs(job_title: str, salary_min: int, location: str, exp_level: str) -> list:
    """
    Searches for live job opportunities using DuckDuckGo and parses structured cards.
    """
    query = f"{job_title} jobs {location} salary {salary_min} INR"
    if exp_level and exp_level != "Any":
        query += f" {exp_level}"

    results = []
    try:
        from ddgs import DDGS
        ddg_results = list(DDGS().text(query, max_results=6))
        results = ddg_results if ddg_results else []
    except Exception as e:
        print(f"DDGS direct search exception: {e}")
        # Fallback to get_job_recommendation tool if available
        if get_job_recommendation:
            try:
                raw_rec = get_job_recommendation.invoke({"what": job_title, "salary_min": salary_min})
                results = [{"title": f"{job_title} Listing", "body": raw_rec, "href": "#"}]
            except Exception:
                results = []

    parsed_jobs = []
    for item in results:
        raw_title = item.get("title", "Not specified").strip()
        body = item.get("body", "Not specified").strip()
        url = item.get("href", "#").strip()

        # Heuristic extraction of source domain
        source = "Web Search"
        if url and url.startswith("http"):
            try:
                domain = urlparse(url).netloc
                source = domain.replace("www.", "")
            except Exception:
                source = "Web Search"

        # Heuristic extraction of company & location from title/body if present
        company = "Not specified"
        if " at " in raw_title:
            parts = raw_title.split(" at ")
            if len(parts) > 1:
                company = parts[1].split("-")[0].split("|")[0].strip()
        elif " - " in raw_title:
            parts = raw_title.split(" - ")
            if len(parts) > 1:
                company = parts[1].strip()

        loc_str = location if location else "Not specified"

        parsed_jobs.append({
            "title": raw_title,
            "company": company if company else "Not specified",
            "location": loc_str,
            "salary": f"As per listing (Min threshold: {salary_min:,} INR)",
            "description": body,
            "url": url,
            "source": source
        })

    return parsed_jobs


# ==========================================
# 3. SKILL GAP ANALYSIS
# ==========================================

def analyze_skill_gaps(resume_dict: dict, parsed_jobs: list) -> dict:
    """
    Compares skills in resume vs skills mentioned in job descriptions.
    """
    tech_keywords = [
        "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular", "vue",
        "html", "css", "sql", "postgresql", "mysql", "mongodb", "redis", "fastapi", "flask",
        "django", "express", "node", "docker", "kubernetes", "aws", "azure", "gcp",
        "git", "github", "ci/cd", "rest", "api", "graphql", "machine learning", "deep learning",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "langchain", "rag",
        "agile", "scrum", "linux", "bash", "data science"
    ]

    resume_text = (
        resume_dict.get("skills", "") + " " +
        resume_dict.get("experience", "") + " " +
        resume_dict.get("projects", "") + " " +
        resume_dict.get("full_context", "")
    ).lower()

    jobs_text = " ".join([j["title"] + " " + j["description"] for j in parsed_jobs]).lower()

    found_resume_skills = [kw.title() for kw in tech_keywords if kw in resume_text]
    found_job_skills = [kw.title() for kw in tech_keywords if kw in jobs_text]

    # Deduplicate while preserving order
    found_resume_skills = list(dict.fromkeys(found_resume_skills))
    found_job_skills = list(dict.fromkeys(found_job_skills))

    matching = [s for s in found_resume_skills if s in found_job_skills]
    missing = [s for s in found_job_skills if s not in found_resume_skills]

    return {
        "resume_skills": found_resume_skills if found_resume_skills else ["No technical skills parsed"],
        "job_skills": found_job_skills if found_job_skills else ["Insufficient info in job descriptions"],
        "matching_skills": matching if matching else ["Insufficient information for detailed matching."],
        "missing_skills": missing if missing else ["No obvious missing skills identified from job descriptions."]
    }


# ==========================================
# 4. QWEN REPORT GENERATION (DIRECT INVOCATION)
# ==========================================

def generate_qwen_analysis(job_title: str, salary_min: int, resume_dict: dict, parsed_jobs: list, skill_data: dict):
    """
    Invokes local Qwen 2.5 3B directly via ChatOllama.invoke() without tool binding
    to generate the final preparation plan and AI report.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.2)
    except Exception as e:
        err_msg = f"Job search is temporarily unavailable. Local Ollama model error: {e}"
        return err_msg, err_msg

    jobs_formatted = ""
    if parsed_jobs:
        for i, j in enumerate(parsed_jobs[:5], 1):
            jobs_formatted += (
                f"{i}. {j['title']}\n"
                f"   Company: {j['company']}\n"
                f"   Location: {j['location']}\n"
                f"   Salary: {j['salary']}\n"
                f"   Source: {j['source']}\n"
                f"   URL: {j['url']}\n"
                f"   Description: {j['description'][:250]}\n\n"
            )
    else:
        jobs_formatted = "No live job search results available."

    prompt = f"""You are an expert AI Career Advisor powered by Qwen 2.5 3B.

A candidate is seeking a role:
- Target Job Title: {job_title}
- Minimum Salary: {salary_min:,} INR

CANDIDATE RESUME SUMMARY:
- Skills: {', '.join(skill_data['resume_skills'])}
- Experience: {resume_dict['experience'][:300]}
- Projects: {resume_dict['projects'][:300]}
- Education: {resume_dict['education'][:200]}

LIVE JOB SEARCH RESULTS:
{jobs_formatted}

SKILL COMPARISON:
- Matching Skills: {', '.join(skill_data['matching_skills'])}
- Required / Missing Skills: {', '.join(skill_data['missing_skills'])}

GROUNDING RULES:
- Do NOT fabricate companies, salaries, or candidate experience.
- Use "Not specified" for missing fields.
- Make the preparation plan realistic and actionable based strictly on the missing skills identified.

Generate your response in TWO SECTIONS separated by '---REPORT_SPLIT---':

SECTION 1: PREPARATION ROADMAP
(A practical 4-week preparation plan covering:
- Week 1: Core foundational skills
- Week 2: Key frameworks & tools
- Week 3: Project build & practical application
- Week 4: Portfolio & interview prep)

---REPORT_SPLIT---

SECTION 2: AI CAREER ANALYSIS REPORT
(Must contain exact headings below):

A. Matching Job Opportunities

B. Skills From My Resume That Match

C. Skills That May Be Required or Missing

D. Practical Preparation Plan
"""

    try:
        response = llm.invoke(prompt)
        content = response.content

        if "---REPORT_SPLIT---" in content:
            parts = content.split("---REPORT_SPLIT---")
            plan_text = parts[0].replace("SECTION 1: PREPARATION ROADMAP", "").strip()
            report_text = parts[1].replace("SECTION 2: AI CAREER ANALYSIS REPORT", "").strip()
            return plan_text, report_text
        else:
            return content, content

    except Exception as e:
        err = f"Job search analysis temporarily unavailable: {e}"
        return err, err


# ==========================================
# 5. MAIN GRADIO PIPELINE (GENERATOR FOR PROGRESS)
# ==========================================

def run_job_search_pipeline(job_title: str, salary_str: str, location: str, exp_level: str):
    """
    Executes deterministic Job Search pipeline with live step updates.
    """
    # Input Validation & Salary Parsing
    if not job_title or not job_title.strip():
        yield (
            "### ❌ Error: Please enter a valid Job Title.",
            "Not found in resume.",
            "No job listings searched.",
            "Insufficient information.",
            "Please fix inputs to generate plan.",
            "Please fix inputs to generate report."
        )
        return

    try:
        salary_min = int(str(salary_str).replace(",", "").replace("INR", "").strip())
        if salary_min < 0:
            salary_min = 400000
    except Exception:
        salary_min = 400000

    job_title_clean = job_title.strip()
    location_clean = location.strip() if location else "India"

    # --------------------------------------------------
    # Step 1/3: Reading resume...
    # --------------------------------------------------
    yield (
        "### ⏳ **Step 1/3**: Reading resume and extracting candidate skills...",
        "Processing resume...",
        "Waiting for step 2...",
        "Waiting for step 3...",
        "Generating roadmap...",
        "Generating final report..."
    )

    resume_summary = retrieve_resume_summary(job_title_clean)

    resume_md = f"""### 📄 Resume Match Analysis
- **Target Role Requested:** {job_title_clean}
- **Minimum Salary:** {salary_min:,} INR
- **Resume Skills Found:** {resume_summary['skills']}
- **Relevant Experience:** {resume_summary['experience']}
- **Relevant Projects:** {resume_summary['projects']}
- **Education:** {resume_summary['education']}
"""

    # --------------------------------------------------
    # Step 2/3: Searching jobs...
    # --------------------------------------------------
    yield (
        "### ⏳ **Step 2/3**: Searching live job listings...",
        resume_md,
        "Searching live web listings...",
        "Analyzing skills...",
        "Generating roadmap...",
        "Generating final report..."
    )

    parsed_jobs = search_and_parse_jobs(job_title_clean, salary_min, location_clean, exp_level)

    if parsed_jobs:
        jobs_md_list = ["### 💼 Job Opportunities"]
        for idx, j in enumerate(parsed_jobs, start=1):
            url_button = f"[🔗 View Job]({j['url']})" if j['url'] != "#" else "No direct link"
            jobs_md_list.append(
                f"#### {idx}. {j['title']}\n"
                f"- **Company:** {j['company']}\n"
                f"- **Location:** {j['location']}\n"
                f"- **Salary:** {j['salary']}\n"
                f"- **Source:** {j['source']}\n"
                f"- **Description:** {j['description']}\n"
                f"- **Action:** {url_button}\n"
            )
        jobs_md = "\n---\n".join(jobs_md_list)
    else:
        jobs_md = "### 💼 Job Opportunities\nNo live job listings found matching criteria. Job search is temporarily unavailable or returned empty results."

    # --------------------------------------------------
    # Step 3/3: Analyzing job matches & generating plan...
    # --------------------------------------------------
    yield (
        "### ⏳ **Step 3/3**: Comparing skills and generating AI preparation plan...",
        resume_md,
        jobs_md,
        "Evaluating skill match...",
        "Invoking local Qwen 2.5 3B...",
        "Synthesizing final report..."
    )

    skill_data = analyze_skill_gaps(resume_summary, parsed_jobs)

    matching_str = ", ".join(skill_data['matching_skills'])
    missing_str = ", ".join(skill_data['missing_skills'])

    skills_md = f"""### 📊 Resume Match & Skill Gap Analysis
- **Skills I Already Have:** {', '.join(skill_data['resume_skills'])}
- **Skills Mentioned in Job Results:** {', '.join(skill_data['job_skills'])}
- **Matching Skills:** {matching_str}
- **Skills That May Need Improvement:** {missing_str}
"""

    # Call local Qwen 2.5 3B for Final Report & Roadmap
    prep_plan_md, final_report_md = generate_qwen_analysis(
        job_title_clean, salary_min, resume_summary, parsed_jobs, skill_data
    )

    # --------------------------------------------------
    # Complete
    # --------------------------------------------------
    yield (
        "### ✅ **Analysis Complete**",
        resume_md,
        jobs_md,
        skills_md,
        f"### 🗺️ Preparation Roadmap\n\n{prep_plan_md}",
        f"### 🤖 AI Career Analysis\n\n{final_report_md}"
    )


def reset_inputs():
    """Resets UI fields."""
    return (
        "Python Developer",
        "400000",
        "India",
        "Any",
        "### Ready for job search.",
        "",
        "",
        "",
        "",
        ""
    )


# ==========================================
# 6. GRADIO UI LAYOUT BUILDER
# ==========================================

def build_ui():
    """
    Creates the Gradio interface with modern dark/light styling.
    """
    css = """
    .main-container { max-width: 1200px; margin: 0 auto; }
    .header-box { text-align: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #1e293b, #0f172a); color: white; border-radius: 12px; }
    .header-box h1 { margin: 0; font-size: 2.2rem; color: #38bdf8; }
    .header-box p { margin-top: 8px; color: #94a3b8; font-size: 1.05rem; }
    .card-panel { border: 1px solid #334155; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
    """

    with gr.Blocks(title="AI Job Search Agent", css=css, theme=gr.themes.Soft()) as demo:

        gr.HTML("""
        <div class="header-box">
            <h1>AI Job Search Agent</h1>
            <p>Find relevant jobs, compare them with your resume, identify skill gaps, and prepare for your target role.</p>
            <small style="color: #64748b;">Powered locally by Qwen 2.5 3B + Ollama + LangChain RAG</small>
        </div>
        """)

        with gr.Row():
            # Left Column: Inputs & Form
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Find Your Next Job")

                job_title_input = gr.Textbox(
                    label="Job Title",
                    value="Python Developer",
                    placeholder="e.g. Python Developer, Data Scientist",
                    interactive=True
                )

                salary_input = gr.Textbox(
                    label="Minimum Salary (INR)",
                    value="400000",
                    placeholder="e.g. 400000",
                    interactive=True
                )

                location_input = gr.Textbox(
                    label="Location",
                    value="India",
                    placeholder="e.g. India, Remote, Bangalore",
                    interactive=True
                )

                exp_level_input = gr.Dropdown(
                    label="Experience Level",
                    choices=["Any", "Fresher", "Internship", "Entry Level", "1-3 Years", "3+ Years"],
                    value="Any",
                    interactive=True
                )

                with gr.Row():
                    search_btn = gr.Button("Search Jobs", variant="primary", size="lg")
                    reset_btn = gr.Button("Reset", variant="secondary", size="lg")

                status_output = gr.Markdown("### Ready for job search.", elem_classes=["card-panel"])

            # Right Column: Outputs
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📄 Resume Match Analysis"):
                        resume_output = gr.Markdown("Enter details and click **Search Jobs** to analyze resume match.")

                    with gr.Tab("💼 Job Opportunities"):
                        jobs_output = gr.Markdown("Live job opportunities will appear here.")

                    with gr.Tab("📊 Skill Gap Analysis"):
                        skills_output = gr.Markdown("Skill matching and gap analysis will appear here.")

                    with gr.Tab("🗺️ Preparation Roadmap"):
                        prep_output = gr.Markdown("Practical 4-week preparation plan will appear here.")

                    with gr.Tab("🤖 AI Career Report"):
                        report_output = gr.Markdown("Complete AI Career Analysis report will appear here.")

        # Bind events
        search_btn.click(
            fn=run_job_search_pipeline,
            inputs=[job_title_input, salary_input, location_input, exp_level_input],
            outputs=[status_output, resume_output, jobs_output, skills_output, prep_output, report_output]
        )

        reset_btn.click(
            fn=reset_inputs,
            inputs=[],
            outputs=[job_title_input, salary_input, location_input, exp_level_input, status_output, resume_output, jobs_output, skills_output, prep_output, report_output]
        )

    return demo


if __name__ == "__main__":
    print("Starting AI Job Search Agent Gradio UI...")
    demo_app = build_ui()
    # Launch app on local server
    demo_app.launch(server_name="127.0.0.1", server_port=7860, share=False)
