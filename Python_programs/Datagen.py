import random
import pandas as pd
from together import Together
import os

os.environ["TOGETHER_API_KEY"] = "d508a94fbfa184d49ded783bb12d8eff3510d94714904a4d58b653be84498391"
client = Together()

roles_skills = {
    "Data Scientist": {"description": "Analyze data, build predictive models, and communicate insights."},
    "Data Engineer": {"description": "Design and maintain data pipelines for scalable data processing."},
    "Software Engineer": {"description": "Develop and maintain software solutions and optimize performance."},
    "Product Manager": {"description": "Define product vision, strategy, and manage cross-functional teams."},
    "UI Engineer": {"description": "Create intuitive and visually appealing user interfaces."}
}

first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]

def generate_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

outcomes = ["selected", "rejected"]

reason_for_selection = [
    "Relevant skills and experience.",
    "Strong cultural fit.",
    "Excellent communication and interpersonal skills.",
    "Proven track record of achievements.",
    "Growth mindset and adaptability."
]

reason_for_rejection = [
    "Lack of relevant skills or experience.",
    "Poor cultural fit.",
    "Inadequate communication or interpersonal skills.",
    "Unsatisfactory references or background check.",
    "Lack of enthusiasm or motivation."
]

def generate_result():
    return random.choice(outcomes)

def generate_job_description(role):
    description_prompt = f"Generate a job description for the {role} role."
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": description_prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_resume(name, role, result):
    resume_prompt = (
        f"Generate a resume for {name}, who applied for the {role} role and was {result}. "
        f"Include skills, experience, achievements, and projects."
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": resume_prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_reason(result):
    reasons = reason_for_selection if result == "selected" else reason_for_rejection
    return random.choice(reasons)

def generate_transcript(name, role, result):
    performance = "performs well" if result == "selected" else "struggles"
    transcript_prompt = (
        f"Simulate an interview for a {role} role. The candidate, {name}, {performance} in demonstrating relevant skills."
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": transcript_prompt}]
    )
    return response.choices[0].message.content.strip()

data = []

for i in range(1, 501):
    candidate_id = f"uppaup{i}"
    name = generate_name()
    role = random.choice(list(roles_skills.keys()))
    result = generate_result()
    job_description = generate_job_description(role)
    resume = generate_resume(name, role, result)
    reason = generate_reason(result)
    transcript = generate_transcript(name, role, result)

    data.append({
        "ID": candidate_id,
        "Name": name,
        "Role": role,
        "Transcript": transcript,
        "Resume": resume,
        "Performance (select/reject)": result,
        "Reason for decision": reason,
        "Job Description": job_description
    })

df = pd.DataFrame(data)

df.to_excel("uppari_upendra_data.xlsx")

print("Dataset saved to 'uppari_upendra_data.xlsx'.")
