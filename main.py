import os
import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_roadmap(name, age, education, interests, goal):
    student_info = {
        "name": name,
        "age": age,
        "education": education,
        "interests": interests,
        "goal": goal
    }

    # Prompts
    academic_prompt = f"""
    You are an Academic Advisor. Based on the student's profile:
    {student_info}
    Suggest a clear semester-by-semester learning plan with courses, skills, and certifications.
    """

    career_prompt = f"""
    You are a Career Mentor. Based on the student's profile:
    {student_info}
    Suggest internships, projects, and career roadmap steps to achieve the goal.
    """

    # Generate AI responses
    academic_plan = model.generate_content(academic_prompt).text
    career_plan = model.generate_content(career_prompt).text

    final_plan = f"""
📘 Academic Plan:
{academic_plan}

💼 Career Plan:
{career_plan}
"""

    # Save TXT
    txt_filename = f"{student_info['name']}_roadmap.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(final_plan)

    # Save PDF
    pdf_filename = f"{student_info['name']}_roadmap.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Career & Academic Roadmap for {student_info['name']}")
    c.setFont("Helvetica", 12)

    y_position = height - 100
    for line in final_plan.split("\n"):
        if y_position < 50:  # new page
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 12)
        c.drawString(50, y_position, line)
        y_position -= 20

    c.save()

    return txt_filename, pdf_filename


# ==== GUI ====
def on_submit():
    name = entry_name.get()
    age = entry_age.get()
    education = entry_education.get()
    interests = entry_interests.get()
    goal = entry_goal.get()

    if not all([name, age, education, interests, goal]):
        messagebox.showerror("Error", "Please fill all fields")
        return

    txt_file, pdf_file = generate_roadmap(name, age, education, interests, goal)
    messagebox.showinfo("Success", f"✅ Roadmap saved as:\n{txt_file}\n{pdf_file}")


root = tk.Tk()
root.title("AI Career & Academic Roadmap Generator")
root.geometry("400x400")

tk.Label(root, text="Name:").pack()
entry_name = tk.Entry(root, width=40)
entry_name.pack()

tk.Label(root, text="Age:").pack()
entry_age = tk.Entry(root, width=40)
entry_age.pack()

tk.Label(root, text="Education:").pack()
entry_education = tk.Entry(root, width=40)
entry_education.pack()

tk.Label(root, text="Interests:").pack()
entry_interests = tk.Entry(root, width=40)
entry_interests.pack()

tk.Label(root, text="Goal:").pack()
entry_goal = tk.Entry(root, width=40)
entry_goal.pack()

tk.Button(root, text="Generate Roadmap", command=on_submit, bg="green", fg="white").pack(pady=20)

root.mainloop()
