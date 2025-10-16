import google.generativeai as genai

# Reusable function
def ask_model(prompt, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


# Academic Advisor Agent
def academic_advisor(student_info):
    prompt = f"""
    You are an Academic Advisor.
    The student info: {student_info}
    Suggest courses, subjects, or majors that match their interests and academic strengths.
    """
    return ask_model(prompt)


# Career Counselor Agent
def career_counselor(student_info, advisor_output):
    prompt = f"""
    You are a Career Counselor.
    Student info: {student_info}
    Academic Advisor Suggestions: {advisor_output}

    Based on this, suggest 2-3 possible career paths and skills to build.
    """
    return ask_model(prompt)


# Coordinator Agent (merges both)
def coordinator(student_info, advisor_output, counselor_output):
    prompt = f"""
    You are a Coordinator Agent.
    Student Info: {student_info}
    Academic Advisor Suggestions: {advisor_output}
    Career Counselor Suggestions: {counselor_output}

    Write a final summary that combines both perspectives
    into a clear roadmap for the student.
    """
    return ask_model(prompt)
