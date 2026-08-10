import ollama

context = """
Chunk 1
SANJANA C A
sanjanaca622@gmail.com | +91 9148132577 | linkedin.com/in/sanjana-c-a-gcu3012-6a11762ab |
github.com/Sanjanaca150

PROFILE
Aspiring Software Engineer and Information Science student with strong expertise in Java, Python, MySQL, and Data Structures.

Chunk 2
Developing CRUD applications, Python automation tools, and full-stack web projects. Seeking an engineering internship to apply technical problem-solving skills and gain hands-on software development experience.

TECHNICAL SKILLS
• Programming Languages: Java, SQL, Python
• Database Technologies:

Chunk 3
MySQL, DBMS
• Web Technologies: HTML, CSS, JavaScript
• Tools & Platforms: Git, GitHub, Eclipse IDE
• Core Concepts: OOP, Data Structures, Exception Handling

EDUCATION
Garden City University, Bengaluru
• Bachelor of Engineering (Information Science & Engineering)
• Expected Graduation: 2027
"""

question = "What technical skills does Sanjana have?"

prompt = f"""
You are a helpful assistant.

Answer ONLY using the information provided in the context.

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== LLM Answer =====\n")
print(response["message"]["content"])