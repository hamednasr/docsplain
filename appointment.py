import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.9,
    streaming=True,
    api_key=api_key
)

system_prompt = """You are a medical appointment scheduler. Follow these steps STRICTLY:

1. **Registration First**:
   - "Welcome to Doctor Appointment Scheduler. Please provide your full name:"
   - If name NOT in {patient_list}:
     - Add to {patient_list} and do not show the list of other patients.
     - "Thank you [Name], you're now registered."

2. **Specialty Selection**:
   - "What specialty do you need? Available: Cardiologist, Dermatologist, Pediatrician, Neurologist, General Practitioner"

3. **Show Availability**:
   - Display ALL available slots for chosen specialty from {doctor_data}
   - Example format:
     ```
     Dr. Sarah Thompson (Cardiologist):
     - Monday 10:00
     - Wednesday 14:00
     ```

4. **Schedule Confirmation**:
   - "Here are available times for Dr. [X]: [List slots]. Would you like to book one?"
     - If "Yes": Proceed with booking
     - If "No":
       - "Is this an emergency requiring immediate care? (Yes/No)"
         - If "Yes":
           - "Understood. What specific time do you need for emergency care?"
           - Book ANY time requested
           - "Emergency appointment confirmed for [Time]"
         - If "No":
           - "Let me check other availability..." [Show next options]

5. **Normal Confirmation**:
   - For regular bookings: "Your appointment with Dr. [X] at [Time] is confirmed."
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

patient_list = ["Ali Rezaei", "Sara Montazeri", "John Williams", "Fatemeh Ghasemi", "David Gilmore"]
doctor_data = [
    {
        "name": "Dr. Sarah Thompson",
        "specialty": "Cardiologist",
        "schedule": {
            "Monday": ["10:00", "11:00"],
            "Wednesday": ["14:00"],
            "Friday": ["09:00"]
        }
    },
    {
        "name": "Dr. Amir Khademi",
        "specialty": "Dermatologist",
        "schedule": {
            "Tuesday": ["09:00", "10:00"],
            "Thursday": ["11:00", "13:00"]
        }
    },
    {
        "name": "Dr. Emily Chen",
        "specialty": "Pediatrician",
        "schedule": {
            "Monday": ["09:00"],
            "Wednesday": ["10:00", "11:00"]
        }
    },
    {
        "name": "Dr. David Patel",
        "specialty": "Neurologist",
        "schedule": {
            "Tuesday": ["12:00"],
            "Thursday": ["10:00", "14:00"]
        }
    },
    {
        "name": "Dr. Leila Gol",
        "specialty": "General Practitioner",
        "schedule": {
            "Monday": ["08:00", "09:00"],
            "Wednesday": ["13:00", "14:00"]
        }
    }
]

chain = prompt | llm

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

print("Medical Appointment Scheduler (Type 'quit' to exit)")
print("-----------------------------------------------")

session_id = "user-session-001"  

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    response = runnable_with_history.invoke(
        {
            "input": user_input,
            "patient_list": patient_list,
            "doctor_data": doctor_data
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    print("\nAssistant:", response.content)
    print("-----------------------------------------------")