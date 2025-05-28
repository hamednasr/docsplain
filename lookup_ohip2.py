from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
import asyncio

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class BillingCodeAgent:
    def __init__(self):
        self.assistant = client.beta.assistants.create(
            name="OHIP Code Expert",
            instructions="""You must return OHIP codes in EXACTLY this format:
            CODE\\tDESCRIPTION\\tFEE
            Found in\\n\\nCATEGORY
            
            Example:
            A007\\tIntermediate assessment or well baby care\\t$37.95
            Found in\\n\\nConsultations and Visits\\n/\\nFamily Practice & Practice in General (00)
            
            Never add any extra text or explanations""",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
    
    async def execute_agent(self, code: str) -> str:
        try:
            thread = client.beta.threads.create()
            
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"""Return ONLY this exact format for code {code}:
                CODE\\tDESCRIPTION\\tFEE
                Found in\\n\\nCATEGORY
                
                Example format:
                A007\\tIntermediate assessment or well baby care\\t$37.95
                Found in\\n\\nConsultations and Visits\\n/\\nFamily Practice & Practice in General (00)"""
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                await asyncio.sleep(2)
            
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
            
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python ohip_agent.py <billing_code>")
            sys.exit(1)
            
        agent = BillingCodeAgent()
        result = await agent.execute_agent(sys.argv[1])
        
        result = result.replace("CODE\t", "").replace("    ", "\t")
        print(result)
    
    asyncio.run(main())