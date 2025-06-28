from fastapi import FastAPI, Header
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pygsheets
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import SystemMessage, AnyMessage
# from langgraph.pregel import RetryPolicy
from langgraph.types import RetryPolicy
import json
from google.oauth2 import service_account
import os
from langchain_groq import ChatGroq
import groq
from datetime import datetime
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from opik.integrations.langchain import OpikTracer
from pytz import timezone 

# Load environment variables - for local development
from dotenv import load_dotenv
load_dotenv()

SHEET_URL = os.getenv("SHEET_URL")
GOOGLESHEETS_CREDENTIALS = os.getenv("GOOGLESHEETS_CREDENTIALS")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GOOGLE_MODEL = "gemini-2.5-flash-preview-05-20"
ist_tz = timezone("Asia/Kolkata")

class TransactionParser(BaseModel):
    """This Pydantic class is used to parse the transaction message. The message is taken and the output is structured in a specific format based upon below definitions."""

    amount: str = Field(description="The amount of the transaction strictly in decimal format. Do not insert currency symbol.", example="123.45")
    dr_or_cr: str = Field(description="Identify if the transaction was debit (spent) or credit (received). Strictly choose one of the values - Debit or Credit")
    receiver: str = Field(description="The recipient of the transaction. Identify the Merchant Name from the message text.")
    category: str = Field(description="The category of the transaction. The category of the transaction is linked to the Merchant Name. Strictly choose from one the of values - Shopping,EMI,Education,Miscellaneous,Grocery,Utility,House Help,Travel,Transport,Food")
    # transaction_date: str = Field(description="Use today's date strictly in yyyy-mm-dd format.")
    transaction_origin: str = Field(description="The origin of the transaction. Provide the card or account number as well.")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("classify_txn_type", self.classify_txn_type, retry=RetryPolicy(retry_on=[groq.APIConnectionError], max_attempts=5))
        graph.add_node("parse_message", self.parse_message, retry=RetryPolicy(retry_on=[groq.APIConnectionError], max_attempts=5))
        graph.add_node("write_message", self.write_message)
        graph.add_conditional_edges(
            "classify_txn_type",
            self.check_txn_and_decide,
            {True: "parse_message", False: END}
            )
        graph.add_edge("parse_message", "write_message")
        graph.add_edge("write_message", END)
        graph.set_entry_point("classify_txn_type")
        self.graph = graph.compile()
        self.model = model

    def classify_txn_type(self, state: AgentState) -> AgentState:
        print(f"{datetime.now(ist_tz)}: Classifying transaction type...")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = self.model.invoke(messages)
        print(f"{datetime.now(ist_tz)}: Classifying transaction type completed.")
        return {"messages": [message]}
    
    def parse_message(self, state: AgentState) -> AgentState:
        print(f"{datetime.now(ist_tz)}: Parsing transaction message...")
        message = state["messages"][0]#.content
        system = """
        You are a helpful assistant skilled at parsing transaction messages and providing structured responses.
        """
        human = "Categorize the transaction message and provide the output in a structed format: {topic}"

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        chain = prompt | self.model.with_structured_output(TransactionParser)
        result = chain.invoke({"topic": message})
        print(f"{datetime.now(ist_tz)}: Parsing transaction message completed.")   
        
        return {"messages": [result]}

    def write_message(self, state: AgentState) -> AgentState:
        print(f"{datetime.now(ist_tz)}: Writing transaction message to Google Sheets...")
        result = state["messages"][-1]

        SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')

        service_account_info = json.loads(GOOGLESHEETS_CREDENTIALS)
        credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        client = pygsheets.authorize(custom_credentials=credentials)
        worksheet = client.open_by_url(SHEET_URL)
        wk = worksheet[0]
        # Get number of rows in the worksheet
        df = wk.get_as_df(start='A1', end='G999')
        nrows = df.shape[0]
        wk.update_value(f'A{nrows+2}', result.amount)
        wk.update_value(f'B{nrows+2}', result.dr_or_cr)
        wk.update_value(f'C{nrows+2}', result.receiver)
        wk.update_value(f'D{nrows+2}', result.category)
        wk.update_value(f'E{nrows+2}', datetime.now(ist_tz).strftime("%Y-%m-%d"))
        wk.update_value(f'F{nrows+2}', result.transaction_origin)
        wk.update_value(f'G{nrows+2}', state["messages"][0])
        print(f"{datetime.now(ist_tz)}: Writing transaction message to Google Sheets completed.")
        return {"messages": ["Transaction Completed"]}
        
    def check_txn_and_decide(self, state: AgentState):
        try:
            result = json.loads(state['messages'][-1].content)['classification']
        except json.JSONDecodeError:
            result = state['messages'][-1].content.strip()

        return result == "Transaction"
    

app = FastAPI()

@app.get("/")
def greetings():
    return {"message": "Hello, this is a transaction bot. Please send a POST request to /write_message with the transaction data."}

@app.post("/write_message")
def write_message(data: dict, header: str = Header()):
    if header != HF_TOKEN:
        raise HTTPException(status_code=400, detail="Invalid header")

    prompt = """You are a smart assistant adept at classifying different messages. \
    You will be penalized heavily for incorrect classification. \
    Your task is to classify the message into one of the following categories: \
    Transaction, OTP, Promotional, Scheduled, Non-Transaction. \
    Consider Salary Credit, Interest Credit, Redemptions as non-transactional messages.
    Output the classification in a structured format like below. \
    {"classification": "OTP"} \
    """

    message = data['message']
    
    try:
        model = ChatGoogleGenerativeAI(model=GOOGLE_MODEL, max_retries=3, callbacks = [OpikTracer()])
    except Exception as e: #fallback model
        model = ChatGroq(model=GROQ_MODEL, temperature=1, callbacks = [OpikTracer()])
    # model = ChatOllama(model="gemma3:1b", temperature=1)
    transaction_bot = Agent(model, system=prompt)
    transaction_bot.graph.invoke({"messages": [message]})
    return {"message": "Transaction completed successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")