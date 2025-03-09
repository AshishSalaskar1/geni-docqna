from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate

# Initialize Ollama locally deployed model
llm = Ollama(model="llava:7b")  # Change model as needed

# Web search tool using SerpAPI
search = SerpAPIWrapper()

def fetch_financial_news(stock_name):
    """Fetch latest financial news for the given stock name."""
    query = f"{stock_name} latest stock news site:moneycontrol.com OR site:economictimes.indiatimes.com OR site:reuters.com"
    return search.run(query)

# Define tool for LangChain agent
financial_news_tool = Tool(
    name="Financial News Fetcher",
    func=fetch_financial_news,
    description="Fetches the latest financial news for a given stock from financial websites."
)

# Initialize agent
agent = initialize_agent(
    tools=[financial_news_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def analyze_stock(stock_name):
    """Fetch and analyze stock data using LangChain and Ollama."""
    print(f"Fetching latest financial data for {stock_name}...")
    news_data = fetch_financial_news(stock_name)
    
    # Generate sentiment analysis prompt
    prompt = PromptTemplate(
        input_variables=["news"],
        template="""
        Based on the following latest financial news:
        {news}
        
        Provide a concise sentiment analysis of the stock's current market perception.
        Summarize the positive and negative aspects influencing the stock.
        """
    )
    
    sentiment_analysis = llm.invoke(prompt.format(news=news_data))
    
    return sentiment_analysis

if __name__ == "__main__":
    stock_name = input("Enter the stock name: ")
    result = analyze_stock(stock_name)
    print("\nSentiment Analysis Result:")
    print(result)
