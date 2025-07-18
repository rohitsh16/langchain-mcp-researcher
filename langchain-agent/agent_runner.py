from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

def run_agent(query: str) -> str:
    """
    Runs a LangChain agent with the given query and returns the response.
    """
    tools = [
        Tool(
            name="Calculator",
            func=lambda x: str(eval(x)),
            description="Useful for simple math operations."
        )
    ]

    llm = OpenAI(temperature=0)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    response = agent.run(query)
    return response

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    result = run_agent(user_query)
    print("Agent response:", result)