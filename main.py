import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from agents.data_analyzer import DataAnalyzer
from agents.graph_plotter import GraphPlotter
from agents.graph_interpreter import GraphInterpreterTool
from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_validator import HypothesisValidator
from agents.summary_generator import SummaryGenerator
import json
from config.config import AZURE_OPENAI_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, OPENAI_API_VERSION
from crewai.llm import LLM

# Load environment variables
load_dotenv()

# Define Azure OpenAI LLM configuration
azure_llm = LLM(
    model=f"azure/{AZURE_DEPLOYMENT_NAME}",
    base_url=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)

# Define CrewAI agent (shared for simplicity)
graph_interpreter_agent = Agent(
    role="Graph Interpreter",
    goal="Interpret generated graphs to provide data-driven insights",
    backstory="Experienced in translating visual wound healing data into actionable insights",
    verbose=True,
    llm=azure_llm
)

def main():
    # Step 1: Load and analyze the data
    print("Analyzing data...")
    analyzer = DataAnalyzer("data/wound_data.csv")
    analysis_results = analyzer.analyze()
    print("Data analysis complete.")

    # Step 2: Plot graphs based on analysis results
    print("Plotting graphs...")
    plotter = GraphPlotter(analysis_results)
    plot_paths = plotter.plot()
    print("Graphs plotted successfully.")

    # Step 3: Interpret the graphs using CrewAI
    print("Interpreting graphs...")
    interpreter_tool = GraphInterpreterTool(plot_paths, analysis_results)
    interpret_task = Task(
        description="Use the GraphInterpreterTool to interpret the 4 plots and return its output directly as a JSON list of 4 dictionaries with 'plot' and 'interpretation' keys. Do not add extra interpretation or text beyond the tool’s output.",
        agent=graph_interpreter_agent,
        expected_output="List of 4 dictionaries with plot names and their interpretations",
        tools=[interpreter_tool],
        output_parser=lambda x: json.loads(x) if isinstance(x, str) else x  # Ensure JSON parsing
    )
    interpret_crew = Crew(agents=[graph_interpreter_agent], tasks=[interpret_task])
    interpretations_raw = interpret_crew.kickoff()
    interpretations = json.loads(interpretations_raw) if isinstance(interpretations_raw, str) else interpretations_raw
    print("Graph interpretation complete.")

    # Step 4: Generate hypotheses based on analysis and interpretations
    print("Generating hypotheses...")
    generator = HypothesisGenerator(analysis_results, interpretations)
    hypotheses = generator.generate()
    print("Hypotheses generated.")

    # Step 5: Validate the hypotheses
    print("Validating hypotheses...")
    validator = HypothesisValidator(hypotheses, analysis_results["data_subset"])
    validations = validator.validate()
    print("Hypothesis validation complete.")

    # Step 6: Generate the executive summary
    print("Generating executive summary...")
    summary_generator = SummaryGenerator(validations)
    summary = summary_generator.generate()
    print("Executive summary generated.")

    # Step 7: Print results (for testing)
    print("\n=== Interpretations ===")
    for interp in interpretations:
        print(json.dumps(interp))
    print("\n=== Hypotheses ===")
    for hyp in hypotheses:
        print(json.dumps(hyp))
    print("\n=== Validations ===")
    for val in validations:
        print(json.dumps(val))
    print("\n=== Executive Summary ===")
    print(summary)
    print("=========================")

if __name__ == "__main__":
    main()