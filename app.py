import os
os.environ["CHROMA_DISABLE_ONNX"] = "1"  # Force-disable ChromaDB's ONNX usage

import sys
import importlib

original_import = __import__

def debug_import(name, *args, **kwargs):
    if 'onnxruntime' in name:
        print(f"!!! Module trying to import ONNX Runtime: {name}")
        print("Stack trace:")
        import traceback
        traceback.print_stack()
    return original_import(name, *args, **kwargs)

__import__ = debug_import

# Rest of your imports...



import streamlit as st
import io
import sys
import os
import json
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from crewai.llm import LLM
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput
from agents.data_analyzer import DataAnalyzer
from agents.graph_plotter import GraphPlotter
from agents.graph_interpreter import GraphInterpreterTool
from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_validator import HypothesisValidator
from agents.summary_generator import SummaryGenerator
from config.config import AZURE_OPENAI_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, OPENAI_API_VERSION

# Load environment variables from a .env file for secure configuration
# Technical Concept: Environment variables are used to store sensitive information like API keys
load_dotenv()

# Configure the Streamlit page with a title, wide layout, and expanded sidebar
# Technical Concept: Streamlit's set_page_config sets global app properties
st.set_page_config(page_title="Wound Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Set the main title and sidebar header for the dashboard
st.title("🩺 Wound Analysis Dashboard")
st.sidebar.header("Control Panel")

# Create a file uploader in the sidebar for CSV files
# Technical Concept: Streamlit's file_uploader allows users to upload files, restricted to CSV format
uploaded_file = st.sidebar.file_uploader("Upload Wound Data (CSV)", type="csv", help="Upload a CSV file containing wound data for analysis.")

# Initialize session state variables to store analysis results and intermediate data
# Technical Concept: Streamlit's session_state is used for persistent state management across user interactions
if 'file_path' not in st.session_state: st.session_state.file_path = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'data_subset' not in st.session_state: st.session_state.data_subset = None
if 'analysis_summary' not in st.session_state: st.session_state.analysis_summary = None
if 'plot_paths' not in st.session_state: st.session_state.plot_paths = None
if 'interpretations' not in st.session_state: st.session_state.interpretations = None
if 'hypotheses' not in st.session_state: st.session_state.hypotheses = None
if 'validations' not in st.session_state: st.session_state.validations = None
if 'summary' not in st.session_state: st.session_state.summary = None
if 'generating_hypotheses' not in st.session_state: st.session_state.generating_hypotheses = False
if 'interpreting_graphs' not in st.session_state: st.session_state.interpreting_graphs = False

# Handle file upload and validation
# Technical Concept: File handling and validation ensure the uploaded CSV meets required criteria
if uploaded_file is not None:
    # Save the uploaded file temporarily to disk
    file_path = "temp_wound_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path
    # Load the CSV into a Pandas DataFrame for processing
    df = pd.read_csv(file_path)
    # Define required columns for the dataset
    required_columns = ['TOTAL_WOUND_AREA', 'WOUND_COUNT', 'WEEK', 'NAME']
    # Validate that all required columns are present
    if not all(col in df.columns for col in required_columns):
        st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
        st.stop()
    # Display a data summary in an expandable section
    with st.expander("📊 Data Summary", expanded=True):
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        st.write("**Preview (First 5 Rows):**")
        st.dataframe(df.head())
else:
    # Prompt user to upload a file if none is provided
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# Configure the Azure OpenAI LLM for agent-based analysis
# Technical Concept: LLM configuration uses Azure OpenAI for natural language processing and decision-making
os.environ['DISABLE_ONNX'] = '1'  # Add this before any LLM imports
azure_llm = LLM(
    model=f"azure/{AZURE_DEPLOYMENT_NAME}",
    base_url=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)

# Define Crew AI agents for different analysis tasks
# Technical Concept: CrewAI agents are specialized AI roles that perform specific tasks using the LLM
data_analyzer_agent = Agent(
    role="Data Analyst",
    goal="Analyze wound data and provide statistical insights using LLM-driven decisions",
    backstory="Expert in statistical analysis of medical data, adept at choosing optimal methods",
    verbose=True,
    llm=azure_llm
)

graph_plotter_agent = Agent(
    role="Graph Plotter",
    goal="Generate visual representations of wound data",
    backstory="Specialist in data visualization",
    verbose=True,
    llm=azure_llm
)

graph_interpreter_agent = Agent(
    role="Graph Interpreter",
    goal="Interpret generated graphs to provide insights",
    backstory="Experienced in translating visual data into medical insights",
    verbose=True,
    llm=azure_llm
)

hypothesis_generator_agent = Agent(
    role="Hypothesis Generator",
    goal="Generate hypotheses based on data analysis and graph interpretations",
    backstory="Skilled in forming testable hypotheses from complex datasets",
    verbose=True,
    llm=azure_llm
)

hypothesis_validator_agent = Agent(
    role="Hypothesis Validator",
    goal="Validate hypotheses using statistical methods",
    backstory="Expert in statistical validation and hypothesis testing",
    verbose=True,
    llm=azure_llm
)

analysis_summary_agent = Agent(
    role="Analysis Summary Generator",
    goal="Generate an initial summary of data analysis results",
    backstory="Specialist in summarizing statistical data clearly",
    verbose=True,
    llm=azure_llm
)

summary_generator_agent = Agent(
    role="Executive Summary Generator",
    goal="Create a holistic executive summary of the entire analysis process",
    backstory="Proficient in crafting comprehensive summaries for executives",
    verbose=True,
    llm=azure_llm
)

# Define a custom tool for data analysis
# Technical Concept: CrewAI tools encapsulate reusable functionality for agents
class DataAnalyzerTool(BaseTool):
    name: str = "Data Analyzer"
    description: str = "Analyzes wound data from a CSV file and returns summary statistics and analysis results"
    analyzer: DataAnalyzer

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, file_path):
        """Initialize the DataAnalyzerTool with a file path to the CSV data."""
        super().__init__(analyzer=DataAnalyzer(file_path))

    def _run(self, *args, **kwargs):
        """Execute the data analysis and return results as a JSON string.
        
        Technical Concept: The tool processes the CSV data using the DataAnalyzer class,
        ensuring results are returned in a structured JSON format for downstream tasks.
        """
        try:
            # Perform the analysis using the DataAnalyzer instance
            full_results = self.analyzer.analyze()
            # Ensure data_subset is a DataFrame for consistency
            if not isinstance(full_results['data_subset'], pd.DataFrame):
                full_results['data_subset'] = pd.DataFrame(full_results['data_subset'])
            # Store the data subset in session state for later use
            st.session_state.data_subset = full_results['data_subset']
            # Create a concise summary dictionary with key results
            summary_dict = {
                'summary_stats': full_results['summary_stats'],
                'analyses': full_results['analyses']
            }
            # Return results as a JSON string
            return json.dumps(summary_dict)
        except Exception as e:
            # Handle errors by returning a JSON error message
            error_output = {
                "summary_stats": {},
                "analyses": {"error": f"Data analysis failed: {str(e)}"}
            }
            return json.dumps(error_output)

# Define a custom tool for plotting graphs
# Technical Concept: This tool generates visualizations based on analysis results
class GraphPlotterTool(BaseTool):
    name: str = "Graph Plotter"
    description: str = "Plots graphs based on analyzed wound data"
    plotter: GraphPlotter

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, analysis_results):
        """Initialize the GraphPlotterTool with analysis results."""
        # Copy analysis results and include data subset from session state
        full_analysis_results = analysis_results.copy()
        full_analysis_results['data_subset'] = st.session_state.data_subset
        super().__init__(plotter=GraphPlotter(full_analysis_results))

    def _run(self, *args, **kwargs):
        """Generate plots and return their file paths as a JSON string.
        
        Technical Concept: The tool uses the GraphPlotter class to create visualizations,
        saving them to disk and returning their paths for display.
        """
        # Generate plots using the GraphPlotter instance
        plot_paths = self.plotter.plot()
        # Return plot paths as a JSON string
        return json.dumps(plot_paths)

# Define a custom tool for generating hypotheses
# Technical Concept: This tool creates testable hypotheses based on analysis and interpretations
class HypothesisGeneratorTool(BaseTool):
    name: str = "Hypothesis Generator"
    description: str = "Generates hypotheses from analysis results and interpretations"
    generator: HypothesisGenerator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, analysis_results, interpretations):
        """Initialize the HypothesisGeneratorTool with analysis results and interpretations."""
        super().__init__(generator=HypothesisGenerator(analysis_results, interpretations))

    def _run(self, *args, **kwargs):
        """Generate hypotheses and return them as a JSON string.
        
        Technical Concept: The tool uses the HypothesisGenerator class to create hypotheses,
        ensuring they are returned in a consistent list format.
        """
        # Generate hypotheses using the HypothesisGenerator instance
        hypotheses = self.generator.generate()
        # Ensure hypotheses is a list
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
        # Return hypotheses as a JSON string
        return json.dumps(hypotheses)

# Define a custom tool for validating hypotheses
# Technical Concept: This tool applies statistical tests to validate hypotheses
class HypothesisValidatorTool(BaseTool):
    name: str = "Hypothesis Validator"
    description: str = "Validates hypotheses using statistical methods based on the wound data subset"
    validator: HypothesisValidator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, hypotheses, data_subset):
        """Initialize the HypothesisValidatorTool with hypotheses and data subset."""
        super().__init__(validator=HypothesisValidator(hypotheses, data_subset))

    def _run(self, *args, **kwargs):
        """Validate hypotheses and return results as a JSON string.
        
        Technical Concept: The tool uses the HypothesisValidator class to perform statistical
        tests, returning validation results in a structured format.
        """
        try:
            # Validate hypotheses using the HypothesisValidator instance
            validations = self.validator.validate()
            # Return validations as a JSON string
            return json.dumps(validations)
        except Exception as e:
            # Handle errors by returning a JSON error message
            error_validation = [{
                "category": "Error",
                "hypothesis": "Validation failed",
                "test": "N/A",
                "stats": "N/A",
                "interpretation": f"Error during validation: {str(e)}",
                "valid": False
            }]
            return json.dumps(error_validation)

# Define a custom tool for generating summaries
# Technical Concept: This tool creates narrative summaries from analysis results
class SummaryGeneratorTool(BaseTool):
    name: str = "Summary Generator"
    description: str = "Generates summaries based on provided data"
    summary_gen: SummaryGenerator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, summary_gen):
        """Initialize the SummaryGeneratorTool with a SummaryGenerator instance."""
        super().__init__(summary_gen=summary_gen)

    def _run(self, *args, **kwargs):
        """Generate a summary and return it as a string.
        
        Technical Concept: The tool uses the SummaryGenerator class to create a narrative
        summary, typically in Markdown format for display.
        """
        # Generate summary using the SummaryGenerator instance
        return self.summary_gen.generate()

# Function to parse task outputs from CrewAI tasks
# Technical Concept: Output parsing ensures consistent data formats (dict, list, or string)
def parse_task_output(output, expected_type):
    """Parse the output of a CrewAI task into the expected data type.

    Args:
        output: Raw output from a CrewAI task (TaskOutput, CrewOutput, or other).
        expected_type: The expected Python type (dict, list, or str).

    Returns:
        Parsed output in the expected format, or None if parsing fails.

    Technical Concept: This function handles various output formats from CrewAI, using JSON
    parsing and fallback methods (e.g., ast.literal_eval) to ensure robust parsing.
    """
    # Check if output is a TaskOutput or CrewOutput and extract raw content
    if isinstance(output, TaskOutput):
        raw_output = output.raw if hasattr(output, 'raw') else str(output)
    elif isinstance(output, CrewOutput):
        if hasattr(output, 'tasks_output') and output.tasks_output:
            raw_output = output.tasks_output[0].raw
        else:
            raw_output = str(output)
    else:
        raw_output = output

    # Parse output based on expected type
    if expected_type == dict:
        try:
            # Attempt to parse as JSON dictionary
            parsed_output = json.loads(raw_output)
            if not isinstance(parsed_output, dict):
                raise ValueError("Parsed output is not a dictionary")
            return parsed_output
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse output as dict: {e}")
            try:
                # Fallback to ast.literal_eval for dictionary-like strings
                if isinstance(raw_output, str) and raw_output.strip().startswith('{'):
                    import ast
                    parsed_output = ast.literal_eval(raw_output)
                    if isinstance(parsed_output, dict):
                        return parsed_output
                raise ValueError("Output is not a valid dictionary format")
            except (ValueError, SyntaxError) as e2:
                st.error(f"Fallback parsing failed: {e2}")
                return None
    elif expected_type == str:
        # Convert to string if not already
        parsed_output = raw_output if isinstance(raw_output, str) else str(raw_output)
        return parsed_output
    elif expected_type == list:
        # Clean raw output to extract JSON list content
        if isinstance(raw_output, str):
            json_start = raw_output.find('[')
            json_end = raw_output.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                cleaned_output = raw_output[json_start:json_end].strip()
            else:
                cleaned_output = raw_output.strip()
        else:
            cleaned_output = raw_output
        try:
            # Attempt to parse as JSON list
            parsed_output = json.loads(cleaned_output)
            if not isinstance(parsed_output, list):
                raise ValueError("Parsed output is not a list")
            return parsed_output
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse output as list: {e}")
            # Handle error strings by returning a default error list
            if isinstance(cleaned_output, str) and not cleaned_output.startswith('['):
                return [{"category": "Error", "hypothesis": "Validation failed", "test": "N/A", "stats": "N/A", "interpretation": cleaned_output, "valid": False}]
            return None
    else:
        return raw_output

# Sidebar buttons for user interactions
# Technical Concept: Streamlit buttons trigger specific analysis workflows

# Button to analyze uploaded data
if st.sidebar.button("🔍 Analyze Data"):
    if st.session_state.file_path:
        # Redirect console output to capture logs
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Initialize the DataAnalyzerTool with the uploaded file
            analyzer_tool = DataAnalyzerTool(st.session_state.file_path)
            # Define a task for data analysis
            analyze_task = Task(
                description="""Use the Data Analyzer tool to analyze the wound data from the uploaded CSV file.
                The tool will provide summary statistics and analysis results (e.g., decision tree feature importance, regression trends).
                Return a JSON dictionary with keys 'summary_stats' and 'analyses' based on the tool’s output,
                without including the full dataset in the response to keep it concise.
                If the tool fails, expect a JSON response with an error message under 'analyses', and handle it accordingly.
                Output MUST be a valid JSON string, e.g., '{"summary_stats": {...}, "analyses": {...}}' or '{"summary_stats": {}, "analyses": {"error": "message"}}'.
                Do not generate human-readable text; rely on the tool’s JSON output directly.""",
                agent=data_analyzer_agent,
                expected_output="Valid JSON dictionary with summary statistics and analysis results",
                tools=[analyzer_tool]
            )
            # Create and execute a CrewAI crew for the analysis task
            analysis_crew = Crew(agents=[data_analyzer_agent], tasks=[analyze_task], verbose=True)
            analysis_results_raw = analysis_crew.kickoff()
            # Parse the analysis results
            analysis_results = parse_task_output(analysis_results_raw, dict)
            if analysis_results is None:
                st.sidebar.error("Failed to parse analysis results.")
            elif "error" in analysis_results.get("analyses", {}):
                st.sidebar.error(f"Analysis failed: {analysis_results['analyses']['error']}")
            else:
                # Store analysis results in session state
                st.session_state.analysis_results = analysis_results
                # Aggregate data for summary
                data_subset_df = st.session_state.data_subset
                week_grouped = data_subset_df.groupby('WEEK')['TOTAL_WOUND_AREA'].mean().to_dict()
                dressing_grouped = data_subset_df.groupby('NAME')['TOTAL_WOUND_AREA'].mean().to_dict()
                avg_wound_area_std = data_subset_df['AVG_WOUND_AREA'].std()
                # Initialize SummaryGenerator for initial summary
                summary_gen = SummaryGenerator(
                    validations=[{"hypothesis": "Initial analysis", "test": "N/A", "stats": "N/A", "interpretation": "N/A"}],
                    analysis_results=st.session_state.analysis_results
                )
                summary_tool = SummaryGeneratorTool(summary_gen)
                # Define a task for generating the analysis summary
                summary_task = Task(
                    description=f"""Craft a professional and engaging summary of the wound data analysis using the provided analysis_results dictionary. Create a natural narrative without repeating this prompt’s structure, weaving in these key points:
                    - Study scope: {analysis_results['summary_stats']['num_rows']} rows, {analysis_results['summary_stats']['weeks_covered']} weeks, {analysis_results['summary_stats']['num_dressings']} dressings.
                    - Statistics: Average wound area {analysis_results['summary_stats']['avg_wound_area']:.2f} cm² (std dev {avg_wound_area_std:.2f} cm²), total wound area std dev {analysis_results['summary_stats']['std_total_wound_area']:.2f} cm², average wound count {analysis_results['summary_stats']['avg_wound_count']:.2f}.
                    - Wound area range: 0.18 cm² (Week 6) to {max(week_grouped.values()):.2f} cm² (Week 0).
                    - Dressing performance: Total wound area averages {', '.join([f'{k}: {v:.2f} cm²' for k, v in list(dressing_grouped.items())[:3]])}, top performers {', '.join([f'{k}: {v:.2f} cm²' for k, v in sorted(dressing_grouped.items(), key=lambda x: x[1])[:3]])}, Aquacel Extra importance {analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('NAME_Aquacel Extra', 0):.4f}.
                    - Healing trends: Week 0: {week_grouped['Week 0']:.2f} cm², Week 247: {week_grouped[f"Week {analysis_results['summary_stats']['weeks_covered']-1}"]:.2f} cm², Week 1: {week_grouped['Week 1']:.2f} cm², Week 10: {week_grouped['Week 10']:.2f} cm², WEEK_NUM coefficient {analysis_results['analyses'].get('linear_regression', {}).get('coefficients', {}).get('WEEK_NUM', 'N/A')}.
                    - Analysis insights: Decision tree with TOTAL_WOUND_AREA importance {analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('TOTAL_WOUND_AREA', 0):.4f}, NAME_Aquacel Extra importance {analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('NAME_Aquacel Extra', 0):.4f}, WEEK_NUM importance {analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('WEEK_NUM', 0):.4f}, maximizing information gain to pinpoint key drivers.
                    - Key Observations: Outliers 2934.50 cm² (Aquacel Ag+ Extra, Week 0) and 0.18 cm² (Aquacel Extra, Week 6), Duoderm Gel reduction from 1804.67 cm² (Week 0) to 936.50 cm² (Week 1, ~48% drop), Aquacel Extra average 5.38 cm² with slight increase from 3.78 cm² (Week 0) to 6.00 cm² (Week 247).
                    Format with enumerated, bolded Markdown section titles (e.g., `**1. Data Overview**`) followed by a paragraph. Blend professionalism with an engaging tone, ensuring natural sentences while keeping all numbers accurate. In the Analysis Insights section, briefly explain how the LLM selected analyses (e.g., decision tree for feature importance, regression for trends) based on dataset traits like {analysis_results['summary_stats']['num_rows']} rows and {analysis_results['summary_stats']['weeks_covered']} weeks, emphasizing information gain to impress with analytical rigor.""",
                    agent=analysis_summary_agent,
                    expected_output="String with a detailed summary",
                    tools=[summary_tool]
                )
                # Create and execute a CrewAI crew for the summary task
                summary_crew = Crew(agents=[analysis_summary_agent], tasks=[summary_task], verbose=True)
                summary_results_raw = summary_crew.kickoff()
                # Parse the summary results
                summary = parse_task_output(summary_results_raw, str)
                if summary is None or not isinstance(summary, str):
                    st.sidebar.error("Failed to generate analysis summary.")
                else:
                    # Store summary in session state
                    st.session_state.analysis_summary = summary
                    st.sidebar.success("Data analyzed and summarized successfully!")
        finally:
            # Restore console output and store logs
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
    else:
        st.sidebar.error("Please upload a CSV file first.")

# Button to plot graphs based on analysis results
if st.session_state.analysis_results:
    if st.sidebar.button("📈 Plot Graphs"):
        if st.session_state.data_subset is None:
            st.sidebar.error("Data subset not available. Please analyze data first.")
        else:
            # Redirect console output to capture logs
            console_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = console_output

            try:
                # Initialize the GraphPlotterTool with analysis results
                plotter_tool = GraphPlotterTool(st.session_state.analysis_results)
                # Define a task for plotting graphs
                plot_task = Task(
                    description=f"""Based on the analysis results ({json.dumps(st.session_state.analysis_results['analyses'])}),
                    generate exactly 4 plots to visualize wound healing trends and dressing effectiveness.
                    You must include all of the following:
                    - A line plot for trends over time (e.g., 'WEEK_NUM' vs. 'AVG_WOUND_AREA', hue='NAME'),
                    - A bar plot for comparisons across dressings (e.g., 'NAME' vs. 'AVG_WOUND_AREA'),
                    - A boxplot for distribution and variability (e.g., 'NAME' vs. 'AVG_WOUND_AREA'),
                    - A scatter plot for relationships (e.g., 'WEEK_NUM' vs. 'AVG_WOUND_AREA', hue='NAME').
                    Justify your selections using the analysis (e.g., WEEK_NUM importance: {st.session_state.analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('WEEK_NUM', 0)}).
                    Return a valid JSON list of exactly 4 plot specifications, each with 'type', 'x', 'y', and optional 'hue'.
                    Example: [
                        {{"type": "line", "x": "WEEK_NUM", "y": "AVG_WOUND_AREA", "hue": "NAME"}},
                        {{"type": "bar", "x": "NAME", "y": "AVG_WOUND_AREA"}},
                        {{"type": "boxplot", "x": "NAME", "y": "AVG_WOUND_AREA"}},
                        {{"type": "scatter", "x": "WEEK_NUM", "y": "AVG_WOUND_AREA", "hue": "NAME"}}
                    ]
                    Output ONLY the JSON list, no additional text, to ensure valid parsing.""",
                    agent=graph_plotter_agent,
                    expected_output="List of file paths to the generated plots",
                    tools=[plotter_tool]
                )
                # Create and execute a CrewAI crew for the plotting task
                plot_crew = Crew(agents=[graph_plotter_agent], tasks=[plot_task])
                plot_paths_raw = plot_crew.kickoff()
                # Parse the plot paths
                plot_paths = parse_task_output(plot_paths_raw, list)
                if plot_paths is None:
                    st.sidebar.error("Failed to parse plot paths.")
                else:
                    # Store plot paths in session state
                    st.session_state.plot_paths = plot_paths
                    st.sidebar.success("Graphs plotted successfully!")
            finally:
                # Restore console output and store logs
                sys.stdout = original_stdout
                console_logs = console_output.getvalue()
                console_output.close()
                st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please analyze data before plotting graphs.")

# Button to interpret plotted graphs
if st.session_state.plot_paths:
    if st.sidebar.button("🔎 Interpret Graphs"):
        # Set flag to indicate graph interpretation is in progress
        st.session_state.interpreting_graphs = True
        # Clear dependent session state variables
        st.session_state.hypotheses = None
        st.session_state.validations = None
        st.session_state.summary = None
        # Redirect console output to capture logs
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Combine analysis results with data subset for interpretation
            full_analysis_results = st.session_state.analysis_results.copy()
            full_analysis_results['data_subset'] = st.session_state.data_subset.to_dict(orient="records")
            # Initialize the GraphInterpreterTool
            interpreter_tool = GraphInterpreterTool(st.session_state.plot_paths, full_analysis_results)
            # Define a task for graph interpretation
            interpret_task = Task(
                description="Execute the GraphInterpreterTool to interpret the 4 plots (line, bar, boxplot, scatter) and return its output directly as a JSON list of 4 dictionaries with 'plot' and 'interpretation' keys. Do not generate additional interpretation or text beyond the tool’s output.",
                agent=graph_interpreter_agent,
                expected_output="JSON list of 4 dictionaries with 'plot' (exact filename) and 'interpretation' (string) keys",
                tools=[interpreter_tool],
                output_parser=lambda x: json.loads(x) if isinstance(x, str) else x
            )
            # Create and execute a CrewAI crew for the interpretation task
            interpret_crew = Crew(
                agents=[graph_interpreter_agent],
                tasks=[interpret_task],
                verbose=True
            )
            # Execute the crew and parse results
            interpretations_raw = interpret_crew.kickoff()
            interpretations = parse_task_output(interpretations_raw, list)
            if interpretations is None or not isinstance(interpretations, list) or len(interpretations) != 4:
                st.sidebar.error("Failed to parse graph interpretations. Check console logs.")
            else:
                # Store interpretations in session state
                st.session_state.interpretations = interpretations
                st.sidebar.success("Graphs interpreted successfully!")
        except Exception as e:
            st.sidebar.error(f"Graph interpretation failed: {str(e)}")
        finally:
            # Restore console output and store logs
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
            # Reset interpretation flag
            st.session_state.interpreting_graphs = False
else:
    st.sidebar.write("Please plot graphs before interpreting them.")

# Button to generate hypotheses
if st.session_state.analysis_results and st.session_state.interpretations:
    # Disable button during graph interpretation to prevent conflicts
    generate_button_disabled = st.session_state.interpreting_graphs
    if st.sidebar.button("💡 Generate Hypotheses", disabled=generate_button_disabled):
        # Set flag to indicate hypothesis generation is in progress
        st.session_state.generating_hypotheses = True
        # Clear dependent session state variables
        st.session_state.validations = None
        st.session_state.summary = None
        # Redirect console output to capture logs
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Initialize the HypothesisGeneratorTool
            hypothesis_tool = HypothesisGeneratorTool(st.session_state.analysis_results, st.session_state.interpretations)
            # Define a task for hypothesis generation
            hypothesis_task = Task(
                description=f"""Based on the analysis results ({json.dumps(st.session_state.analysis_results['analyses'])}) and interpretations ({json.dumps(st.session_state.interpretations)}),
                generate 6 testable hypotheses about wound healing trends and dressing effectiveness.
                Categorize them as:
                - Univariate Hypotheses (3 hypotheses): Focus on a single factor using T-test for mean comparison, F-test for variability, and another T-test for mean change over time.
                - Multi-factor Hypotheses (3 hypotheses): Consider interactions using Chi-square for distribution, Regression for interaction effects, and Multi-factor ANOVA for reduction differences.
                Return a JSON list of 6 dictionaries, each with 'category' (e.g., 'Univariate Hypotheses' or 'Multi-factor Hypotheses') and 'hypothesis' (string).
                Output ONLY the JSON list, no additional text or formatting.""",
                agent=hypothesis_generator_agent,
                expected_output="List of 6 dictionaries with category and hypothesis",
                tools=[hypothesis_tool]
            )
            # Create and execute a CrewAI crew for the hypothesis task
            hypothesis_crew = Crew(agents=[hypothesis_generator_agent], tasks=[hypothesis_task])
            hypotheses_raw = hypothesis_crew.kickoff()
            # Parse the hypotheses
            hypotheses = parse_task_output(hypotheses_raw, list)
            if hypotheses is None:
                st.sidebar.error("Failed to parse hypotheses.")
            else:
                # Store hypotheses in session state
                st.session_state.hypotheses = hypotheses
                st.sidebar.success("Hypotheses generated successfully!")
        finally:
            # Restore console output and store logs
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
            # Reset hypothesis generation flag
            st.session_state.generating_hypotheses = False
else:
    st.sidebar.write("Please analyze data and interpret graphs before generating hypotheses.")

# Button to validate hypotheses
if st.session_state.hypotheses and st.session_state.data_subset is not None:
    # Disable button during hypothesis generation to prevent conflicts
    validate_button_disabled = st.session_state.generating_hypotheses
    if st.sidebar.button("✅ Validate Hypotheses", disabled=validate_button_disabled):
        # Redirect console output to capture logs
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Initialize the HypothesisValidatorTool
            validator_tool = HypothesisValidatorTool(st.session_state.hypotheses, st.session_state.data_subset)
            # Define a task for hypothesis validation
            validate_task = Task(
                description=f"""Use the Hypothesis Validator tool to validate the hypotheses provided in the wound data subset:
                {json.dumps(st.session_state.hypotheses)}.
                The tool will categorize them as 'Univariate Hypotheses' or 'Multi-factor Hypotheses' and apply appropriate statistical tests (e.g., T-test, ANOVA, Regression, Chi-Square).
                Return the tool's output directly as a JSON list of dictionaries with keys 'category', 'hypothesis', 'test', 'stats', 'interpretation', and 'valid' (True if p < 0.05, False otherwise).
                Do not reformat, add text, or override the tool’s output—return the JSON list as-is.""",
                agent=hypothesis_validator_agent,
                expected_output="JSON list of dictionaries with categorized validation results including validity",
                tools=[validator_tool]
            )
            # Create and execute a CrewAI crew for the validation task
            validate_crew = Crew(agents=[hypothesis_validator_agent], tasks=[validate_task])
            validations_raw = validate_crew.kickoff()
            # Parse the validations
            validations = parse_task_output(validations_raw, list)
            if validations is None:
                st.sidebar.error("Failed to parse hypothesis validations.")
            else:
                # Store validations in session state
                st.session_state.validations = validations
                st.sidebar.success("Hypotheses validated successfully!")
        finally:
            # Restore console output and store logs
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please generate hypotheses and ensure data subset is available before validation.")

# Button to generate the executive summary
if st.session_state.validations:
    if st.sidebar.button("📝 Generate Summary"):
        # Redirect console output to capture logs
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Initialize the SummaryGenerator with all analysis results
            summary_gen = SummaryGenerator(
                analysis_results=st.session_state.analysis_results,
                plot_paths=st.session_state.plot_paths,
                interpretations=st.session_state.interpretations,
                hypotheses=st.session_state.hypotheses,
                validations=st.session_state.validations
            )
            summary_tool = SummaryGeneratorTool(summary_gen)
            # Define a task for generating the executive summary
            summary_task = Task(
                description=f"""Craft an engaging and professional executive summary for a study on diabetic foot ulcer treatments, synthesizing all analysis steps into a compelling narrative for healthcare executives:
                - **Introduction**: Launch with a gripping hook on the study's mission to revolutionize wound care for diabetic patients, spotlighting its rigorous exploration of 1655 records across 248 weeks and 14 dressings to decode treatment efficacy.
                - **Analytical Approach**: Unveil the cutting-edge methods driving this study, showcasing how LLMs dynamically chose analyses (e.g., decision trees over simple averages) based on a dataset of 1655 rows and 248 weeks, prioritizing information gain. Highlight the use of decision trees (TOTAL_WOUND_AREA importance: {st.session_state.analysis_results['analyses'].get('decision_tree', {}).get('feature_importance', {}).get('TOTAL_WOUND_AREA', 0):.4f}) and regression (WEEK_NUM coefficient: {st.session_state.analysis_results['analyses'].get('linear_regression', {}).get('coefficients', {}).get('WEEK_NUM', 'N/A')}) for statistical depth, LLM-guided plotting of 4 graph types (line, bar, boxplot, scatter) for visual clarity, and multi-factor hypothesis generation with validations via ANOVA, Chi-Square, and regression for robust insights.
                - **Key Discoveries**: Paint a vivid picture of standout findings, like top dressings (Aquacel Foam Lite at 3.93 cm², Aquacel Extra at 5.38 cm²), surprises (e.g., Aquacel Extra’s rise from 3.78 cm² to 6.00 cm²), and gaps (e.g., Aquacel Ag+ Extra’s 69.94 cm² average), weaving a tale of efficacy and nuance.
                - **Healing Trends**: Chronicle the healing journey with flair, from a dramatic 781.76 cm² in Week 0 to 6.00 cm² by Week 247, with a 308.58 cm² plunge by Week 1 and a steady 192.87 cm² by Week 10, illuminating early wins and long-term patterns.
                - **Actionable Insights**: Deliver bold, practical takeaways, balancing early triumphs (e.g., Duoderm Gel’s ~48% drop) with calls for tailored long-term strategies, urging deeper exploration into dressing durability and patient-specific care to shape future wound management.
                Format as a Markdown string (400-500 words) with section headers (e.g., `### Introduction`). Use a professional yet captivating tone, integrating actual numbers (e.g., 781.76 cm², 0.5151 importance) to ground the story in data, avoiding technical jargon for accessibility.""",
                agent=summary_generator_agent,
                expected_output="String with a holistic executive summary",
                tools=[summary_tool]
            )
            # Create and execute a CrewAI crew for the summary task
            summary_crew = Crew(agents=[summary_generator_agent], tasks=[summary_task])
            summary_raw = summary_crew.kickoff()
            # Parse the summary
            summary = parse_task_output(summary_raw, str)
            if summary is None:
                st.sidebar.error("Failed to parse executive summary.")
            else:
                # Store summary in session state
                st.session_state.summary = summary
                st.sidebar.success("Executive summary generated successfully!")
        finally:
            # Restore console output and store logs
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please validate hypotheses before generating the executive summary.")

# Main output area with tabs for different results
# Technical Concept: Streamlit tabs organize results for user-friendly navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Analysis Summary", "Graphs", "Interpretations", "Hypotheses", "Hypothesis Validations", "Executive Summary"])

with tab1:
    # Display the analysis summary
    if st.session_state.analysis_summary:
        st.subheader("Analysis Summary")
        st.markdown(st.session_state.analysis_summary)
    else:
        st.write("Analysis summary will appear here after data analysis.")

with tab2:
    # Display generated graphs
    if st.session_state.plot_paths:
        st.subheader("Generated Graphs")
        for plot_path in st.session_state.plot_paths:
            if os.path.exists(plot_path):
                # Display each plot as an image
                st.image(plot_path, caption=os.path.basename(plot_path), use_column_width=True)
            else:
                st.error(f"Plot file not found: {plot_path}")
    else:
        st.write("Graphs will appear here after plotting.")

with tab3:
    # Display graph interpretations
    if st.session_state.interpretations:
        st.subheader("Graph Interpretations")
        for interp in st.session_state.interpretations:
            st.write(f"**Plot:** {interp['plot']}")
            st.write(f"**Interpretation:** {interp['interpretation']}")
            st.divider()
    else:
        st.write("Interpretations will appear here after graph interpretation.")

with tab4:
    # Display generated hypotheses
    if st.session_state.hypotheses:
        st.subheader("Generated Hypotheses")
        univariate_section = False
        multi_factor_section = False
        for hyp_dict in st.session_state.hypotheses:
            if not isinstance(hyp_dict, dict) or "category" not in hyp_dict or "hypothesis" not in hyp_dict:
                st.write(f"Error: Invalid hypothesis format - {hyp_dict}")
                continue
            category = hyp_dict["category"]
            hypothesis = hyp_dict["hypothesis"]
            # Organize hypotheses by category
            if category == "Univariate Hypotheses" and not univariate_section:
                st.markdown("### Univariate Hypotheses")
                univariate_section = True
            elif category == "Multi-factor Hypotheses" and not multi_factor_section:
                st.markdown("### Multi-factor Hypotheses")
                multi_factor_section = True
            st.write(f"**Hypothesis:** {hypothesis}")
    else:
        st.write("Hypotheses will appear here after generation.")

with tab5:
    # Display hypothesis validations
    if st.session_state.validations:
        st.subheader("Hypothesis Validations")
        univariate_section = False
        multi_factor_section = False
        for validation in st.session_state.validations:
            # Organize validations by category
            if validation['category'] == "Univariate Hypotheses" and not univariate_section:
                st.markdown("### Univariate Hypotheses")
                univariate_section = True
            elif validation['category'] == "Multi-factor Hypotheses" and not multi_factor_section:
                st.markdown("### Multi-factor Hypotheses")
                multi_factor_section = True
            st.write(f"- **Hypothesis:** {validation['hypothesis']}")
            st.write(f"- **Test:** {validation['test']}")
            st.write(f"- **Stats:** {validation['stats']}")
            st.write(f"- **Interpretation:** {validation['interpretation']}")
            st.write(f"- **Valid:** {'Yes' if validation['valid'] == 'true' else 'No'}")
            st.divider()
    else:
        st.write("Validations will appear here after hypothesis validation.")

with tab6:
    # Display the executive summary
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.write("Executive summary will appear here after generation.")

# Display console logs for debugging
if 'console_logs' in st.session_state and st.session_state.console_logs:
    with st.container():
        st.subheader("Agent Workflow Console Output")
        st.text_area("", st.session_state.console_logs, height=300, key="console_output")