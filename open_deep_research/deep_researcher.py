"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    build_chat_model,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)


def _trace_message_update(message: AIMessage) -> dict[str, dict[str, list[AIMessage] | str]]:
    """Expose raw model messages for LangGraph message streaming and token tracing."""
    return {
        "trace_messages": {
            "type": "override",
            "value": [message],
        }
    }


def _build_final_report_findings(notes: list[str], raw_notes: list[str]) -> str:
    """Combine compressed findings with raw evidence while keeping their roles explicit."""
    sections: list[str] = []

    if notes:
        sections.append("Compressed findings:\n" + "\n\n".join(notes))

    if raw_notes:
        sections.append("Raw evidence:\n" + "\n\n".join(raw_notes))

    return "\n\n".join(sections)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    clarification_model = (
        build_chat_model(
            configurable.research_model,
            configurable.research_model_max_tokens,
            get_api_key_for_model(configurable.research_model, config),
            tags=[TAG_NOSTREAM],
        )
        .with_structured_output(ClarifyWithUser, include_raw=True)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    structured_response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    if structured_response["parsing_error"] is not None:
        raise structured_response["parsing_error"]
    response = structured_response["parsed"]
    raw_response = structured_response["raw"]

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={
                "messages": [AIMessage(content=response.question)],
                **_trace_message_update(raw_response),
            }
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={
                "messages": [AIMessage(content=response.verification)],
                **_trace_message_update(raw_response),
            }
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    # Configure model for structured research question generation
    research_model = (
        build_chat_model(
            configurable.research_model,
            configurable.research_model_max_tokens,
            get_api_key_for_model(configurable.research_model, config),
            tags=[TAG_NOSTREAM],
        )
        .with_structured_output(ResearchQuestion, include_raw=True)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    structured_response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    if structured_response["parsing_error"] is not None:
        raise structured_response["parsing_error"]
    response = structured_response["parsed"]
    raw_response = structured_response["raw"]
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            **_trace_message_update(raw_response),
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        build_chat_model(
            configurable.research_model,
            configurable.research_model_max_tokens,
            get_api_key_for_model(configurable.research_model, config),
            tags=[TAG_NOSTREAM],
        )
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Stop exactly at the configured supervisor iteration ceiling.
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        # Limit concurrent research units to prevent resource exhaustion
        allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

        # Execute research tasks in parallel while preserving successful siblings.
        research_tasks = [
            researcher_subgraph.ainvoke({
                "messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config)
            for tool_call in allowed_conduct_research_calls
        ]

        tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)
        successful_observations = []
        child_errors = []

        for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
            if isinstance(observation, Exception):
                child_errors.append(observation)
                all_tool_messages.append(ToolMessage(
                    content=f"Error running delegated research: {observation}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
                continue

            successful_observations.append(observation)
            all_tool_messages.append(ToolMessage(
                content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))

        # Handle overflow research calls with error messages
        for overflow_call in overflow_conduct_research_calls:
            all_tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_call["id"]
            ))

        # Preserve successful raw evidence even when sibling tasks fail.
        raw_notes_concat = "\n".join([
            "\n".join(observation.get("raw_notes", []))
            for observation in successful_observations
        ])

        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

        if child_errors and not successful_observations and all(
            is_token_limit_exceeded(error, configurable.research_model) for error in child_errors
        ):
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher_agent_node(state: ResearcherState, config: RunnableConfig) -> dict:
    """Run a create_agent ReAct loop for a single research topic.

    Replaces the old hand-rolled researcher + researcher_tools nodes.
    create_agent handles the LLM-call → tool-execution → loop cycle internally.
    The result messages are written back to state so compress_research can read them.

    Args:
        state: Current researcher state containing the initial research message
        config: Runtime configuration with model settings and tool availability

    Returns:
        Dictionary updating messages with the full agent conversation
    """
    configurable = Configuration.from_runnable_config(config)

    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    model = build_chat_model(
        configurable.research_model,
        configurable.research_model_max_tokens,
        get_api_key_for_model(configurable.research_model, config),
        tags=[TAG_NOSTREAM],
    )

    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=researcher_prompt,
        name="researcher",
    )

    # max_react_tool_calls=N means N tool-call rounds.
    # Each round = 1 model step + 1 tool step = 2 graph steps, plus 1 final model step.
    recursion_limit = configurable.max_react_tool_calls * 2 + 1

    result = await agent.ainvoke(
        {"messages": state["messages"]},
        {**config, "recursion_limit": recursion_limit},
    )

    return {"messages": result["messages"]}

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = build_chat_model(
        configurable.compression_model,
        configurable.compression_model_max_tokens,
        get_api_key_for_model(configurable.compression_model, config),
        tags=[TAG_NOSTREAM],
    )
    
    # Step 2: Prepare messages for compression
    messages = list(state.get("messages", []))

    # Add instruction to switch from research mode to compression mode
    messages.append(HumanMessage(content=compress_research_simple_human_message))

    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            compression_messages = [SystemMessage(content=compression_prompt)] + messages

            # Execute compression
            response = await synthesizer_model.ainvoke(compression_messages)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(messages, include_types=["tool", "ai"])
            ])

            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                **_trace_message_update(response),
            }

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                messages = remove_up_to_last_ai_message(messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# create_agent handles the ReAct loop; compress_research post-processes the findings.
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
)

researcher_builder.add_node("researcher", researcher_agent_node)
researcher_builder.add_node("compress_research", compress_research)

researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("researcher", "compress_research")
researcher_builder.add_edge("compress_research", END)

researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = _build_final_report_findings(notes, raw_notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await build_chat_model(
                configurable.final_report_model,
                configurable.final_report_model_max_tokens,
                get_api_key_for_model(configurable.final_report_model, config),
                tags=[TAG_NOSTREAM],
            ).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState,
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
