"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StreamWriter

from open_deep_research.configuration import Configuration
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
from open_deep_research.stream_protocol import NodeStreamEmitter
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


async def clarify_with_user(
    state: AgentState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    emitter = NodeStreamEmitter(writer, node="clarify_with_user", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        if not configurable.allow_clarification:
            return Command(goto="write_research_brief")

        messages = state["messages"]
        clarification_model = (
            build_chat_model(
                configurable.research_model,
                configurable.research_model_max_tokens,
                get_api_key_for_model(configurable.research_model, config),
                tags=["langsmith:nostream"],
            )
            .with_structured_output(ClarifyWithUser)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        )

        prompt_content = clarify_with_user_instructions.format(
            messages=get_buffer_string(messages),
            date=get_today_str()
        )
        emitter.status("llm_start")
        response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
        emitter.status("llm_end")

        if response.need_clarification:
            return Command(
                goto=END,
                update={"messages": [AIMessage(content=response.question)]}
            )

        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()


async def write_research_brief(
    state: AgentState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["research_supervisor"]]:
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
    emitter = NodeStreamEmitter(writer, node="write_research_brief", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        research_model = (
            build_chat_model(
                configurable.research_model,
                configurable.research_model_max_tokens,
                get_api_key_for_model(configurable.research_model, config),
                tags=["langsmith:nostream"],
            )
            .with_structured_output(ResearchQuestion)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        )

        prompt_content = transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        )
        emitter.status("llm_start")
        response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
        emitter.status("llm_end")

        supervisor_system_prompt = lead_researcher_prompt.format(
            date=get_today_str(),
            max_concurrent_research_units=configurable.max_concurrent_research_units,
            max_researcher_iterations=configurable.max_researcher_iterations
        )

        return Command(
            goto="research_supervisor",
            update={
                "research_brief": response.research_brief,
                "supervisor_messages": {
                    "type": "override",
                    "value": [
                        SystemMessage(content=supervisor_system_prompt),
                        HumanMessage(content=response.research_brief)
                    ]
                }
            }
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()


async def supervisor(
    state: SupervisorState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["supervisor_tools"]]:
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
    emitter = NodeStreamEmitter(writer, node="supervisor", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

        research_model = (
            build_chat_model(
                configurable.research_model,
                configurable.research_model_max_tokens,
                get_api_key_for_model(configurable.research_model, config),
                tags=["langsmith:nostream"],
            )
            .bind_tools(lead_researcher_tools)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        )

        supervisor_messages = state.get("supervisor_messages", [])
        emitter.status("llm_start")
        response = await research_model.ainvoke(supervisor_messages, config)
        emitter.status("llm_end")

        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

async def supervisor_tools(
    state: SupervisorState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["supervisor", "__end__"]]:
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
    emitter = NodeStreamEmitter(writer, node="supervisor_tools", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        supervisor_messages = state.get("supervisor_messages", [])
        research_iterations = state.get("research_iterations", 0)
        most_recent_message = supervisor_messages[-1]

        exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
        no_tool_calls = not most_recent_message.tool_calls
        research_complete_tool_call = any(
            tool_call["name"] == "ResearchComplete"
            for tool_call in most_recent_message.tool_calls
        )

        if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )

        all_tool_messages = []
        update_payload = {"supervisor_messages": []}

        think_tool_calls = [
            tool_call for tool_call in most_recent_message.tool_calls
            if tool_call["name"] == "think_tool"
        ]

        for tool_call in think_tool_calls:
            emitter.tool_input_start(tool_call["id"], tool_call["name"])
            emitter.tool_input_available(tool_call["id"], tool_call["name"], tool_call["args"])
            reflection_content = tool_call["args"]["reflection"]
            output = f"Reflection recorded: {reflection_content}"
            emitter.tool_output_available(tool_call["id"], output)
            all_tool_messages.append(ToolMessage(
                content=output,
                name="think_tool",
                tool_call_id=tool_call["id"]
            ))

        conduct_research_calls = [
            tool_call for tool_call in most_recent_message.tool_calls
            if tool_call["name"] == "ConductResearch"
        ]

        if conduct_research_calls:
            try:
                allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
                overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

                for tool_call in allowed_conduct_research_calls:
                    emitter.tool_input_start(tool_call["id"], tool_call["name"])
                    emitter.tool_input_available(tool_call["id"], tool_call["name"], tool_call["args"])

                research_tasks = [
                    researcher_subgraph.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    }, config)
                    for tool_call in allowed_conduct_research_calls
                ]

                tool_results = await asyncio.gather(*research_tasks)

                for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                    output = observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded")
                    emitter.tool_output_available(tool_call["id"], output)
                    all_tool_messages.append(ToolMessage(
                        content=output,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))

                for overflow_call in overflow_conduct_research_calls:
                    output = (
                        f"Error: Did not run this research as you have already exceeded the maximum number of concurrent "
                        f"research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units."
                    )
                    emitter.error("tool_error", output, tool_call_id=overflow_call["id"])
                    emitter.tool_output_available(overflow_call["id"], output)
                    all_tool_messages.append(ToolMessage(
                        content=output,
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"]
                    ))

                raw_notes_concat = "\n".join([
                    "\n".join(observation.get("raw_notes", []))
                    for observation in tool_results
                ])

                if raw_notes_concat:
                    update_payload["raw_notes"] = [raw_notes_concat]

            except Exception as e:
                emitter.error("tool_error", e)
                if is_token_limit_exceeded(e, configurable.research_model):
                    return Command(
                        goto=END,
                        update={
                            "notes": get_notes_from_tool_calls(supervisor_messages),
                            "research_brief": state.get("research_brief", "")
                        }
                    )
                raise

        update_payload["supervisor_messages"] = all_tool_messages
        return Command(
            goto="supervisor",
            update=update_payload
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

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

async def researcher(
    state: ResearcherState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    emitter = NodeStreamEmitter(writer, node="researcher", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        researcher_messages = state.get("researcher_messages", [])

        tools = await get_all_tools(config)
        if len(tools) == 0:
            raise ValueError(
                "No tools found to conduct research: Please configure either your "
                "search API or add MCP tools to your configuration."
            )

        researcher_prompt = research_system_prompt.format(
            mcp_prompt=configurable.mcp_prompt or "",
            date=get_today_str()
        )

        research_model = (
            build_chat_model(
                configurable.research_model,
                configurable.research_model_max_tokens,
                get_api_key_for_model(configurable.research_model, config),
                tags=["langsmith:nostream"],
            )
            .bind_tools(tools)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        )

        messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
        emitter.status("llm_start")
        response = await research_model.ainvoke(messages, config)
        emitter.status("llm_end")

        return Command(
            goto="researcher_tools",
            update={
                "researcher_messages": [response],
                "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
            }
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return {"ok": True, "output": await tool.ainvoke(args, config)}
    except Exception as e:
        return {"ok": False, "output": f"Error executing tool: {str(e)}", "error": str(e)}


async def researcher_tools(
    state: ResearcherState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (exa_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    emitter = NodeStreamEmitter(writer, node="researcher_tools", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        researcher_messages = state.get("researcher_messages", [])
        most_recent_message = researcher_messages[-1]

        has_tool_calls = bool(most_recent_message.tool_calls)
        if not has_tool_calls:
            return Command(goto="compress_research")

        tools = await get_all_tools(config)
        tools_by_name = {
            tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
            for tool in tools
        }

        tool_calls = most_recent_message.tool_calls
        for tool_call in tool_calls:
            emitter.tool_input_start(tool_call["id"], tool_call["name"])
            emitter.tool_input_available(tool_call["id"], tool_call["name"], tool_call["args"])

        tool_execution_tasks = [
            execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
            for tool_call in tool_calls
        ]
        observations = await asyncio.gather(*tool_execution_tasks)

        tool_outputs = []
        for observation, tool_call in zip(observations, tool_calls):
            if not observation["ok"]:
                emitter.error("tool_error", observation["error"], tool_call_id=tool_call["id"])
            emitter.tool_output_available(tool_call["id"], observation["output"])
            emitter.sources_from_output(observation["output"])
            tool_outputs.append(ToolMessage(
                content=observation["output"],
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))

        exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
        research_complete_called = any(
            tool_call["name"] == "ResearchComplete"
            for tool_call in most_recent_message.tool_calls
        )

        if exceeded_iterations or research_complete_called:
            return Command(
                goto="compress_research",
                update={"researcher_messages": tool_outputs}
            )

        return Command(
            goto="researcher",
            update={"researcher_messages": tool_outputs}
        )
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

async def compress_research(
    state: ResearcherState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
):
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
    emitter = NodeStreamEmitter(writer, node="compress_research", config=config)
    emitter.start()
    try:
        configurable = Configuration.from_runnable_config(config)
        synthesizer_model = build_chat_model(
            configurable.compression_model,
            configurable.compression_model_max_tokens,
            get_api_key_for_model(configurable.compression_model, config),
            tags=["langsmith:nostream"],
        )

        researcher_messages = state.get("researcher_messages", [])
        researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

        synthesis_attempts = 0
        max_attempts = 3

        while synthesis_attempts < max_attempts:
            try:
                compression_prompt = compress_research_system_prompt.format(date=get_today_str())
                messages = [SystemMessage(content=compression_prompt)] + researcher_messages

                emitter.status("llm_start")
                response = await synthesizer_model.ainvoke(messages, config)
                emitter.status("llm_end")

                raw_notes_content = "\n".join([
                    str(message.content)
                    for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
                ])

                return {
                    "compressed_research": str(response.content),
                    "raw_notes": [raw_notes_content]
                }

            except Exception as e:
                synthesis_attempts += 1
                emitter.error("llm_error", e)
                if is_token_limit_exceeded(e, configurable.research_model):
                    researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                    continue
                continue

        raw_notes_content = "\n".join([
            str(message.content)
            for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
        ])

        return {
            "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
            "raw_notes": [raw_notes_content]
        }
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState,
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(
    state: AgentState,
    config: RunnableConfig,
    *,
    writer: StreamWriter,
):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    emitter = NodeStreamEmitter(writer, node="final_report_generation", config=config)
    emitter.start()
    try:
        notes = state.get("notes", [])
        cleared_state = {"notes": {"type": "override", "value": []}}
        findings = "\n".join(notes)

        configurable = Configuration.from_runnable_config(config)
        max_retries = 3
        current_retry = 0
        findings_token_limit = None

        while current_retry <= max_retries:
            try:
                final_report_prompt = final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=get_buffer_string(state.get("messages", [])),
                    findings=findings,
                    date=get_today_str()
                )

                emitter.status("llm_start")
                final_report = await build_chat_model(
                    configurable.final_report_model,
                    configurable.final_report_model_max_tokens,
                    get_api_key_for_model(configurable.final_report_model, config),
                    tags=["langsmith:nostream"],
                ).ainvoke([
                    HumanMessage(content=final_report_prompt)
                ], config)
                emitter.status("llm_end")

                return {
                    "final_report": final_report.content,
                    "messages": [final_report],
                    **cleared_state
                }

            except Exception as e:
                emitter.error("llm_error", e)
                if is_token_limit_exceeded(e, configurable.final_report_model):
                    current_retry += 1

                    if current_retry == 1:
                        model_token_limit = get_model_token_limit(configurable.final_report_model)
                        if not model_token_limit:
                            return {
                                "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                                "messages": [AIMessage(content="Report generation failed due to token limits")],
                                **cleared_state
                            }
                        findings_token_limit = model_token_limit * 4
                    else:
                        findings_token_limit = int(findings_token_limit * 0.9)

                    findings = findings[:findings_token_limit]
                    continue

                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

        return {
            "final_report": "Error generating final report: Maximum retries exceeded",
            "messages": [AIMessage(content="Report generation failed after maximum retries")],
            **cleared_state
        }
    except Exception as exc:
        emitter.error("node_error", exc)
        raise
    finally:
        emitter.finish()

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
