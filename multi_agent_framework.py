"""
多智能体协作框架 - Multi-Agent Collaboration Framework

这是一个基于 LangChain 和 LangGraph 的可扩展多智能体框架，支持：
1. 动态添加/删除智能体
2. 多种协作模式（流水线、监督者、并行等）
3. 智能体注册与管理
4. 灵活的通信机制

使用示例:
    from multi_agent_framework import AgentFramework, BaseAgent
    
    # 创建框架
    framework = AgentFramework()
    
    # 注册智能体
    framework.register_agent(analyst_agent)
    framework.register_agent(writer_agent)
    
    # 执行任务
    result = framework.execute("分析并撰写关于AI的文章")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# =============================================================================
# 配置
# =============================================================================

API_KEY = os.environ.get("SILICONFLOW_API_KEY")
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")


def get_llm(temperature: float = 0.2, max_tokens: int = 2048) -> ChatOpenAI:
    """获取 LLM 实例"""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# =============================================================================
# 基础智能体类
# =============================================================================

class BaseAgent(ABC):
    """智能体基类 - 所有智能体都应继承此类"""
    
    def __init__(self, name: str, description: str, system_prompt: str, 
                 temperature: float = 0.2, tools: Optional[List[Any]] = None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.tools = tools or []
        self.llm = get_llm(temperature=temperature)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体任务 - 子类必须实现"""
        pass
    
    def __repr__(self):
        return f"BaseAgent(name='{self.name}', description='{self.description}')"


# =============================================================================
# 通用智能体实现
# =============================================================================

class LLMAgent(BaseAgent):
    """基于 LLM 的通用智能体"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行 LLM 调用"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=input_data.get("content", ""))
        ]
        response = self.llm.invoke(messages)
        return {"output": response.content, "agent_name": self.name}


class ReactAgent(BaseAgent):
    """ReAct 模式智能体（推理+行动）"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行 ReAct 循环"""
        if not self.tools:
            raise ValueError("ReactAgent 需要至少一个工具")
        
        agent = create_react_agent(self.llm, tools=self.tools)
        result = agent.invoke({
            "messages": [HumanMessage(content=input_data.get("content", ""))]
        })
        
        messages = result.get("messages", [])
        final_answer = getattr(messages[-1], "content", "") if messages else ""
        
        return {
            "output": final_answer,
            "agent_name": self.name,
            "full_trace": messages
        }


# =============================================================================
# 框架状态定义
# =============================================================================

class FrameworkState(TypedDict):
    """框架全局状态"""
    task: str
    current_step: str
    agent_outputs: Dict[str, Any]
    final_result: str
    metadata: Dict[str, Any]


# =============================================================================
# 智能体框架核心
# =============================================================================

class AgentFramework:
    """
    多智能体协作框架
    
    特性:
    - 动态注册/注销智能体
    - 多种协作模式
    - 灵活的工作流编排
    """
    
    def __init__(self, name: str = "MultiAgentFramework"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Callable] = {}
        self._compiled_graph = None
        
    def register_agent(self, agent: BaseAgent) -> None:
        """注册智能体到框架"""
        self.agents[agent.name] = agent
        print(f"✓ 已注册智能体: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> bool:
        """从框架中移除智能体"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"✓ 已移除智能体: {agent_name}")
            return True
        return False
    
    def list_agents(self) -> List[str]:
        """列出所有已注册的智能体"""
        return list(self.agents.keys())
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """获取指定智能体"""
        return self.agents.get(agent_name)
    
    # =========================================================================
    # 协作模式构建器
    # =========================================================================
    
    def build_pipeline_workflow(self, agent_sequence: List[str]) -> Callable[..., Any]:
        """
        构建流水线工作流
        
        Args:
            agent_sequence: 智能体执行顺序列表
            
        Returns:
            编译后的工作流函数
        """
        # 验证智能体是否存在
        for name in agent_sequence:
            if name not in self.agents:
                raise ValueError(f"智能体 '{name}' 未注册")
        
        def pipeline_executor(state: FrameworkState) -> FrameworkState:
            """流水线执行器"""
            current_input = state["task"]
            
            for agent_name in agent_sequence:
                agent = self.agents[agent_name]
                result = agent.execute({"content": current_input})
                
                # 保存每个智能体的输出
                state["agent_outputs"][agent_name] = result
                
                # 将输出作为下一个智能体的输入
                current_input = result.get("output", current_input)
            
            state["final_result"] = current_input
            return state
        
        return pipeline_executor
    
    def build_supervisor_workflow(self, supervisor_name: str, 
                                  worker_names: List[str],
                                  router_func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        """
        构建监督者工作流
        
        Args:
            supervisor_name: 监督者智能体名称
            worker_names: 工作者智能体列表
            router_func: 可选的路由函数，决定调用哪个工作者
            
        Returns:
            编译后的工作流函数
        """
        # 验证智能体是否存在
        if supervisor_name not in self.agents:
            raise ValueError(f"监督者智能体 '{supervisor_name}' 未注册")
        
        for name in worker_names:
            if name not in self.agents:
                raise ValueError(f"工作者智能体 '{name}' 未注册")
        
        def supervisor_executor(state: FrameworkState) -> FrameworkState:
            """监督者执行器"""
            supervisor = self.agents[supervisor_name]
            
            # 监督者决定路由
            route_result = supervisor.execute({"content": state["task"]})
            route_decision = route_result.get("output", "").strip().lower()
            
            # 使用自定义路由函数或默认匹配
            selected_worker = None
            if router_func:
                selected_worker = router_func(route_decision, worker_names)
            else:
                # 默认：尝试匹配工作者名称
                for worker_name in worker_names:
                    if worker_name.lower() in route_decision:
                        selected_worker = worker_name
                        break
            
            # 如果没有匹配，使用第一个工作者
            if not selected_worker and worker_names:
                selected_worker = worker_names[0]
            
            # 执行选中的工作者
            if selected_worker:
                worker = self.agents[selected_worker]
                worker_result = worker.execute({"content": state["task"]})
                state["agent_outputs"][selected_worker] = worker_result
                state["final_result"] = worker_result.get("output", "")
            
            state["metadata"]["route"] = selected_worker
            return state
        
        return supervisor_executor
    
    def build_parallel_workflow(self, agent_names: List[str], 
                                aggregator: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        """
        构建并行工作流
        
        Args:
            agent_names: 并行执行的智能体列表
            aggregator: 可选的结果聚合函数
            
        Returns:
            编译后的工作流函数
        """
        # 验证智能体是否存在
        for name in agent_names:
            if name not in self.agents:
                raise ValueError(f"智能体 '{name}' 未注册")
        
        def parallel_executor(state: FrameworkState) -> FrameworkState:
            """并行执行器"""
            results = {}
            
            # 并行执行所有智能体
            for agent_name in agent_names:
                agent = self.agents[agent_name]
                result = agent.execute({"content": state["task"]})
                results[agent_name] = result.get("output", "")
                state["agent_outputs"][agent_name] = result
            
            # 聚合结果
            if aggregator:
                state["final_result"] = aggregator(results)
            else:
                # 默认聚合：简单拼接
                aggregated = "\n\n".join([
                    f"### {name} 的输出:\n{output}" 
                    for name, output in results.items()
                ])
                state["final_result"] = aggregated
            
            return state
        
        return parallel_executor
    
    # =========================================================================
    # 执行接口
    # =========================================================================
    
    def execute(self, task: str, workflow_type: str = "pipeline", 
                workflow_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
            workflow_type: 工作流类型 (pipeline/supervisor/parallel)
            workflow_config: 工作流配置
            
        Returns:
            执行结果
        """
        if not self.agents:
            raise ValueError("没有注册的智能体")
        
        workflow_config = workflow_config or {}
        
        # 初始化状态
        initial_state: FrameworkState = {
            "task": task,
            "current_step": "",
            "agent_outputs": {},
            "final_result": "",
            "metadata": {}
        }
        
        # 根据配置选择工作流
        if workflow_type == "pipeline":
            agent_sequence = workflow_config.get("agent_sequence", list(self.agents.keys()))
            executor = self.build_pipeline_workflow(agent_sequence)
        
        elif workflow_type == "supervisor":
            supervisor_name = workflow_config.get("supervisor_name", "")
            worker_names = workflow_config.get("worker_names", [])
            router_func = workflow_config.get("router_func")
            executor = self.build_supervisor_workflow(supervisor_name, worker_names, router_func)
        
        elif workflow_type == "parallel":
            agent_names = workflow_config.get("agent_names", list(self.agents.keys()))
            aggregator = workflow_config.get("aggregator")
            executor = self.build_parallel_workflow(agent_names, aggregator)
        
        else:
            raise ValueError(f"未知的工作流类型: {workflow_type}")
        
        # 执行工作流
        result = executor(initial_state)
        
        return {
            "task": task,
            "workflow_type": workflow_type,
            "result": result["final_result"],
            "agent_outputs": result["agent_outputs"],
            "metadata": result["metadata"]
        }
    
    def __repr__(self):
        return f"AgentFramework(name='{self.name}', agents={list(self.agents.keys())})"


# =============================================================================
# 预定义智能体模板
# =============================================================================

def create_analyst_agent(name: str = "analyst") -> BaseAgent:
    """创建分析型智能体"""
    return LLMAgent(
        name=name,
        description="分析问题并提供结构化要点",
        system_prompt="你是一个专业的分析师。请分析用户提供的主题，输出3-6个关键要点，每行一个，简洁明了。",
        temperature=0.3
    )


def create_writer_agent(name: str = "writer") -> BaseAgent:
    """创建写作型智能体"""
    return LLMAgent(
        name=name,
        description="将要点整理成流畅的文章",
        system_prompt="你是一个优秀的作家。请将提供的要点整合成一篇连贯、流畅的短文，用中文表达。",
        temperature=0.5
    )


def create_researcher_agent(name: str = "researcher") -> BaseAgent:
    """创建研究型智能体"""
    return LLMAgent(
        name=name,
        description="进行深入研究并提供详细信息",
        system_prompt="你是一个研究专家。请对用户的问题进行深入分析，提供详细、准确的信息和见解。",
        temperature=0.2
    )


def create_critic_agent(name: str = "critic") -> BaseAgent:
    """创建评审型智能体"""
    return LLMAgent(
        name=name,
        description="评审内容并提出改进建议",
        system_prompt="你是一个严格的审稿人。请检查提供的内容，指出逻辑漏洞、事实错误或改进建议。",
        temperature=0.2
    )


def create_summarizer_agent(name: str = "summarizer") -> BaseAgent:
    """创建总结型智能体"""
    return LLMAgent(
        name=name,
        description="总结长篇内容为简洁要点",
        system_prompt="你是一个总结专家。请将提供的内容总结为2-3句简洁的中文摘要。",
        temperature=0.1
    )


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  多智能体协作框架演示")
    print("=" * 60)
    
    # 创建框架
    framework = AgentFramework(name="DemoFramework")
    
    # 注册智能体
    analyst = create_analyst_agent()
    writer = create_writer_agent()
    researcher = create_researcher_agent()
    summarizer = create_summarizer_agent()
    
    framework.register_agent(analyst)
    framework.register_agent(writer)
    framework.register_agent(researcher)
    framework.register_agent(summarizer)
    
    print(f"\n已注册智能体: {framework.list_agents()}")
    
    # 示例 1: 流水线模式
    print("\n" + "=" * 60)
    print("  示例 1: 流水线模式 (Analyst -> Writer)")
    print("=" * 60)
    
    result = framework.execute(
        task="多智能体系统在客服场景中的应用价值",
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["analyst", "writer"]
        }
    )
    
    print(f"任务: {result['task']}")
    print(f"结果:\n{result['result']}")
    
    # 示例 2: 并行模式
    print("\n" + "=" * 60)
    print("  示例 2: 并行模式 (Researcher + Analyst)")
    print("=" * 60)
    
    result = framework.execute(
        task="什么是强化学习？",
        workflow_type="parallel",
        workflow_config={
            "agent_names": ["researcher", "analyst"]
        }
    )
    
    print(f"任务: {result['task']}")
    print(f"结果:\n{result['result'][:1000]}")
    
    # 示例 3: 动态添加/删除智能体
    print("\n" + "=" * 60)
    print("  示例 3: 动态管理智能体")
    print("=" * 60)
    
    critic = create_critic_agent()
    framework.register_agent(critic)
    print(f"添加后智能体: {framework.list_agents()}")
    
    framework.unregister_agent("critic")
    print(f"移除后智能体: {framework.list_agents()}")
    
    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)
