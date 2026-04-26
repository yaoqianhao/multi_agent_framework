"""
高级多智能体框架 - 基于 LangGraph StateGraph

这个版本使用 LangGraph 的状态图来管理智能体间的复杂协作流程，
支持条件路由、循环、并行等高级模式。

特性:
1. 基于状态图的可视化工作流
2. 条件分支和动态路由
3. 循环迭代（如反思模式）
4. 嵌套子图（支持层级化智能体组织）
5. 持久化和恢复能力
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 导入基础框架
from multi_agent_framework import (
    BaseAgent, 
    LLMAgent, 
    get_llm,
    create_analyst_agent,
    create_writer_agent,
    create_researcher_agent,
    create_critic_agent,
    create_summarizer_agent,
)

# =============================================================================
# 配置
# =============================================================================

API_KEY = os.environ.get("SILICONFLOW_API_KEY")
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")


# =============================================================================
# 增强的状态定义
# =============================================================================

class AdvancedFrameworkState(TypedDict):
    """高级框架状态"""
    task: str                    # 原始任务
    current_data: str           # 当前处理的数据
    history: List[Dict]         # 执行历史
    iteration_count: int        # 迭代次数
    max_iterations: int         # 最大迭代次数
    agent_results: Dict[str, Any]  # 各智能体结果
    final_output: str           # 最终输出
    metadata: Dict[str, Any]    # 元数据
    should_continue: bool       # 是否继续执行


# =============================================================================
# 高级智能体框架
# =============================================================================

class AdvancedAgentFramework:
    """
    基于 LangGraph 的高级多智能体框架
    
    支持:
    - 动态构建状态图
    - 条件路由
    - 循环迭代
    - 记忆保存
    - 子图嵌套
    """
    
    def __init__(self, name: str = "AdvancedFramework", 
                 use_memory: bool = True):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_nodes: Dict[str, Callable] = {}
        self.graph_builder = StateGraph(AdvancedFrameworkState)
        self.compiled_graph = None
        self.use_memory = use_memory
        self.memory = MemorySaver() if use_memory else None
        
    def register_agent(self, agent: BaseAgent) -> None:
        """注册智能体"""
        self.agents[agent.name] = agent
        print(f"✓ 已注册智能体: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> bool:
        """注销智能体"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"✓ 已移除智能体: {agent_name}")
            return True
        return False
    
    def list_agents(self) -> List[str]:
        """列出所有智能体"""
        return list(self.agents.keys())
    
    # =========================================================================
    # 节点构建器
    # =========================================================================
    
    def create_agent_node(self, agent_name: str) -> Callable:
        """为指定智能体创建图节点"""
        if agent_name not in self.agents:
            raise ValueError(f"智能体 '{agent_name}' 未注册")
        
        def node_func(state: AdvancedFrameworkState) -> AdvancedFrameworkState:
            agent = self.agents[agent_name]
            result = agent.execute({"content": state["current_data"]})
            
            # 更新状态
            output = result.get("output", "")
            state["agent_results"][agent_name] = result
            state["current_data"] = output
            state["history"].append({
                "agent": agent_name,
                "input": state["current_data"],
                "output": output
            })
            
            return state
        
        return node_func
    
    def create_router_node(self, router_func: Callable, 
                          routes: Dict[str, str]) -> Callable:
        """
        创建路由节点
        
        Args:
            router_func: 路由决策函数
            routes: 路由映射 {决策值: 目标节点}
        """
        def router_node(state: AdvancedFrameworkState) -> str:
            decision = router_func(state)
            return routes.get(decision, END)
        
        return router_node
    
    def create_condition_edge(self, condition_func: Callable, 
                             true_target: str, false_target: str) -> Callable:
        """创建条件边"""
        def conditional_edge(state: AdvancedFrameworkState) -> str:
            if condition_func(state):
                return true_target
            return false_target
        
        return conditional_edge
    
    # =========================================================================
    # 工作流模板
    # =========================================================================
    
    def build_pipeline_graph(self, agent_sequence: List[str]) -> None:
        """
        构建流水线图
        
        Args:
            agent_sequence: 智能体执行顺序
        """
        # 验证智能体
        for name in agent_sequence:
            if name not in self.agents:
                raise ValueError(f"智能体 '{name}' 未注册")
        
        # 添加节点
        for agent_name in agent_sequence:
            node_func = self.create_agent_node(agent_name)
            self.graph_builder.add_node(agent_name, node_func)
        
        # 添加边
        self.graph_builder.add_edge(START, agent_sequence[0])
        for i in range(len(agent_sequence) - 1):
            self.graph_builder.add_edge(agent_sequence[i], agent_sequence[i+1])
        self.graph_builder.add_edge(agent_sequence[-1], END)
    
    def build_reflection_graph(self, generator_name: str, critic_name: str,
                              max_iterations: int = 3) -> None:
        """
        构建反思循环图
        
        流程: generate -> reflect -> (如果不满意) -> revise -> generate -> ...
        
        Args:
            generator_name: 生成者智能体
            critic_name: 批评者智能体
            max_iterations: 最大迭代次数
        """
        if generator_name not in self.agents or critic_name not in self.agents:
            raise ValueError("生成者和批评者智能体必须已注册")
        
        # 生成节点
        def generate_node(state: AdvancedFrameworkState) -> AdvancedFrameworkState:
            agent = self.agents[generator_name]
            
            # 如果有批评意见，进行修订
            if state["iteration_count"] > 0 and "critique" in state["metadata"]:
                critique = state["metadata"]["critique"]
                revised_input = f"{state['task']}\n\n上一版本:\n{state['current_data']}\n\n改进建议:\n{critique}"
                result = agent.execute({"content": revised_input})
            else:
                result = agent.execute({"content": state["task"]})
            
            output = result.get("output", "")
            state["current_data"] = output
            state["agent_results"][generator_name] = result
            state["iteration_count"] += 1
            
            return state
        
        # 批评节点
        def critique_node(state: AdvancedFrameworkState) -> AdvancedFrameworkState:
            agent = self.agents[critic_name]
            critique_input = f"任务: {state['task']}\n\n当前版本:\n{state['current_data']}"
            result = agent.execute({"content": critique_input})
            
            critique = result.get("output", "")
            state["metadata"]["critique"] = critique
            state["agent_results"][critic_name] = result
            
            # 判断是否满意（简单启发式）
            satisfied_keywords = ["满意", "通过", "无需修改", "satisfied", "good"]
            state["should_continue"] = not any(
                kw in critique.lower() for kw in satisfied_keywords
            ) and state["iteration_count"] < max_iterations
            
            return state
        
        # 添加节点
        self.graph_builder.add_node("generate", generate_node)
        self.graph_builder.add_node("critique", critique_node)
        
        # 添加边
        self.graph_builder.add_edge(START, "generate")
        self.graph_builder.add_edge("generate", "critique")
        
        # 条件边：根据满意度决定是否继续
        def should_continue(state: AdvancedFrameworkState) -> str:
            return "generate" if state["should_continue"] else END
        
        self.graph_builder.add_conditional_edges("critique", should_continue)
    
    def build_branching_graph(self, branch_agents: Dict[str, List[str]],
                             merger_name: Optional[str] = None) -> None:
        """
        构建分支图
        
        Args:
            branch_agents: 分支配置 {分支名: [智能体列表]}
            merger_name: 可选的合并智能体
        """
        # 验证所有智能体
        all_agents = []
        for agents_in_branch in branch_agents.values():
            all_agents.extend(agents_in_branch)
        if merger_name:
            all_agents.append(merger_name)
        
        for name in all_agents:
            if name not in self.agents:
                raise ValueError(f"智能体 '{name}' 未注册")
        
        # 路由决策节点
        def router_decision(state: AdvancedFrameworkState) -> str:
            # 简单示例：默认选择第一个分支
            # 实际应用中可以根据任务内容动态决策
            return list(branch_agents.keys())[0]
        
        # 添加分支节点
        branch_start_nodes = []
        for branch_name, agent_list in branch_agents.items():
            # 每个分支的第一个节点
            first_agent = agent_list[0]
            node_func = self.create_agent_node(first_agent)
            node_name = f"{branch_name}_{first_agent}"
            self.graph_builder.add_node(node_name, node_func)
            branch_start_nodes.append(node_name)
            
            # 分支内的流水线
            for i in range(len(agent_list) - 1):
                curr_node = f"{branch_name}_{agent_list[i]}"
                next_node = f"{branch_name}_{agent_list[i+1]}"
                
                if i == 0 and curr_node != node_name:
                    self.graph_builder.add_node(curr_node, 
                                              self.create_agent_node(agent_list[i]))
                
                self.graph_builder.add_node(next_node, 
                                          self.create_agent_node(agent_list[i+1]))
                self.graph_builder.add_edge(curr_node, next_node)
        
        # 添加合并节点
        if merger_name:
            merger_node = self.create_agent_node(merger_name)
            self.graph_builder.add_node(merger_name, merger_node)
            
            # 所有分支最后节点连接到合并节点
            for branch_name, agent_list in branch_agents.items():
                last_node = f"{branch_name}_{agent_list[-1]}"
                self.graph_builder.add_edge(last_node, merger_name)
            
            self.graph_builder.add_edge(merger_name, END)
        else:
            # 没有合并节点，直接结束
            for branch_name, agent_list in branch_agents.items():
                last_node = f"{branch_name}_{agent_list[-1]}"
                self.graph_builder.add_edge(last_node, END)
        
        # 起始路由
        self.graph_builder.add_edge(START, branch_start_nodes[0])
    
    # =========================================================================
    # 编译和执行
    # =========================================================================
    
    def compile(self) -> None:
        """编译图"""
        if self.use_memory:
            self.compiled_graph = self.graph_builder.compile(checkpointer=self.memory)
        else:
            self.compiled_graph = self.graph_builder.compile()
        print("✓ 图已编译")
    
    def execute(self, task: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
            thread_id: 可选的线程ID（用于记忆）
            
        Returns:
            执行结果
        """
        if self.compiled_graph is None:
            raise ValueError("图未编译，请先调用 compile()")
        
        # 初始化状态
        initial_state: AdvancedFrameworkState = {
            "task": task,
            "current_data": task,
            "history": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "agent_results": {},
            "final_output": "",
            "metadata": {},
            "should_continue": True
        }
        
        # 配置
        config = {"configurable": {"thread_id": thread_id or "default"}} if self.use_memory else {}
        
        # 执行
        result = self.compiled_graph.invoke(initial_state, config=config)
        
        return {
            "task": task,
            "output": result["current_data"],
            "history": result["history"],
            "agent_results": result["agent_results"],
            "iterations": result["iteration_count"]
        }
    
    def visualize(self) -> None:
        """可视化图结构（需要安装 graphviz）"""
        if self.compiled_graph:
            try:
                from IPython.display import Image, display
                import io
                
                # 尝试生成 PNG
                png_data = self.compiled_graph.get_graph().draw_mermaid_png()
                display(Image(data=png_data))
            except Exception as e:
                print(f"可视化失败: {e}")
                print("提示: 安装 graphviz 和 mermaid 以支持可视化")
    
    def __repr__(self):
        return f"AdvancedAgentFramework(name='{self.name}', agents={list(self.agents.keys())})"


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  高级多智能体框架演示 (LangGraph StateGraph)")
    print("=" * 60)
    
    # 示例 1: 流水线模式
    print("\n" + "=" * 60)
    print("  示例 1: 流水线模式")
    print("=" * 60)
    
    framework1 = AdvancedAgentFramework(name="PipelineFramework")
    framework1.register_agent(create_analyst_agent())
    framework1.register_agent(create_writer_agent())
    
    framework1.build_pipeline_graph(["analyst", "writer"])
    framework1.compile()
    
    result = framework1.execute("人工智能在医疗领域的应用")
    print(f"\n任务: {result['task']}")
    print(f"输出:\n{result['output']}")
    print(f"执行历史: {len(result['history'])} 步")
    
    # 示例 2: 反思模式
    print("\n" + "=" * 60)
    print("  示例 2: 反思模式 (Generate -> Critique -> Revise)")
    print("=" * 60)
    
    framework2 = AdvancedAgentFramework(name="ReflectionFramework")
    framework2.register_agent(create_writer_agent("generator"))
    framework2.register_agent(create_critic_agent("critic"))
    
    framework2.build_reflection_graph("generator", "critic", max_iterations=2)
    framework2.compile()
    
    result = framework2.execute("用三句话解释量子计算")
    print(f"\n任务: {result['task']}")
    print(f"输出:\n{result['output']}")
    print(f"迭代次数: {result['iterations']}")
    
    # 示例 3: 动态添加智能体
    print("\n" + "=" * 60)
    print("  示例 3: 动态扩展框架")
    print("=" * 60)
    
    framework3 = AdvancedAgentFramework(name="ExtensibleFramework")
    framework3.register_agent(create_researcher_agent())
    
    # 动态添加更多智能体
    framework3.register_agent(create_analyst_agent())
    framework3.register_agent(create_summarizer_agent())
    
    print(f"\n当前智能体: {framework3.list_agents()}")
    
    # 构建新的流水线
    framework3.build_pipeline_graph(["researcher", "analyst", "summarizer"])
    framework3.compile()
    
    result = framework3.execute("区块链技术的基本原理和应用场景")
    print(f"\n任务: {result['task']}")
    print(f"输出:\n{result['output'][:800]}...")
    
    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)
