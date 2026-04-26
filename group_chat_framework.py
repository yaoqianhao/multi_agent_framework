"""
多智能体沟通与协作架构 (Group Chat / Discussion Framework)

该架构允许多个智能体在一个共享的“聊天室”中自由交流、辩论或协作。
通过一个“主持人(Manager)”智能体来动态决定下一个发言者，实现真正的沟通和互动。
"""

import os
from typing import Dict, List, Any, TypedDict, Callable
from langgraph.graph import StateGraph, START, END

# 导入基础框架的 Agent 类
from multi_agent_framework import BaseAgent, LLMAgent, get_llm

class GroupChatState(TypedDict):
    """群聊状态定义"""
    topic: str                 # 讨论的主题
    messages: List[Dict]       # 聊天记录 [{"sender": "A", "content": "..."}]
    next_speaker: str          # 下一个发言者的名字
    turn_count: int            # 当前轮数
    max_turns: int             # 最大轮数


class GroupChatFramework:
    """
    群聊架构核心类
    允许多个 Agent 共享上下文，由一个 Manager 决定话语权流转。
    """
    def __init__(self, name: str = "GroupChat"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.manager: BaseAgent = None
        self.graph_builder = StateGraph(GroupChatState)
        self.compiled_graph = None

    def register_agent(self, agent: BaseAgent):
        """注册参与讨论的普通智能体"""
        self.agents[agent.name] = agent
        print(f"✓ 已注册参与者: {agent.name}")

    def set_manager(self, manager: BaseAgent):
        """设置主持人智能体"""
        self.manager = manager
        print(f"✓ 已设置主持人: {manager.name}")

    def _format_history(self, messages: List[Dict]) -> str:
        """格式化聊天记录为纯文本，供大模型阅读"""
        if not messages:
            return "暂无聊天记录。"
        return "\n".join([f"[{m['sender']}]: {m['content']}" for m in messages])

    def compile(self):
        """编译沟通状态图"""
        if not self.manager:
            raise ValueError("必须设置一个 Manager 来管理沟通。")
        if not self.agents:
            raise ValueError("至少需要注册一个 Agent 参与沟通。")

        # 1. 主持人节点：审阅历史，决定下一个谁发言
        def manager_node(state: GroupChatState) -> GroupChatState:
            history_str = self._format_history(state["messages"])
            agent_names = list(self.agents.keys())
            prompt = f"""
任务主题: {state['topic']}
参与者: {', '.join(agent_names)}

当前讨论记录:
{history_str}

请作为主持人，根据当前的讨论记录，决定下一个该谁发言。
规则:
1. 如果讨论已经得出最终结论、圆满解决，或者偏离主题需要结束，请回复 "END"。
2. 否则，请仅仅回复下一个应该发言的参与者名字（必须精确匹配 {', '.join(agent_names)} 中的一个，不要输出任何其他标点或字符）。
"""
            result = self.manager.execute({"content": prompt})
            next_speaker = result.get("output", "").strip()

            # 清理和匹配可能的回答
            matched_speaker = "END"
            for name in agent_names:
                if name.lower() in next_speaker.lower():
                    matched_speaker = name
                    break
            
            if matched_speaker == "END" and "END" not in next_speaker:
                # Fallback: 如果大模型乱回答，轮询兜底
                matched_speaker = agent_names[state["turn_count"] % len(agent_names)]

            state["next_speaker"] = matched_speaker
            state["turn_count"] += 1
            return state

        # 2. 构建所有参与者的发言节点
        def create_agent_node(agent_name: str):
            def node(state: GroupChatState) -> GroupChatState:
                agent = self.agents[agent_name]
                history_str = self._format_history(state["messages"])
                prompt = f"""
你正在参与一个群组讨论。
主题: {state['topic']}

之前的讨论记录:
{history_str}

现在轮到你 ({agent_name}) 发言了。
请结合你的专业角色，推进讨论、提出问题或给出建议。直接输出你想说的话即可，不要带上你的名字前缀。尽量简明扼要。
"""
                result = agent.execute({"content": prompt})
                output = result.get("output", "")
                
                state["messages"].append({
                    "sender": agent_name,
                    "content": output
                })
                return state
            return node

        # 3. 组装状态图
        self.graph_builder.add_node("manager", manager_node)
        
        for name in self.agents:
            self.graph_builder.add_node(name, create_agent_node(name))

        self.graph_builder.add_edge(START, "manager")
        
        # 4. 定义路由规则（从 manager 指向对应的发言者，或者结束）
        def route_from_manager(state: GroupChatState) -> str:
            if state["turn_count"] > state["max_turns"]:
                return END
            if state["next_speaker"] == "END":
                return END
            return state["next_speaker"]

        self.graph_builder.add_conditional_edges("manager", route_from_manager)

        # 5. 每个参与者发言后，把话语权交回给主持人
        for name in self.agents:
            self.graph_builder.add_edge(name, "manager")

        self.compiled_graph = self.graph_builder.compile()
        print("✓ 群聊沟通架构已编译完成！")

    def execute(self, topic: str, max_turns: int = 8) -> Dict:
        """执行群聊交流"""
        if not self.compiled_graph:
            self.compile()
            
        initial_state: GroupChatState = {
            "topic": topic,
            "messages": [],
            "next_speaker": "",
            "turn_count": 0,
            "max_turns": max_turns
        }
        
        print(f"\n[开始沟通] 主题: {topic}")
        print("=" * 60)
        
        final_state = initial_state
        # 使用流式执行，实时打印过程
        for event in self.compiled_graph.stream(initial_state):
            for node_name, state in event.items():
                if node_name in self.agents:
                    last_msg = state["messages"][-1]
                    print(f"\n🤖 [{last_msg['sender']}]:\n{last_msg['content']}")
                    print("-" * 60)
                elif node_name == "manager":
                    if state["next_speaker"] == "END":
                        print(f"\n👔 [主持人]: 讨论已充分，结束本次会议。")
                    else:
                        print(f"\n👔 [主持人]: 下面请 {state['next_speaker']} 发言...")
                
                final_state = state

        print("=" * 60)
        print(f"[沟通结束] 会议共进行了 {final_state['turn_count']} 轮（包含主持人判定）。")
        return final_state

# =============================================================================
# 示例用法
# =============================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # 1. 创建聊天架构
    chat_framework = GroupChatFramework()
    
    # 2. 定义主持人 (Manager)
    manager = LLMAgent(
        name="Host",
        description="会议主持人",
        system_prompt="你是经验丰富的主持人，善于引导讨论走向结论，不废话。"
    )
    chat_framework.set_manager(manager)
    
    # 3. 定义参与讨论的各个角色
    developer = LLMAgent(
        name="Developer",
        description="前端开发工程师",
        system_prompt="你是一个资深的前端开发，倾向于用 React 和现代化的技术栈解决问题，注重代码可维护性。发言尽量不超过100字。"
    )
    
    designer = LLMAgent(
        name="Designer",
        description="UI/UX 设计师",
        system_prompt="你是一个追求完美的设计师，注重用户体验、色彩搭配和动画流畅度。发言尽量不超过100字。"
    )
    
    product_manager = LLMAgent(
        name="PM",
        description="产品经理",
        system_prompt="你是一个看重按时交付和商业价值的产品经理。你喜欢化繁为简，督促大家快速达成共识。发言尽量不超过100字。"
    )
    
    # 4. 注册参与者
    chat_framework.register_agent(developer)
    chat_framework.register_agent(designer)
    chat_framework.register_agent(product_manager)
    
    # 5. 开始沟通！
    chat_framework.execute(topic="我们新电商App的首屏应该是注重酷炫的动画展示，还是注重商品加载速度和功能优先？", max_turns=6)
