"""
互联网搜索智能体 - Web Search Agent

这是一个可以集成到多智能体框架中的智能体，能够利用互联网进行实时查询。
支持网页搜索、新闻查询和事实核查功能。

依赖安装:
    pip install duckduckgo-search langchain-community

使用示例:
    from web_search_agent import create_web_search_agent
    from multi_agent_framework import AgentFramework
    
    framework = AgentFramework()
    searcher = create_web_search_agent()
    framework.register_agent(searcher)
    
    result = framework.execute(
        task="2024年人工智能的最新发展",
        workflow_type="pipeline",
        workflow_config={"agent_sequence": ["web_searcher"]}
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from multi_agent_framework import ReactAgent, BaseAgent, get_llm


# =============================================================================
# 搜索工具定义
# =============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    在互联网上搜索信息
    
    Args:
        query: 搜索查询词
        num_results: 返回结果数量（默认5个）
        
    Returns:
        搜索结果摘要
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        
        # 创建搜索工具
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=num_results)
        search = DuckDuckGoSearchRun(api_wrapper=wrapper)
        
        # 执行搜索
        results = search.run(query)
        return results
    
    except ImportError:
        return "错误: 未安装 duckduckgo-search 包。请运行: pip install duckduckgo-search"
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def news_search(topic: str, max_results: int = 3) -> str:
    """
    搜索最新新闻
    
    Args:
        topic: 新闻主题
        max_results: 最大结果数
        
    Returns:
        相关新闻摘要
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchNewsRun
        
        news_search_tool = DuckDuckGoSearchNewsRun()
        results = news_search_tool.run(topic)
        return results
    
    except ImportError:
        return "错误: 未安装 duckduckgo-search 包。请运行: pip install duckduckgo-search"
    except Exception as e:
        return f"新闻搜索失败: {str(e)}"


@tool
def fact_check(claim: str) -> str:
    """
    验证某个声明或事实的真实性
    
    Args:
        claim: 需要验证的声明
        
    Returns:
        验证结果和相关信息
    """
    try:
        # 使用搜索来验证实时信息
        search_query = f"fact check {claim}"
        results = web_search.invoke({"query": search_query, "num_results": 3})
        return results
    
    except Exception as e:
        return f"事实验证失败: {str(e)}"


# =============================================================================
# 互联网搜索智能体类
# =============================================================================

class WebSearchAgent(ReactAgent):
    """
    互联网搜索智能体
    
    能够使用搜索工具获取实时信息，适合：
    - 研究最新趋势
    - 获取实时数据
    - 事实核查
    - 市场调研
    - 新闻追踪
    
    特性:
        - 基于 ReAct 模式（推理+行动）
        - 支持多种搜索工具
        - 可配置的功能模块
        - 与框架无缝集成
    """
    
    def __init__(self, 
                 name: str = "web_searcher", 
                 description: str = "从互联网搜索最新信息",
                 temperature: float = 0.2,
                 enable_news: bool = True,
                 enable_fact_check: bool = True,
                 custom_prompt: Optional[str] = None):
        """
        初始化互联网搜索智能体
        
        Args:
            name: 智能体名称
            description: 智能体描述
            temperature: LLM温度参数（0-1），控制创造性
            enable_news: 是否启用新闻搜索功能
            enable_fact_check: 是否启用事实核查功能
            custom_prompt: 自定义系统提示词（可选）
        """
        
        # 构建工具列表
        tools = [web_search]
        if enable_news:
            tools.append(news_search)
        if enable_fact_check:
            tools.append(fact_check)
        
        # 默认系统提示
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = """你是一个专业的互联网研究员和信息搜集专家。

你的能力：
1. 使用 web_search 工具进行通用网页搜索
2. 使用 news_search 工具获取最新新闻（如果启用）
3. 使用 fact_check 工具验证事实真伪（如果启用）

工作原则：
- 对于时效性强的问题，优先使用搜索工具获取最新信息
- 提供准确、可靠的信息来源
- 如果搜索结果不充分，尝试不同的搜索关键词
- 总结搜索结果时保持客观中立
- 引用重要信息的来源

当用户询问需要实时信息的问题时，主动使用搜索工具而不是依赖训练数据。

回答格式：
1. 先说明使用了哪个搜索工具
2. 展示搜索的关键结果
3. 提供结构化的总结
4. 标注信息来源（如果有）"""
        
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            tools=tools
        )
        
        # 保存配置
        self.enable_news = enable_news
        self.enable_fact_check = enable_fact_check


# =============================================================================
# 专用搜索智能体工厂函数
# =============================================================================

def create_web_search_agent(name: str = "web_searcher") -> BaseAgent:
    """
    创建通用的互联网搜索智能体
    
    这是最常用的搜索智能体，具有全部功能。
    
    Args:
        name: 智能体名称（默认: "web_searcher"）
        
    Returns:
        配置好的搜索智能体实例
        
    Example:
        >>> searcher = create_web_search_agent()
        >>> framework.register_agent(searcher)
    """
    return WebSearchAgent(
        name=name,
        description="从互联网搜索和收集最新信息",
        temperature=0.2,
        enable_news=True,
        enable_fact_check=True
    )


def create_news_analyst_agent(name: str = "news_analyst") -> BaseAgent:
    """
    创建新闻分析智能体（专注于新闻搜索）
    
    适合追踪最新动态、监控行业新闻等场景。
    
    Args:
        name: 智能体名称（默认: "news_analyst"）
        
    Returns:
        配置好的新闻分析智能体实例
        
    Example:
        >>> news_agent = create_news_analyst_agent()
        >>> result = news_agent.execute({"content": "今天的科技新闻"})
    """
    return WebSearchAgent(
        name=name,
        description="分析和总结最新新闻动态",
        temperature=0.3,
        enable_news=True,
        enable_fact_check=False,
        custom_prompt="""你是专业的新闻分析师。

职责：
- 使用 news_search 获取最新相关新闻
- 分析新闻的重要性和影响
- 提供简洁的新闻摘要
- 识别关键趋势和模式

输出要求：
- 按重要性排序新闻
- 突出关键信息
- 提供背景上下文"""
    )


def create_fact_checker_agent(name: str = "fact_checker") -> BaseAgent:
    """
    创建事实核查智能体
    
    专门用于验证信息的真实性和准确性。
    
    Args:
        name: 智能体名称（默认: "fact_checker"）
        
    Returns:
        配置好的事实核查智能体实例
        
    Example:
        >>> checker = create_fact_checker_agent()
        >>> result = checker.execute({"content": "验证：维生素C可以预防感冒"})
    """
    return WebSearchAgent(
        name=name,
        description="验证信息的真实性和准确性",
        temperature=0.1,
        enable_news=False,
        enable_fact_check=True,
        custom_prompt="""你是专业的事实核查员。

工作流程：
1. 使用 fact_check 工具验证声明
2. 使用 web_search 查找相关证据
3. 交叉验证多个来源
4. 给出可信度评估

评估标准：
- 已证实 (Verified)
- 部分正确 (Partially True)
- 误导性的 (Misleading)
- 虚假 (False)
- 无法验证 (Unverifiable)

输出格式：
- 原始声明
- 验证结果
- 证据来源
- 可信度评级
- 详细说明"""
    )


def create_market_research_agent(name: str = "market_researcher") -> BaseAgent:
    """
    创建市场研究智能体
    
    专注于市场趋势、竞争分析和商业情报。
    
    Args:
        name: 智能体名称（默认: "market_researcher"）
        
    Returns:
        配置好的市场研究智能体实例
        
    Example:
        >>> researcher = create_market_research_agent()
        >>> result = researcher.execute({"content": "分析电动汽车市场趋势"})
    """
    return WebSearchAgent(
        name=name,
        description="进行市场研究和竞争分析",
        temperature=0.3,
        enable_news=True,
        enable_fact_check=True,
        custom_prompt="""你是专业的市场研究分析师。

职责：
1. 搜索最新的市场趋势和行业动态
2. 分析竞争对手信息
3. 收集消费者反馈和市场数据
4. 识别商业机会和挑战

工作方法：
- 使用 web_search 搜索市场报告和行业分析
- 使用 news_search 跟踪最新行业新闻
- 使用 fact_check 验证市场数据的准确性
- 提供结构化的市场分析和建议

输出要求：
- 清晰的数据和事实
- 可操作的建议
- 信息来源引用
- SWOT分析（如适用）"""
    )


# =============================================================================
# 使用示例和演示
# =============================================================================

if __name__ == "__main__":
    from multi_agent_framework import AgentFramework, create_analyst_agent, create_summarizer_agent
    
    print("=" * 70)
    print("  互联网搜索智能体演示")
    print("=" * 70)
    
    # 创建框架
    framework = AgentFramework(name="WebSearchDemo")
    
    # 注册搜索智能体
    searcher = create_web_search_agent()
    framework.register_agent(searcher)
    
    # 示例 1: 直接搜索
    print("\n" + "=" * 70)
    print("  示例 1: 搜索最新技术趋势")
    print("=" * 70)
    
    try:
        result = framework.execute(
            task="2024年人工智能领域的最新发展趋势是什么？",
            workflow_type="pipeline",
            workflow_config={
                "agent_sequence": ["web_searcher"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n搜索结果:\n{result['result'][:1500]}...")
    except Exception as e:
        print(f"执行出错: {e}")
        print("提示: 请确保已安装 duckduckgo-search 包")
    
    # 示例 2: 结合其他智能体
    print("\n" + "=" * 70)
    print("  示例 2: 搜索 + 分析 + 总结")
    print("=" * 70)
    
    try:
        framework.register_agent(create_analyst_agent())
        framework.register_agent(create_summarizer_agent())
        
        result = framework.execute(
            task="量子计算的最新突破和应用前景",
            workflow_type="pipeline",
            workflow_config={
                "agent_sequence": ["web_searcher", "analyst", "summarizer"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n最终结果:\n{result['result']}")
    except Exception as e:
        print(f"执行出错: {e}")
    
    # 示例 3: 并行搜索和分析
    print("\n" + "=" * 70)
    print("  示例 3: 并行模式 - 多角度研究")
    print("=" * 70)
    
    try:
        from multi_agent_framework import create_researcher_agent
        framework.register_agent(create_researcher_agent("domain_expert"))
        
        result = framework.execute(
            task="区块链技术在供应链管理中的应用",
            workflow_type="parallel",
            workflow_config={
                "agent_names": ["web_searcher", "domain_expert"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n综合结果:\n{result['result'][:1500]}...")
    except Exception as e:
        print(f"执行出错: {e}")
    
    print("\n" + "=" * 70)
    print("  演示完成")
    print("=" * 70)
