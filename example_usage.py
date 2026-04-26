"""
多智能体框架使用示例

演示如何使用 multi_agent_framework 和 advanced_multi_agent_framework
构建各种实际应用场景
"""

from __future__ import annotations

import os
from langchain_core.tools import tool
from multi_agent_framework import (
    AgentFramework, 
    LLMAgent, 
    ReactAgent,
    create_analyst_agent,
    create_writer_agent,
    create_researcher_agent,
    create_critic_agent,
    create_summarizer_agent,
)
from advanced_multi_agent_framework import AdvancedAgentFramework

# =============================================================================
# 自定义工具
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """计算算术表达式"""
    import re
    expr = re.sub(r"[^0-9+\-*/().]", "", expression)
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误: {e}"


@tool
def word_counter(text: str) -> str:
    """统计文本中的词数"""
    words = text.split()
    return f"词数: {len(words)}"


# =============================================================================
# 自定义智能体
# =============================================================================

def create_math_agent() -> ReactAgent:
    """创建数学计算智能体"""
    return ReactAgent(
        name="math_expert",
        description="执行数学计算",
        system_prompt="你是一个数学专家。使用提供的工具进行精确计算。",
        temperature=0.1,
        tools=[calculate]
    )


def create_text_analyzer() -> ReactAgent:
    """创建文本分析智能体"""
    return ReactAgent(
        name="text_analyzer",
        description="分析文本特征",
        system_prompt="你分析文本的各种特征，如长度、词数等。",
        temperature=0.2,
        tools=[word_counter]
    )


def create_code_reviewer() -> LLMAgent:
    """创建代码审查智能体"""
    return LLMAgent(
        name="code_reviewer",
        description="审查代码质量和最佳实践",
        system_prompt="""你是一个资深的代码审查专家。请检查代码的：
1. 可读性和命名规范
2. 潜在的性能问题
3. 安全性考虑
4. 最佳实践遵循
5. 可能的bug

提供具体的改进建议。""",
        temperature=0.2
    )


def create_content_planner() -> LLMAgent:
    """创建内容规划智能体"""
    return LLMAgent(
        name="content_planner",
        description="制定内容创作计划",
        system_prompt="""你是一个内容策划专家。为给定的主题制定详细的内容创作计划，包括：
1. 目标受众分析
2. 核心要点（3-5个）
3. 内容结构建议
4. 写作风格推荐
5. 预期的读者反馈

用中文输出，格式清晰。""",
        temperature=0.4
    )


def create_seo_optimizer() -> LLMAgent:
    """创建SEO优化智能体"""
    return LLMAgent(
        name="seo_optimizer",
        description="优化内容的SEO表现",
        system_prompt="""你是一个SEO专家。请对提供的内容进行SEO优化建议：
1. 关键词分析和推荐
2. 标题优化建议
3. 元描述建议
4. 内容结构改进
5. 内部链接建议

用中文输出具体可操作的建议。""",
        temperature=0.3
    )


# =============================================================================
# 示例 1: 内容创作工作流
# =============================================================================

def demo_content_creation():
    """演示内容创作工作流"""
    print("\n" + "=" * 70)
    print("  示例 1: 内容创作工作流")
    print("  流程: Planner -> Writer -> SEO Optimizer -> Critic")
    print("=" * 70)
    
    framework = AgentFramework(name="ContentCreationFramework")
    
    # 注册智能体
    framework.register_agent(create_content_planner())
    framework.register_agent(create_writer_agent())
    framework.register_agent(create_seo_optimizer())
    framework.register_agent(create_critic_agent())
    
    # 执行流水线
    task = "撰写一篇关于'人工智能在教育中的应用'的博客文章"
    
    result = framework.execute(
        task=task,
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["content_planner", "writer", "seo_optimizer"]
        }
    )
    
    print(f"\n任务: {result['task']}")
    print(f"\n最终结果:\n{result['result'][:1500]}...")
    
    # 显示每个智能体的贡献
    print(f"\n各智能体输出:")
    for agent_name, output in result['agent_outputs'].items():
        preview = output.get('output', '')[:200]
        print(f"  - {agent_name}: {preview}...")


# =============================================================================
# 示例 2: 代码审查工作流
# =============================================================================

def demo_code_review():
    """演示代码审查工作流"""
    print("\n" + "=" * 70)
    print("  示例 2: 代码审查工作流")
    print("  流程: Code Reviewer -> Critic -> Summarizer")
    print("=" * 70)
    
    framework = AgentFramework(name="CodeReviewFramework")
    
    framework.register_agent(create_code_reviewer())
    framework.register_agent(create_critic_agent("senior_reviewer"))
    framework.register_agent(create_summarizer_agent())
    
    code_snippet = """
def process_data(data_list):
    result = []
    for i in range(len(data_list)):
        if data_list[i] > 0:
            result.append(data_list[i] * 2)
    return result
    """
    
    task = f"审查以下Python代码并提供改进建议:\n{code_snippet}"
    
    result = framework.execute(
        task=task,
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["code_reviewer", "senior_reviewer", "summarizer"]
        }
    )
    
    print(f"\n任务: 代码审查")
    print(f"\n审查摘要:\n{result['result']}")


# =============================================================================
# 示例 3: 研究分析工作流（并行模式）
# =============================================================================

def demo_research_analysis():
    """演示研究分析工作流"""
    print("\n" + "=" * 70)
    print("  示例 3: 研究分析工作流（并行模式）")
    print("  流程: Researcher + Analyst (并行) -> Summarizer")
    print("=" * 70)
    
    framework = AgentFramework(name="ResearchFramework")
    
    framework.register_agent(create_researcher_agent())
    framework.register_agent(create_analyst_agent())
    framework.register_agent(create_summarizer_agent())
    
    # 先并行收集信息，再总结
    task = "深度学习在自然语言处理中的最新进展"
    
    # 第一步：并行研究和分析
    parallel_result = framework.execute(
        task=task,
        workflow_type="parallel",
        workflow_config={
            "agent_names": ["researcher", "analyst"]
        }
    )
    
    print(f"\n任务: {task}")
    print(f"\n并行收集的信息:\n{parallel_result['result'][:1000]}...")
    
    # 第二步：总结
    summary_result = framework.execute(
        task=f"总结以下研究发现:\n{parallel_result['result']}",
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["summarizer"]
        }
    )
    
    print(f"\n最终总结:\n{summary_result['result']}")


# =============================================================================
# 示例 4: 监督者路由模式
# =============================================================================

def demo_supervisor_routing():
    """演示监督者路由模式"""
    print("\n" + "=" * 70)
    print("  示例 4: 监督者路由模式")
    print("  流程: Supervisor -> [Math Expert | Text Analyzer]")
    print("=" * 70)
    
    framework = AgentFramework(name="SupervisorFramework")
    
    # 创建监督者
    supervisor = LLMAgent(
        name="supervisor",
        description="根据任务类型路由到合适的专家",
        system_prompt="""分析用户任务，判断需要什么类型的专家：
- 如果涉及数学计算，回复"math"
- 如果涉及文本分析，回复"text"
只回复一个词，不要其他内容。""",
        temperature=0.1
    )
    
    framework.register_agent(supervisor)
    framework.register_agent(create_math_agent())
    framework.register_agent(create_text_analyzer())
    
    # 定义路由函数
    def custom_router(decision, worker_names):
        if "math" in decision.lower():
            return "math_expert"
        elif "text" in decision.lower():
            return "text_analyzer"
        return worker_names[0] if worker_names else None
    
    # 测试数学任务
    math_task = "计算 (123 + 456) * 2 的结果"
    result = framework.execute(
        task=math_task,
        workflow_type="supervisor",
        workflow_config={
            "supervisor_name": "supervisor",
            "worker_names": ["math_expert", "text_analyzer"],
            "router_func": custom_router
        }
    )
    
    print(f"\n任务 1: {math_task}")
    print(f"路由到: {result['metadata'].get('route')}")
    print(f"结果: {result['result']}")
    
    # 测试文本任务
    text_task = "统计这段文字的詞数：人工智能正在改变世界"
    result = framework.execute(
        task=text_task,
        workflow_type="supervisor",
        workflow_config={
            "supervisor_name": "supervisor",
            "worker_names": ["math_expert", "text_analyzer"],
            "router_func": custom_router
        }
    )
    
    print(f"\n任务 2: {text_task}")
    print(f"路由到: {result['metadata'].get('route')}")
    print(f"结果: {result['result']}")


# =============================================================================
# 示例 5: 高级框架 - 反思模式
# =============================================================================

def demo_advanced_reflection():
    """演示高级框架的反思模式"""
    print("\n" + "=" * 70)
    print("  示例 5: 高级框架 - 反思循环")
    print("  流程: Generate -> Critique -> Revise (循环)")
    print("=" * 70)
    
    framework = AdvancedAgentFramework(name="ReflectionFramework")
    
    # 创建生成者和批评者
    generator = LLMAgent(
        name="generator",
        description="生成技术解释",
        system_prompt="""你是一个技术作家。用通俗易懂的语言解释技术概念。
要求：
- 使用类比和例子
- 避免过于专业的术语
- 控制在150字以内""",
        temperature=0.5
    )
    
    critic = LLMAgent(
        name="critic",
        description="评估解释的质量",
        system_prompt="""你是教育专家，评估技术解释的质量。
检查：
1. 是否易于理解
2. 是否有恰当的类比
3. 是否过于简化或复杂

如果满意，回复"满意，通过"。
如果不满意，提供具体的改进建议。""",
        temperature=0.2
    )
    
    framework.register_agent(generator)
    framework.register_agent(critic)
    
    # 构建反思图
    framework.build_reflection_graph("generator", "critic", max_iterations=3)
    framework.compile()
    
    # 执行
    task = "解释什么是区块链"
    result = framework.execute(task)
    
    print(f"\n任务: {task}")
    print(f"迭代次数: {result['iterations']}")
    print(f"\n最终输出:\n{result['output']}")
    
    # 显示历史
    print(f"\n执行历史:")
    for i, step in enumerate(result['history'], 1):
        print(f"  步骤 {i}: {step['agent']}")


# =============================================================================
# 示例 6: 动态添加/删除智能体
# =============================================================================

def demo_dynamic_management():
    """演示动态管理智能体"""
    print("\n" + "=" * 70)
    print("  示例 6: 动态智能体管理")
    print("=" * 70)
    
    framework = AgentFramework(name="DynamicFramework")
    
    # 初始注册
    framework.register_agent(create_analyst_agent())
    framework.register_agent(create_writer_agent())
    
    print(f"\n初始智能体: {framework.list_agents()}")
    
    # 动态添加
    print("\n添加新智能体...")
    framework.register_agent(create_researcher_agent())
    framework.register_agent(create_summarizer_agent())
    
    print(f"添加后: {framework.list_agents()}")
    
    # 使用新添加的智能体
    task = "机器学习的基本概念"
    result = framework.execute(
        task=task,
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["researcher", "analyst", "summarizer"]
        }
    )
    
    print(f"\n任务: {task}")
    print(f"结果:\n{result['result'][:800]}...")
    
    # 动态移除
    print("\n移除 researcher...")
    framework.unregister_agent("researcher")
    print(f"移除后: {framework.list_agents()}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  多智能体框架 - 实用示例集")
    print("=" * 70)
    
    try:
        # 运行所有示例
        demo_content_creation()
        demo_code_review()
        demo_research_analysis()
        demo_supervisor_routing()
        demo_advanced_reflection()
        demo_dynamic_management()
        
        print("\n" + "=" * 70)
        print("  所有示例执行完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
