"""
多智能体框架 - 快速测试脚本

用于验证框架的基本功能是否正常
"""

from multi_agent_framework import (
    AgentFramework,
    LLMAgent,
    create_analyst_agent,
    create_writer_agent,
)

def test_basic_framework():
    """测试基础框架功能"""
    print("=" * 60)
    print("  测试基础框架")
    print("=" * 60)
    
    # 1. 创建框架
    framework = AgentFramework(name="TestFramework")
    print(f"✓ 创建框架: {framework}")
    
    # 2. 注册智能体
    analyst = create_analyst_agent()
    writer = create_writer_agent()
    
    framework.register_agent(analyst)
    framework.register_agent(writer)
    
    print(f"✓ 已注册智能体: {framework.list_agents()}")
    
    # 3. 执行流水线任务
    print("\n执行流水线任务...")
    result = framework.execute(
        task="人工智能的未来发展趋势",
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["analyst", "writer"]
        }
    )
    
    print(f"✓ 任务完成")
    print(f"  结果长度: {len(result['result'])} 字符")
    print(f"  参与的智能体: {list(result['agent_outputs'].keys())}")
    print(f"\n结果预览:\n{result['result'][:300]}...\n")
    
    # 4. 动态管理测试
    print("测试动态管理...")
    framework.unregister_agent("writer")
    print(f"✓ 移除后: {framework.list_agents()}")
    
    return True


def test_custom_agent():
    """测试自定义智能体"""
    print("\n" + "=" * 60)
    print("  测试自定义智能体")
    print("=" * 60)
    
    # 创建自定义智能体
    translator = LLMAgent(
        name="translator",
        description="翻译专家",
        system_prompt="你是一个翻译专家，将英文翻译成中文。保持原意，语言流畅。",
        temperature=0.2
    )
    
    framework = AgentFramework()
    framework.register_agent(translator)
    
    # 执行翻译任务
    result = framework.execute(
        task="Artificial Intelligence is transforming the world.",
        workflow_type="pipeline",
        workflow_config={
            "agent_sequence": ["translator"]
        }
    )
    
    print(f"✓ 翻译结果: {result['result']}")
    
    return True


def test_parallel_mode():
    """测试并行模式"""
    print("\n" + "=" * 60)
    print("  测试并行模式")
    print("=" * 60)
    
    from multi_agent_framework import create_researcher_agent, create_analyst_agent
    
    framework = AgentFramework()
    framework.register_agent(create_researcher_agent("researcher1"))
    framework.register_agent(create_analyst_agent("analyst1"))
    
    result = framework.execute(
        task="什么是机器学习？",
        workflow_type="parallel",
        workflow_config={
            "agent_names": ["researcher1", "analyst1"]
        }
    )
    
    print(f"✓ 并行执行完成")
    print(f"  结果长度: {len(result['result'])} 字符")
    print(f"  包含的输出: {list(result['agent_outputs'].keys())}")
    print(f"\n结果预览:\n{result['result'][:300]}...\n")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  多智能体框架 - 快速测试")
    print("=" * 60 + "\n")
    
    try:
        # 运行测试
        test_basic_framework()
        test_custom_agent()
        test_parallel_mode()
        
        print("\n" + "=" * 60)
        print("  ✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
