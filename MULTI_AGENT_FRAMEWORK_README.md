# 多智能体协作框架 (Multi-Agent Collaboration Framework)

这是一个基于 LangChain 和 LangGraph 构建的可扩展多智能体协作框架，支持动态添加/删除智能体，并提供多种协作模式。

## 🌟 特性

### 基础框架 (`multi_agent_framework.py`)
- ✅ **模块化设计** - 智能体即插即用
- ✅ **三种协作模式**:
  - 流水线 (Pipeline): 顺序执行
  - 监督者 (Supervisor): 智能路由
  - 并行 (Parallel): 同时执行
- ✅ **动态管理** - 运行时添加/删除智能体
- ✅ **预定义模板** - 快速创建常用智能体

### 高级框架 (`advanced_multi_agent_framework.py`)
- ✅ **基于状态图** - 使用 LangGraph StateGraph
- ✅ **复杂工作流**:
  - 条件分支
  - 循环迭代（反思模式）
  - 嵌套子图
- ✅ **记忆持久化** - 支持对话历史保存
- ✅ **可视化** - 生成工作流图

## 📦 安装依赖

```bash
pip install langchain langchain-openai langgraph
```

## 🚀 快速开始

### 1. 基础用法 - 流水线模式

```python
from multi_agent_framework import AgentFramework, create_analyst_agent, create_writer_agent

# 创建框架
framework = AgentFramework()

# 注册智能体
framework.register_agent(create_analyst_agent())
framework.register_agent(create_writer_agent())

# 执行任务
result = framework.execute(
    task="人工智能在医疗中的应用",
    workflow_type="pipeline",
    workflow_config={
        "agent_sequence": ["analyst", "writer"]
    }
)

print(result['result'])
```

### 2. 并行模式

```python
from multi_agent_framework import AgentFramework, create_researcher_agent, create_analyst_agent

framework = AgentFramework()
framework.register_agent(create_researcher_agent())
framework.register_agent(create_analyst_agent())

# 并行执行多个智能体
result = framework.execute(
    task="什么是量子计算？",
    workflow_type="parallel",
    workflow_config={
        "agent_names": ["researcher", "analyst"]
    }
)
```

### 3. 监督者路由模式

```python
from multi_agent_framework import AgentFramework, LLMAgent

framework = AgentFramework()

# 创建监督者
supervisor = LLMAgent(
    name="supervisor",
    description="路由决策",
    system_prompt="判断任务是数学计算还是文本分析，回复'math'或'text'"
)

framework.register_agent(supervisor)
framework.register_agent(math_expert)
framework.register_agent(text_expert)

# 自定义路由函数
def router(decision, workers):
    return "math_expert" if "math" in decision else "text_expert"

result = framework.execute(
    task="计算 123 * 456",
    workflow_type="supervisor",
    workflow_config={
        "supervisor_name": "supervisor",
        "worker_names": ["math_expert", "text_expert"],
        "router_func": router
    }
)
```

### 4. 高级框架 - 反思模式

```python
from advanced_multi_agent_framework import AdvancedAgentFramework
from multi_agent_framework import create_writer_agent, create_critic_agent

framework = AdvancedAgentFramework()
framework.register_agent(create_writer_agent("generator"))
framework.register_agent(create_critic_agent("critic"))

# 构建反思循环图
framework.build_reflection_graph("generator", "critic", max_iterations=3)
framework.compile()

# 执行（会自动迭代优化）
result = framework.execute("解释相对论")
print(f"迭代次数: {result['iterations']}")
print(f"最终输出: {result['output']}")
```

### 5. 动态管理智能体

```python
framework = AgentFramework()

# 注册
framework.register_agent(agent1)
framework.register_agent(agent2)

# 查看
print(framework.list_agents())  # ['agent1', 'agent2']

# 移除
framework.unregister_agent("agent1")

# 添加新智能体
framework.register_agent(new_agent)
```

## 🏗️ 创建自定义智能体

### 方法 1: 继承 BaseAgent

```python
from multi_agent_framework import BaseAgent

class MyCustomAgent(BaseAgent):
    def execute(self, input_data):
        # 你的逻辑
        result = process(input_data["content"])
        return {"output": result, "agent_name": self.name}

# 使用
my_agent = MyCustomAgent(
    name="my_agent",
    description="我的自定义智能体",
    system_prompt="你是一个..."
)
framework.register_agent(my_agent)
```

### 方法 2: 使用预定义的 LLMAgent

```python
from multi_agent_framework import LLMAgent

agent = LLMAgent(
    name="translator",
    description="翻译专家",
    system_prompt="你是一个专业的翻译，将英文翻译成中文",
    temperature=0.3
)
```

### 方法 3: 使用 ReAct 智能体（带工具）

```python
from multi_agent_framework import ReactAgent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

agent = ReactAgent(
    name="researcher",
    description="研究专家",
    system_prompt="使用工具进行研究",
    tools=[search]
)
```

## 📊 协作模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **流水线** | 固定流程的任务 | 简单、可预测 | 不够灵活 |
| **监督者** | 需要分类路由 | 智能决策 | 需要训练监督者 |
| **并行** | 独立子任务 | 高效、全面 | 结果需要聚合 |
| **反思** | 需要迭代优化 | 质量高 | 耗时较长 |

## 🎯 实际应用示例

查看 `example_usage.py` 文件，包含以下完整示例：

1. **内容创作工作流**: Planner → Writer → SEO Optimizer → Critic
2. **代码审查工作流**: Code Reviewer → Senior Reviewer → Summarizer
3. **研究分析工作流**: Researcher + Analyst (并行) → Summarizer
4. **监督者路由**: Supervisor → [Math Expert | Text Analyzer]
5. **反思循环**: Generate → Critique → Revise (循环)
6. **动态管理**: 运行时添加/删除智能体

运行示例：
```bash
python example_usage.py
```

## 🔧 高级功能

### 1. 自定义聚合函数（并行模式）

```python
def custom_aggregator(results):
    """自定义结果聚合逻辑"""
    return f"综合观点:\n" + "\n".join([
        f"- {name}: {output[:100]}" 
        for name, output in results.items()
    ])

result = framework.execute(
    task="分析AI的影响",
    workflow_type="parallel",
    workflow_config={
        "agent_names": ["researcher", "analyst"],
        "aggregator": custom_aggregator
    }
)
```

### 2. 构建分支工作流（高级框架）

```python
framework = AdvancedAgentFramework()

# 注册智能体...

# 构建分支图
branch_config = {
    "technical_branch": ["researcher", "analyst"],
    "creative_branch": ["writer", "critic"]
}

framework.build_branching_graph(branch_config, merger_name="summarizer")
framework.compile()
```

### 3. 持久化记忆

```python
# 启用记忆
framework = AdvancedAgentFramework(use_memory=True)

# 执行时指定线程ID
result1 = framework.execute("问题1", thread_id="session_1")
result2 = framework.execute("问题2", thread_id="session_1")  # 共享上下文
```

## 📁 项目结构

```
week18/
├── multi_agent_framework.py          # 基础框架
├── advanced_multi_agent_framework.py # 高级框架
├── example_usage.py                  # 使用示例
├── MULTI_AGENT_FRAMEWORK_README.md   # 本文档
├── 0_langchain_multi_agent.py        # 参考代码
├── 1_deepagents_multi_agent.py       # 参考代码
└── ...
```

## 💡 最佳实践

1. **智能体粒度**: 每个智能体应该有明确的职责，避免过于通用
2. **系统提示词**: 编写清晰的 system prompt，明确智能体的角色和输出格式
3. **温度设置**: 
   - 分析类任务: 0.1-0.3
   - 创作类任务: 0.4-0.7
   - 事实类任务: 0.0-0.2
4. **错误处理**: 在执行前验证智能体是否已注册
5. **性能优化**: 对于简单任务，避免过度使用多智能体

## 🐛 常见问题

**Q: 如何调试智能体的输出？**
```python
result = framework.execute(...)
# 查看每个智能体的详细输出
for agent_name, output in result['agent_outputs'].items():
    print(f"{agent_name}: {output}")
```

**Q: 智能体执行失败怎么办？**
```python
try:
    result = framework.execute(task, ...)
except ValueError as e:
    print(f"检查智能体是否注册: {e}")
```

**Q: 如何保存和加载框架配置？**
```python
import pickle

# 保存
with open('framework.pkl', 'wb') as f:
    pickle.dump(framework.agents, f)

# 加载
with open('framework.pkl', 'rb') as f:
    agents = pickle.load(f)
```

## 📚 参考资料

- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- 项目中的参考实现: `0_langchain_multi_agent.py`, `1_deepagents_multi_agent.py`

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
