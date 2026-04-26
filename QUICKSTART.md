# 快速入门指南 - 5分钟上手多智能体框架

## 🚀 安装

```bash
pip install langchain langchain-openai langgraph
```

## 📝 第一个多智能体应用

### 步骤 1: 导入框架

```python
from multi_agent_framework import AgentFramework, create_analyst_agent, create_writer_agent
```

### 步骤 2: 创建框架并注册智能体

```python
# 创建框架
framework = AgentFramework()

# 注册智能体（就像搭积木一样简单）
framework.register_agent(create_analyst_agent())
framework.register_agent(create_writer_agent())
```

### 步骤 3: 执行任务

```python
result = framework.execute(
    task="人工智能在教育中的应用",
    workflow_type="pipeline",
    workflow_config={
        "agent_sequence": ["analyst", "writer"]  # 先分析，后写作
    }
)

print(result['result'])
```

**完成！** 🎉 你已经创建了一个多智能体系统。

---

## 💡 常见场景速查

### 场景 1: 内容创作流水线

```python
from multi_agent_framework import (
    AgentFramework, 
    create_researcher_agent,
    create_writer_agent,
    create_critic_agent
)

framework = AgentFramework()
framework.register_agent(create_researcher_agent())
framework.register_agent(create_writer_agent())
framework.register_agent(create_critic_agent())

# 研究 → 写作 → 评审
result = framework.execute(
    task="撰写关于区块链的文章",
    workflow_type="pipeline",
    workflow_config={
        "agent_sequence": ["researcher", "writer", "critic"]
    }
)
```

### 场景 2: 多角度分析（并行）

```python
from multi_agent_framework import AgentFramework, create_analyst_agent, create_researcher_agent

framework = AgentFramework()
framework.register_agent(create_analyst_agent())
framework.register_agent(create_researcher_agent())

# 同时从多个角度分析
result = framework.execute(
    task="分析AI的影响",
    workflow_type="parallel",
    workflow_config={
        "agent_names": ["analyst", "researcher"]
    }
)
```

### 场景 3: 智能路由

```python
from multi_agent_framework import AgentFramework, LLMAgent

framework = AgentFramework()

# 创建监督者
supervisor = LLMAgent(
    name="supervisor",
    description="路由决策",
    system_prompt="判断任务是'数学'还是'文本'类型，只回复一个词"
)
framework.register_agent(supervisor)
framework.register_agent(math_expert)
framework.register_agent(text_expert)

# 自动路由到合适的专家
result = framework.execute(
    task="计算 123 * 456",
    workflow_type="supervisor",
    workflow_config={
        "supervisor_name": "supervisor",
        "worker_names": ["math_expert", "text_expert"]
    }
)
```

### 场景 4: 动态添加智能体

```python
# 随时添加新智能体
from multi_agent_framework import create_summarizer_agent

framework.register_agent(create_summarizer_agent())

# 立即使用
result = framework.execute(
    task="总结这篇文章...",
    workflow_type="pipeline",
    workflow_config={
        "agent_sequence": ["summarizer"]
    }
)
```

---

## 🎯 预定义智能体清单

框架提供了以下开箱即用的智能体：

| 智能体 | 用途 | 创建函数 |
|--------|------|----------|
| **分析师** | 分析主题，提取要点 | `create_analyst_agent()` |
| **作家** | 撰写流畅文章 | `create_writer_agent()` |
| **研究员** | 深入研究提供详情 | `create_researcher_agent()` |
| **评论家** | 评审并提出建议 | `create_critic_agent()` |
| **总结者** | 总结长篇内容 | `create_summarizer_agent()` |

---

## 🔧 自定义智能体（3步）

### 第 1 步: 定义智能体

```python
from multi_agent_framework import LLMAgent

translator = LLMAgent(
    name="translator",
    description="翻译专家",
    system_prompt="你是一个翻译专家，将英文翻译成中文",
    temperature=0.2
)
```

### 第 2 步: 注册

```python
framework.register_agent(translator)
```

### 第 3 步: 使用

```python
result = framework.execute(
    task="Hello World",
    workflow_type="pipeline",
    workflow_config={"agent_sequence": ["translator"]}
)
```

---

## 🌟 高级功能（可选）

### 反思模式（迭代优化）

```python
from advanced_multi_agent_framework import AdvancedAgentFramework
from multi_agent_framework import create_writer_agent, create_critic_agent

framework = AdvancedAgentFramework()
framework.register_agent(create_writer_agent("generator"))
framework.register_agent(create_critic_agent("critic"))

# 构建反思循环：生成 → 批评 → 修改（最多3次）
framework.build_reflection_graph("generator", "critic", max_iterations=3)
framework.compile()

# 执行（会自动迭代优化）
result = framework.execute("解释量子计算")
print(f"迭代次数: {result['iterations']}")
print(f"最终结果: {result['output']}")
```

---

## 📊 三种工作流模式对比

```
┌─────────────┬──────────────────┬────────────────┬─────────────────┐
│   模式       │     适用场景      │     优点        │      缺点        │
├─────────────┼──────────────────┼────────────────┼─────────────────┤
│ Pipeline    │ 固定流程          │ 简单可预测      │ 不够灵活         │
│ Parallel    │ 独立子任务        │ 高效全面        │ 需聚合结果       │
│ Supervisor  │ 需要分类路由      │ 智能决策        │ 需训练监督者     │
└─────────────┴──────────────────┴────────────────┴─────────────────┘
```

---

## ❓ 常见问题

### Q1: 如何查看每个智能体的输出？

```python
result = framework.execute(...)

# 查看所有智能体的输出
for agent_name, output in result['agent_outputs'].items():
    print(f"{agent_name}: {output['output'][:200]}...")
```

### Q2: 如何调试工作流？

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看执行历史（高级框架）
result = framework.execute(task)
for step in result['history']:
    print(f"步骤: {step['agent']}")
    print(f"输出: {step['output'][:100]}")
```

### Q3: 如何保存和复用配置？

```python
import pickle

# 保存
config = {
    'agents': framework.list_agents(),
    'workflow_type': 'pipeline',
    'workflow_config': {...}
}
with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)

# 加载
with open('config.pkl', 'rb') as f:
    config = pickle.load(f)
```

---

## 🎓 下一步

1. **阅读完整文档**: `MULTI_AGENT_FRAMEWORK_README.md`
2. **查看示例**: `example_usage.py`
3. **理解设计**: `DESIGN_DOCUMENT.md`
4. **运行测试**: `python test_framework.py`

---

## 💬 需要帮助？

- 📖 查看文档
- 🐛 提交 Issue
- 💡 查看示例代码

**祝使用愉快！** 🚀
