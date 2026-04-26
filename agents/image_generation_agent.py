"""
图片生成智能体 - Image Generation Agent

这是一个可以集成到多智能体框架中的智能体，能够根据文本描述生成图片。
基于 SiliconFlow API，支持多种AI绘画模型和图像编辑功能。

依赖安装:
    pip install openai requests  # openai用于调用 SiliconFlow API，requests用于下载图片

使用示例:
    from agents.image_generation_agent import create_image_generator_agent
    from multi_agent_framework import AgentFramework
    
    framework = AgentFramework()
    image_agent = create_image_generator_agent()
    framework.register_agent(image_agent)
    
    result = framework.execute(
        task="生成一张未来城市的图片",
        workflow_type="pipeline",
        workflow_config={"agent_sequence": ["image_generator"]}
    )
"""

from __future__ import annotations

import os
import sys
import base64
import requests
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from multi_agent_framework import BaseAgent, get_llm


# =============================================================================
# 配置
# =============================================================================

API_KEY = os.environ.get("SILICONFLOW_API_KEY")
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
IMAGE_MODEL = os.environ.get("SILICONFLOW_IMAGE_MODEL", "Kwai-Kolors/Kolors")


# =============================================================================
# 图片生成工具定义
# =============================================================================

@tool
def generate_image_flux(prompt: str, size: str = "1024x1024", num_images: int = 1,
                       guidance_scale: float = 7.5, seed: Optional[int] = None) -> str:
    """
    使用 FLUX 模型生成图片（通过 SiliconFlow API）
    
    Args:
        prompt: 图片描述提示词（英文效果更佳）
        size: 图片尺寸，可选："1024x1024", "768x1344", "1344x768"
        num_images: 生成图片数量（默认1，最大4）
        guidance_scale: 提示词相关性（3.0-20.0，默认7.5）
        seed: 随机种子（可选，用于复现结果）
        
    Returns:
        生成结果信息（包含图片URL）
    """
    try:
        from openai import OpenAI
        
        # 获取 API Key
        api_key = os.environ.get("SILICONFLOW_API_KEY", API_KEY)
        if not api_key:
            return "错误: 未设置 SILICONFLOW_API_KEY 环境变量"
        
        base_url = os.environ.get("SILICONFLOW_BASE_URL", BASE_URL)
        model = os.environ.get("SILICONFLOW_IMAGE_MODEL", IMAGE_MODEL)
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 构建请求参数
        params: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": min(num_images, 4),  # 最多4张
        }
        
        # 添加可选参数
        extra_body: Dict[str, Any] = {}
        if seed is not None:
            extra_body["seed"] = seed
        if guidance_scale:
            extra_body["guidance_scale"] = max(3.0, min(guidance_scale, 20.0))
        if extra_body:
            params["extra_body"] = extra_body
        
        # 调用 FLUX 模型
        response = client.images.generate(**params)
        
        # 提取图片URL
        image_urls = [img.url for img in response.data] if response.data else []
        
        result = f"""
图片生成成功！

模型: {model}
提示词: {prompt}
尺寸: {size}
生成数量: {len(image_urls)}
引导比例: {guidance_scale}
"""
        
        if seed is not None:
            result += f"随机种子: {seed}\n"
        
        for i, url in enumerate(image_urls, 1):
            result += f"\n图片 {i} URL: {url}"
        
        result += "\n\n注意: URL有效期为60分钟，请及时下载保存。"
        
        return result
    
    except ImportError:
        return "错误: 未安装 openai 包。请运行: pip install openai"
    except Exception as e:
        return f"图片生成失败: {str(e)}"


@tool
def generate_image_kolors(prompt: str, size: str = "1024x1024", num_images: int = 1) -> str:
    """
    使用 Kwai-Kolors 模型生成图片（通过 SiliconFlow API）
    
    Args:
        prompt: 图片描述提示词
        size: 图片尺寸，可选："1024x1024"
        num_images: 生成图片数量（默认1）
        
    Returns:
        生成结果信息
    """
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("SILICONFLOW_API_KEY", API_KEY)
        if not api_key:
            return "错误: 未设置 SILICONFLOW_API_KEY 环境变量"
        
        base_url = os.environ.get("SILICONFLOW_BASE_URL", BASE_URL)
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        from typing import cast, Literal
        response = client.images.generate(
            model="Kwai-Kolors/Kolors",
            prompt=prompt,
            size=cast(Any, size),
            n=min(num_images, 4),
        )
        
        image_urls = [img.url for img in response.data] if response.data else []
        
        result = f"""
Kolors 图片生成完成！

提示词: {prompt}
尺寸: {size}
生成数量: {len(image_urls)}
"""
        
        for i, url in enumerate(image_urls, 1):
            result += f"\n图片 {i} URL: {url}"
        
        result += "\n\n注意: URL有效期为60分钟，请及时下载保存。"
        
        return result
    
    except ImportError:
        return "错误: 未安装 openai 包。请运行: pip install openai"
    except Exception as e:
        return f"Kolors 生成失败: {str(e)}"


@tool
def enhance_prompt(original_prompt: str) -> str:
    """
    优化图片生成提示词，使其更详细和专业
    
    Args:
        original_prompt: 原始提示词
        
    Returns:
        优化后的提示词
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = get_llm(temperature=0.7)
        
        system_prompt = """你是专业的AI绘画提示词工程师。

你的任务是将简单的图片描述优化为详细、专业的提示词。

优化原则：
1. 添加详细的视觉细节（颜色、光线、构图、风格等）
2. 指定艺术风格和媒介（如：photorealistic, digital art, oil painting等）
3. 描述氛围和情感
4. 添加技术参数（如：8k, highly detailed, professional lighting等）
5. 保持原意不变

输出格式：
只输出优化后的提示词，不要其他内容。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"优化以下提示词：{original_prompt}")
        ]
        
        response = llm.invoke(messages)
        return str(response.content)
    
    except Exception as e:
        return f"提示词优化失败: {str(e)}"


@tool
def create_image_variations_siliconflow(image_url: str, prompt: str = "", 
                                       num_variations: int = 2) -> str:
    """
    基于现有图片和提示词创建变体（通过 SiliconFlow API）
    
    Args:
        image_url: 原始图片URL
        prompt: 额外的提示词描述（可选）
        num_variations: 变体数量
        
    Returns:
        变体生成结果
    """
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("SILICONFLOW_API_KEY", API_KEY)
        if not api_key:
            return "错误: 未设置 SILICONFLOW_API_KEY 环境变量"
        
        base_url = os.environ.get("SILICONFLOW_BASE_URL", BASE_URL)
        model = os.environ.get("SILICONFLOW_IMAGE_MODEL", IMAGE_MODEL)
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 构建完整提示词
        full_prompt = f"Based on this image: {image_url}"
        if prompt:
            full_prompt += f". {prompt}"
        
        from typing import cast, Literal
        response = client.images.generate(
            model=cast(str, model),
            prompt=full_prompt,
            size="1024x1024",
            n=min(num_variations, 4),
        )
        
        urls = [img.url for img in response.data] if response.data else []
        
        result = f"""
图片变体生成成功！

原始图片: {image_url}
额外提示: {prompt if prompt else '无'}
生成 {len(urls)} 个变体:
"""
        for i, url in enumerate(urls, 1):
            result += f"\n变体 {i}: {url}"
        
        result += "\n\n注意: URL有效期为60分钟，请及时下载保存。"
        
        return result
    
    except Exception as e:
        return f"图片变体生成失败: {str(e)}"


# =============================================================================
# 图片生成智能体类
# =============================================================================

class ImageGenerationAgent(BaseAgent):
    """
    图片生成智能体
    
    能够根据文本描述生成高质量图片，适合：
    - 创意设计和概念可视化
    - 营销素材生成
    - 艺术作品创作
    - 产品原型展示
    - 教育和演示材料
    
    特性:
        - 支持多种AI绘画模型
        - 自动优化提示词
        - 可自定义风格和参数
        - 与框架无缝集成
    """
    
    def __init__(self, 
                 name: str = "image_generator", 
                 description: str = "根据文本描述生成图片",
                 temperature: float = 0.7,
                 default_model: str = "dall-e-3",
                 default_size: str = "1024x1024",
                 default_quality: str = "standard"):
        """
        初始化图片生成智能体
        
        Args:
            name: 智能体名称
            description: 智能体描述
            temperature: LLM温度参数
            default_model: 默认使用的模型 ("dall-e-3" 或 "stable-diffusion")
            default_size: 默认图片尺寸
            default_quality: 默认图片质量
        """
        
        system_prompt = """你是专业的AI绘画助手和图片生成专家。

你的能力：
1. 理解用户的图片需求并转化为详细的提示词
2. 使用 generate_image_flux 工具通过 FLUX 模型生成高质量图片
3. 使用 generate_image_kolors 工具通过 Kolors 模型生成图片
4. 使用 enhance_prompt 工具优化提示词以获得更好的效果
5. 使用 create_image_variations_siliconflow 工具创建图片变体

工作流程：
1. 分析用户需求，理解想要生成的图片类型和风格
2. 如果用户提供的描述较简单，先使用 enhance_prompt 优化提示词
3. 根据需求选择合适的模型和参数
4. 调用相应的生成工具
5. 向用户报告生成结果，包括图片URL和重要信息

提示词编写技巧：
- 详细描述主体、背景、光线、色彩、构图
- 指定艺术风格（如：photorealistic, anime, watercolor, cyberpunk等）
- 添加质量和细节关键词（如：8k, ultra detailed, professional lighting）
- 对于人物，描述外貌、表情、动作、服装
- 对于场景，描述环境、氛围、时间、天气

注意事项：
- FLUX 模型适合快速生成高质量图片，支持多种尺寸
- Kolors 模型是快手推出的高质量文生图模型
- 提示词越详细，生成效果越好
- 英文提示词通常效果更好
- 生成的图片URL有效期为60分钟，提醒用户及时下载保存
- 可以使用 seed 参数复现相同的生成结果"""
        
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            tools=[generate_image_flux, generate_image_kolors, 
                   enhance_prompt, create_image_variations_siliconflow]
        )
        
        # 保存默认配置
        self.default_model = default_model
        self.default_size = default_size
        self.default_quality = default_quality
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行图片生成任务
        
        Args:
            input_data: 包含 "content" 键的字典，值为图片描述
            
        Returns:
            包含生成结果的字典
        """
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
        from langgraph.prebuilt import create_react_agent
        
        # 创建 ReAct agent
        agent = create_react_agent(self.llm, tools=self.tools)
        
        # 构建消息
        user_prompt = input_data.get("content", "")
        
        # 添加默认参数建议
        enhanced_prompt = f"""
请帮我生成一张图片。

描述：{user_prompt}

建议使用以下参数：
- 模型：{self.default_model}
- 尺寸：{self.default_size}
- 质量：{self.default_quality}

请根据需求选择合适的工具并生成图片。
"""
        
        result = agent.invoke({
            "messages": [HumanMessage(content=enhanced_prompt)]
        })
        
        messages = result.get("messages", [])
        
        # 从消息历史中提取最终结果
        final_answer = ""
        image_urls = []
        
        # 遍历所有消息，找到工具调用的结果
        for msg in messages:
            # 如果是 ToolMessage，说明是工具执行的结果
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                # 检查是否包含图片URL
                if 'url' in str(tool_result).lower() or 'http' in str(tool_result):
                    final_answer = tool_result
                    break
            # 如果是 AIMessage 且没有工具调用，可能是最终回答
            elif hasattr(msg, 'tool_calls') and not msg.tool_calls:
                if hasattr(msg, 'content') and msg.content:
                    final_answer = msg.content
        
        # 如果没找到ToolMessage，使用最后一条消息
        if not final_answer and messages:
            final_answer = getattr(messages[-1], "content", "")
        
        return {
            "output": final_answer,
            "agent_name": self.name,
            "full_trace": messages,
            "image_urls": image_urls
        }


# =============================================================================
# 专用图片生成智能体工厂函数
# =============================================================================

def create_image_generator_agent(name: str = "image_generator") -> BaseAgent:
    """
    创建通用的图片生成智能体
    
    使用 FLUX 模型作为默认模型，适合大多数场景。
    
    Args:
        name: 智能体名称（默认: "image_generator"）
        
    Returns:
        配置好的图片生成智能体实例
        
    Example:
        >>> image_agent = create_image_generator_agent()
        >>> framework.register_agent(image_agent)
    """
    return ImageGenerationAgent(
        name=name,
        description="使用 FLUX 模型生成高质量图片",
        temperature=0.7,
        default_model="flux",
        default_size="1024x1024",
        default_quality="standard"
    )


def create_creative_artist_agent(name: str = "creative_artist") -> BaseAgent:
    """
    创建创意艺术家智能体
    
    专注于艺术创作，使用更高的温度和更创意的提示词优化。
    
    Args:
        name: 智能体名称（默认: "creative_artist"）
        
    Returns:
        配置好的创意艺术家智能体实例
        
    Example:
        >>> artist = create_creative_artist_agent()
        >>> result = artist.execute({"content": "梦幻的星空风景"})
    """
    agent = ImageGenerationAgent(
        name=name,
        description="创作富有艺术感的图片",
        temperature=0.9,
        default_model="flux",
        default_size="1344x768",
        default_quality="hd"
    )
    
    # 自定义系统提示，强调艺术性
    agent.system_prompt = """你是富有创造力的AI艺术家。

专长：
- 创作独特、富有想象力的艺术作品
- 运用各种艺术风格（抽象、印象派、超现实主义等）
- 创造视觉冲击力强、情感丰富的图像
- 实验性的色彩搭配和构图

工作方式：
1. 理解用户的情感需求和艺术偏好
2. 创造性地解释和优化提示词
3. 选择最适合的艺术风格和表现手法
4. 生成令人惊艳的视觉作品

艺术风格参考：
- 数字艺术 (Digital Art)
- 油画风格 (Oil Painting)
- 水彩画 (Watercolor)
- 赛博朋克 (Cyberpunk)
- 蒸汽波 (Vaporwave)
- 极简主义 (Minimalism)
- 超现实主义 (Surrealism)

记住：艺术没有规则，大胆创新！"""
    
    return agent


def create_product_designer_agent(name: str = "product_designer") -> BaseAgent:
    """
    创建产品设计师智能体
    
    专注于产品设计、原型展示和商业用途的图片生成。
    
    Args:
        name: 智能体名称（默认: "product_designer"）
        
    Returns:
        配置好的产品设计师智能体实例
        
    Example:
        >>> designer = create_product_designer_agent()
        >>> result = designer.execute({"content": "设计一个现代风格的咖啡杯"})
    """
    agent = ImageGenerationAgent(
        name=name,
        description="生成产品设计和商业图片",
        temperature=0.5,
        default_model="flux",
        default_size="1024x1024",
        default_quality="hd"
    )
    
    # 自定义系统提示，强调专业性
    agent.system_prompt = """你是专业的产品设计师和商业摄影师。

专长：
- 产品渲染和原型设计
- 商业摄影风格的图片
- 产品展示和营销素材
- UI/UX 设计可视化

工作要求：
1. 生成专业、清晰的产品图片
2. 使用适当的背景和光线
3. 突出产品特点和优势
4. 符合商业标准和品牌调性

图片风格：
- 产品摄影 (Product Photography)
- 极简背景 (Clean Background)
- 专业光线 (Professional Lighting)
- 高分辨率 (High Resolution)
- 商业级质量 (Commercial Quality)

输出要求：
- 简洁专业的描述
- 突出产品功能
- 适合商业用途"""
    
    return agent


def create_concept_visualizer_agent(name: str = "concept_visualizer") -> BaseAgent:
    """
    创建概念可视化智能体
    
    将抽象概念、想法和数据可视化为图片。
    
    Args:
        name: 智能体名称（默认: "concept_visualizer"）
        
    Returns:
        配置好的概念可视化智能体实例
        
    Example:
        >>> visualizer = create_concept_visualizer_agent()
        >>> result = visualizer.execute({"content": "可视化人工智能的工作原理"})
    """
    agent = ImageGenerationAgent(
        name=name,
        description="将概念和想法可视化为图片",
        temperature=0.6,
        default_model="flux",
        default_size="1024x1024",
        default_quality="standard"
    )
    
    # 自定义系统提示
    agent.system_prompt = """你是概念可视化和信息图表专家。

职责：
- 将抽象概念转化为视觉图像
- 创建信息图表和示意图
- 可视化数据和流程
- 制作教育和演示材料

可视化类型：
- 流程图和架构图
- 概念图和思维导图
- 数据可视化图表
- 科学和技术插图
- 教育和培训材料

设计原则：
1. 清晰易懂
2. 视觉层次分明
3. 色彩协调
4. 重点突出
5. 专业美观

目标：让复杂的概念一目了然！"""
    
    return agent


# =============================================================================
# 辅助函数：下载和预览图片
# =============================================================================

def download_and_preview_image(image_url: str, save_dir: str = "generated_images", 
                               filename: Optional[str] = None, auto_open: bool = True) -> Optional[str]:
    """
    下载图片并在Windows上打开预览
    
    Args:
        image_url: 图片URL
        save_dir: 保存目录（默认: generated_images）
        filename: 文件名（可选，自动生成）
        auto_open: 是否自动打开预览（默认: True）
        
    Returns:
        本地文件路径
    """
    try:
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        if not filename:
            import time
            timestamp = int(time.time())
            filename = f"image_{timestamp}.png"
        
        file_path = save_path / filename
        
        # 下载图片
        print(f"正在下载图片: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # 保存图片
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"图片已保存到: {file_path.absolute()}")
        
        # 在Windows上自动打开预览
        if auto_open and sys.platform == 'win32':
            try:
                print("正在打开图片预览...")
                os.startfile(str(file_path.absolute()))
            except Exception as e:
                print(f"自动打开失败: {e}，请手动打开文件")
        elif auto_open:
            # macOS 和 Linux
            try:
                import subprocess
                if sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', str(file_path)])
                else:  # Linux
                    subprocess.call(['xdg-open', str(file_path)])
            except Exception as e:
                print(f"自动打开失败: {e}，请手动打开文件")
        
        return str(file_path.absolute())
    
    except Exception as e:
        print(f"下载或预览图片失败: {e}")
        return None


def extract_and_preview_images(result_text: str, save_dir: str = "generated_images",
                              auto_open: bool = True) -> List[str]:
    """
    从结果文本中提取图片URL并下载预览
    
    Args:
        result_text: 包含图片URL的结果文本
        save_dir: 保存目录
        auto_open: 是否自动打开预览
        
    Returns:
        下载的本地文件路径列表
    """
    import re
    import json
    
    print(f"\n正在分析结果文本...")
    print(f"结果长度: {len(result_text)} 字符")
    
    # 首先尝试解析JSON格式（ReAct agent可能返回JSON）
    image_urls = []
    
    # 方法1: 尝试从JSON中提取
    try:
        # 查找类似 {"generate_image_flux", "arguments": {...}} 的模式
        json_pattern = r'\{[^{}]*"arguments"[^{}]*\{[^{}]*"prompt"[^{}]*\}[^{}]*\}'
        json_matches = re.findall(json_pattern, result_text, re.DOTALL)
        
        if json_matches:
            print(f"找到 {len(json_matches)} 个JSON工具调用")
            # 这种情况下，工具还未执行，需要提示用户
            print("⚠ 检测到工具调用JSON，但工具未执行。这可能是因为：")
            print("  1. API密钥无效或未设置")
            print("  2. 网络连接问题")
            print("  3. 模型配置错误")
            print("\n原始输出片段:")
            print(json_matches[0][:500] if json_matches else "")
            return []
    except Exception as e:
        print(f"JSON解析尝试失败: {e}")
    
    # 方法2: 直接提取URL
    url_pattern = r'https?://[^\s<>"\']+'
    all_urls = re.findall(url_pattern, result_text)
    
    print(f"找到 {len(all_urls)} 个URL")
    
    # 过滤出可能是图片的URL
    image_urls = [url for url in all_urls if any(
        keyword in url.lower() 
        for keyword in ['image', 'png', 'jpg', 'jpeg', 'webp', 'cdn', 'siliconflow', 'amazonaws']
    )]
    
    if not image_urls and all_urls:
        # 如果没有匹配到典型图片URL，但有URL，打印出来供调试
        print(f"发现URL但未识别为图片URL:")
        for url in all_urls[:5]:  # 只显示前5个
            print(f"  - {url[:100]}...")
    
    if not image_urls:
        print("❌ 未找到图片URL")
        print("\n可能的原因：")
        print("  1. 图片生成失败（检查API密钥和网络）")
        print("  2. 返回格式不正确")
        print("\n结果文本预览:")
        print(result_text[:500] if len(result_text) > 500 else result_text)
        return []
    
    print(f"\n✅ 找到 {len(image_urls)} 个图片URL")
    downloaded_files = []
    
    for i, url in enumerate(image_urls, 1):
        print(f"\n处理图片 {i}/{len(image_urls)}")
        print(f"URL: {url[:100]}...")
        filename = f"image_{i}_{int(__import__('time').time())}.png"
        local_path = download_and_preview_image(
            url, 
            save_dir=save_dir, 
            filename=filename,
            auto_open=auto_open if i == 1 else False  # 只自动打开第一张
        )
        if local_path:
            downloaded_files.append(local_path)
            print(f"✓ 图片 {i} 已下载")
        else:
            print(f"✗ 图片 {i} 下载失败")
    
    return downloaded_files


# =============================================================================
# 使用示例和演示
# =============================================================================

if __name__ == "__main__":
    from multi_agent_framework import AgentFramework
    
    print("=" * 70)
    print("  图片生成智能体演示")
    print("=" * 70)
    
    # 创建框架
    framework = AgentFramework(name="ImageGenDemo")
    
    # 注册图片生成智能体
    image_agent = create_image_generator_agent()
    framework.register_agent(image_agent)
    
    # 示例 1: 基本图片生成
    print("\n" + "=" * 70)
    print("  示例 1: 生成一张未来城市的图片")
    print("=" * 70)
    
    try:
        result = framework.execute(
            task="生成一张未来城市的图片，有飞行汽车和高楼大厦，赛博朋克风格",
            workflow_type="pipeline",
            workflow_config={
                "agent_sequence": ["image_generator"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n生成结果:\n{result['result'][:1500]}...")
        
        # 显示详细调试信息
        print("\n" + "=" * 70)
        print("调试信息:")
        print(f"结果类型: {type(result['result'])}")
        print(f"结果长度: {len(result['result'])} 字符")
        
        # 检查是否有 agent_outputs
        if 'agent_outputs' in result:
            print(f"\n智能体输出数量: {len(result['agent_outputs'])}")
            for agent_name, output in result['agent_outputs'].items():
                print(f"\n智能体 '{agent_name}':")
                if isinstance(output, dict):
                    print(f"  - 键: {list(output.keys())}")
                    if 'output' in output:
                        print(f"  - 输出预览: {str(output['output'])[:200]}...")
        
        # 下载并预览图片
        print("\n" + "-" * 70)
        print("正在下载和预览图片...")
        downloaded_files = extract_and_preview_images(
            result['result'], 
            save_dir="generated_images",
            auto_open=True
        )
        if downloaded_files:
            print(f"\n✓ 成功下载 {len(downloaded_files)} 张图片")
            for f in downloaded_files:
                print(f"  - {f}")
        else:
            print("\n✗ 未能下载图片")
        
    except Exception as e:
        print(f"执行出错: {e}")
        print("提示: 请确保已设置 SILICONFLOW_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
    
    # 示例 2: 使用创意艺术家
    print("\n" + "=" * 70)
    print("  示例 2: 创意艺术创作")
    print("=" * 70)
    
    try:
        creative_agent = create_creative_artist_agent()
        framework.register_agent(creative_agent)
        
        result = framework.execute(
            task="梦幻的星空下的神秘森林",
            workflow_type="pipeline",
            workflow_config={
                "agent_sequence": ["creative_artist"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n生成结果:\n{result['result'][:1500]}...")
        
        # 下载并预览图片
        print("\n" + "-" * 70)
        print("正在下载和预览图片...")
        downloaded_files = extract_and_preview_images(
            result['result'], 
            save_dir="generated_images/artistic",
            auto_open=True
        )
        if downloaded_files:
            print(f"\n✓ 成功下载 {len(downloaded_files)} 张图片")
        
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 示例 3: 结合其他智能体
    print("\n" + "=" * 70)
    print("  示例 3: 描述优化 + 图片生成")
    print("=" * 70)
    
    try:
        from multi_agent_framework import create_writer_agent
        
        writer = create_writer_agent("description_writer")
        framework.register_agent(writer)
        
        # 先让writer生成详细描述，再生成图片
        result = framework.execute(
            task="为一个环保主题的海报生成图片描述，然后生成图片",
            workflow_type="pipeline",
            workflow_config={
                "agent_sequence": ["description_writer", "image_generator"]
            }
        )
        
        print(f"\n任务: {result['task']}")
        print(f"\n最终结果:\n{result['result'][:1500]}...")
        
        # 下载并预览图片
        print("\n" + "-" * 70)
        print("正在下载和预览图片...")
        downloaded_files = extract_and_preview_images(
            result['result'], 
            save_dir="generated_images/poster",
            auto_open=True
        )
        if downloaded_files:
            print(f"\n✓ 成功下载 {len(downloaded_files)} 张图片")
        
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("  演示完成")
    print("=" * 70)
