"""
图片生成智能体调试脚本

用于诊断图片生成问题
"""

import os
import sys
from agents.image_generation_agent import (
    create_image_generator_agent,
    generate_image_flux,
    extract_and_preview_images
)

def test_api_connection():
    """测试API连接"""
    print("=" * 70)
    print("测试 1: API 连接")
    print("=" * 70)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    
    print(f"API Key: {api_key[:20]}...{api_key[-10:] if len(api_key) > 30 else ''}")
    print(f"Base URL: {base_url}")
    print(f"API Key 长度: {len(api_key)}")
    
    if not api_key or api_key == "":
        print("❌ 错误: API Key 为空")
        return False
    
    print("✓ API Key 已设置")
    return True


def test_direct_image_generation():
    """直接调用图片生成工具"""
    print("\n" + "=" * 70)
    print("测试 2: 直接调用图片生成工具")
    print("=" * 70)
    
    try:
        print("正在生成测试图片...")
        result = generate_image_flux.invoke({
            "prompt": "a simple red apple on white background",
            "size": "1024x1024",
            "num_images": 1
        })
        
        print(f"\n结果类型: {type(result)}")
        print(f"结果长度: {len(result)} 字符")
        print(f"\n结果内容:\n{result}")
        
        # 尝试提取URL
        print("\n" + "-" * 70)
        print("尝试下载图片...")
        files = extract_and_preview_images(result, save_dir="test_images", auto_open=True)
        
        if files:
            print(f"\n✓ 成功! 生成了 {len(files)} 张图片")
            return True
        else:
            print("\n✗ 失败: 未能下载图片")
            return False
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_execution():
    """测试智能体执行"""
    print("\n" + "=" * 70)
    print("测试 3: 智能体执行")
    print("=" * 70)
    
    try:
        from multi_agent_framework import AgentFramework
        
        framework = AgentFramework(name="TestFramework")
        agent = create_image_generator_agent()
        framework.register_agent(agent)
        
        print("执行简单任务...")
        result = framework.execute(
            task="a blue sky with white clouds",
            workflow_type="pipeline",
            workflow_config={"agent_sequence": ["image_generator"]}
        )
        
        print(f"\n任务: {result['task']}")
        print(f"结果类型: {type(result['result'])}")
        print(f"结果长度: {len(result['result'])} 字符")
        print(f"\n结果预览:\n{result['result'][:1000]}")
        
        # 检查 agent_outputs
        if 'agent_outputs' in result:
            print(f"\n智能体输出:")
            for name, output in result['agent_outputs'].items():
                print(f"\n  {name}:")
                if isinstance(output, dict):
                    for key, value in output.items():
                        if key != 'full_trace':
                            print(f"    {key}: {str(value)[:200]}")
        
        # 尝试下载
        print("\n" + "-" * 70)
        print("尝试下载图片...")
        files = extract_and_preview_images(result['result'], save_dir="test_images", auto_open=True)
        
        if files:
            print(f"\n✓ 成功!")
            return True
        else:
            print("\n✗ 失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("  图片生成智能体调试工具")
    print("=" * 70)
    
    results = {}
    
    # 测试 1: API 连接
    results['api_connection'] = test_api_connection()
    
    # 测试 2: 直接调用
    if results['api_connection']:
        results['direct_call'] = test_direct_image_generation()
    else:
        print("\n⚠ 跳过测试 2 (API 未配置)")
        results['direct_call'] = False
    
    # 测试 3: 智能体执行
    if results['api_connection']:
        results['agent_execution'] = test_agent_execution()
    else:
        print("\n⚠ 跳过测试 3 (API 未配置)")
        results['agent_execution'] = False
    
    # 总结
    print("\n" + "=" * 70)
    print("  测试结果总结")
    print("=" * 70)
    print(f"API 连接: {'✓ 通过' if results['api_connection'] else '✗ 失败'}")
    print(f"直接调用: {'✓ 通过' if results.get('direct_call', False) else '✗ 失败'}")
    print(f"智能体执行: {'✓ 通过' if results.get('agent_execution', False) else '✗ 失败'}")
    
    if all(results.values()):
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠ 部分测试失败，请检查上述错误信息")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
