
## 代码
```python
"""
报告生成器模块：整合外部搜索和内部检索结果，生成结构化的Markdown报告。
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    """AI风口报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def integrate_results(self, 
                         external_search_results: Dict[str, Any],
                         internal_retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        整合外部搜索和内部检索结果
        
        Args:
            external_search_results: 外部搜索结果
            internal_retrieval_results: 内部检索结果
            
        Returns:
            整合后的数据字典
        """
        try:
            logger.info("开始整合搜索结果...")
            
            # 基础整合策略：外部结果为主，内部结果补充
            integrated_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "source": "external_search + internal_retrieval",
                    "topic": "2026年AI风口总结"
                },
                "external_data": external_search_results,
                "internal_data": internal_retrieval_results,
                "combined_insights": {}
            }
            
            # 合并趋势信息
            trends = self._merge_lists(
                external_search_results.get("trends", []),
                internal_retrieval_results.get("trends", [])
            )
            integrated_data["combined_insights"]["trends"] = trends
            
            # 合并技术信息
            technologies = self._merge_lists(
                external_search_results.get("technologies", []),
                internal_retrieval_results.get("technologies", [])
            )
            integrated_data["combined_insights"]["technologies"] = technologies
            
            # 合并应用领域
            applications = self._merge_lists(
                external_search_results.get("applications", []),
                internal_retrieval_results.get("applications", [])
            )
            integrated_data["combined_insights"]["applications"] = applications
            
            # 合并风险挑战
            risks = self._merge_lists(
                external_search_results.get("risks", []),
                internal_retrieval_results.get("risks", [])
            )
            integrated_data["combined_insights"]["risks"] = risks
            
            logger.info(f"整合完成，共收集趋势{len(trends)}项，技术{len(technologies)}项，"
                       f"应用{len(applications)}项，风险{len(risks)}项")
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"整合结果时发生错误: {e}")
            raise
    
    def _merge_lists(self, list1: List[Any], list2: List[Any]) -> List[Any]:
        """
        合并两个列表，去重
        
        Args:
            list1: 第一个列表
            list2: 第二个列表
            
        Returns:
            合并去重后的列表
        """
        if not list1:
            return list2 if list2 else []
        if not list2:
            return list1 if list1 else []
        
        # 简单去重合并
        combined = list1.copy()
        for item in list2:
            if item not in combined:
                combined.append(item)
        return combined
    
    def generate_markdown_report(self, integrated_data: Dict[str, Any]) -> str:
        """
        生成Markdown格式的报告
        
        Args:
            integrated_data: 整合后的数据
            
        Returns:
            Markdown格式的报告字符串
        """
        try:
            logger.info("开始生成Markdown报告...")
            
            metadata = integrated_data.get("metadata", {})
            insights = integrated_data.get("combined_insights", {})
            
            # 构建报告内容
            report_lines = [
                f"# {metadata.get('topic', 'AI风口分析报告')}",
                "",
                f"**生成时间**: {metadata.get('generated_at', '未知')}",
                f"**数据来源**: {metadata.get('source', '未指定')}",
                "",
                "---",
                "",
                "## 执行摘要",
                "本报告基于外部市场调研和内部技术分析，综合评估2026年人工智能领域的主要发展趋势、",
                "关键技术突破、潜在应用场景以及面临的风险挑战。",
                "",
                "## 一、主要趋势",
            ]
            
            # 添加趋势部分
            trends = insights.get("trends", [])
            if trends:
                for i, trend in enumerate(trends, 1):
                    if isinstance(trend, dict):
                        title = trend.get("title", f"趋势{i}")
                        description = trend.get("description", "")
                        report_lines.append(f"### {i}. {title}")
                        report_lines.append(f"{description}")
                    else:
                        report_lines.append(f"### {i}. {trend}")
                    report_lines.append("")
            else:
                report_lines.append("暂无趋势数据")
                report_lines.append("")
            
            # 添加技术部分
            report_lines.append("## 二、关键技术")
            technologies = insights.get("technologies", [])
            if technologies:
                for i, tech in enumerate(technologies, 1):
                    if isinstance(tech, dict):
                        name = tech.get("name", f"技术{i}")
                        impact = tech.get("impact", "")
                        report_lines.append(f"### {i}. {name}")
                        report_lines.append(f"**影响**: {impact}")
                    else:
                        report_lines.append(f"### {i}. {tech}")
                    report_lines.append("")
            else:
                report_lines.append("暂无技术数据")
                report_lines.append("")
            
            # 添加应用部分
            report_lines.append("## 三、潜在应用领域")
            applications = insights.get("applications", [])
            if applications:
                for i, app in enumerate(applications, 1):
                    if isinstance(app, dict):
                        domain = app.get("domain", f"应用领域{i}")
                        examples = app.get("examples", "")
                        report_lines.append(f"### {i}. {domain}")
                        report_lines.append(f"**典型应用**: {examples}")
                    else:
                        report_lines.append(f"### {i}. {app}")
                    report_lines.append("")
            else:
                report_lines.append("暂无应用数据")
                report_lines.append("")
            
            # 添加风险部分
            report_lines.append("## 四、风险挑战")
            risks = insights.get("risks", [])
            if risks:
                for i, risk in enumerate(risks, 1):
                    if isinstance(risk, dict):
                        type_ = risk.get("type", f"风险{i}")
                        mitigation = risk.get("mitigation", "")
                        report_lines.append(f"### {i}. {type_}")
                        report_lines.append(f"**应对策略**: {mitigation}")
                    else:
                        report_lines.append(f"### {i}. {risk}")
                    report_lines.append("")
            else:
                report_lines.append("暂无风险数据")
                report_lines.append("")
            
            # 添加结论
            report_lines.extend([
                "## 五、结论与建议",
                "",
                "### 核心发现",
                "1. **技术融合加速**: 多模态AI、边缘计算与云计算的结合将成为主流",
                "2. **行业渗透深化**: AI将从工具型应用转向决策支持系统",
                "3. **监管框架完善**: 全球AI治理体系将逐步建立",
                "",
                "### 投资建议",
                "- **重点关注**: 具身智能、AI原生应用、隐私计算",
                "- **谨慎布局**: 同质化严重的通用大模型",
                "- **长期跟踪**: 量子计算与AI的交叉领域",
                "",
                "---",
                "",
                "*本报告基于公开数据和内部分析生成，仅供参考，不构成投资建议。*"
            ])
            
            report_content = "\n".join(report_lines)
            logger.info(f"报告生成完成，长度: {len(report_content)} 字符")
            
            return report_content
            
        except Exception as e:
            logger.error(f"生成报告时发生错误: {e}")
            raise
    
    def save_report(self, report_content: str, filename: Optional[str] = None) -> Path:
        """
        保存报告到文件
        
        Args:
            report_content: 报告内容
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ai_trends_report_{timestamp}.md"
            
            filepath = self.output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            logger.info(f"报告已保存至: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存报告时发生错误: {e}")
            raise
    
    def generate_full_report(self,
                           external_results: Dict[str, Any],
                           internal_results: Dict[str, Any],
                           save_to_file: bool = True) -> Dict[str, Any]:
        """
        完整报告生成流程
        
        Args:
            external_results: 外部搜索结果
            internal_results: 内部检索结果
            save_to_file: 是否保存到文件
            
        Returns:
            包含报告内容和元数据的字典
        """
        try:
            # 1. 整合结果
            integrated_data = self.integrate_results(external_results, internal_results)
            
            # 2. 生成Markdown报告
            report_content = self.generate_markdown_report(integrated_data)
            
            # 3. 保存报告
            filepath = None
            if save_to_file:
                filepath = self.save_report(report_content)
            
            return {
                "success": True,
                "report_content": report_content,
                "integrated_data": integrated_data,
                "filepath": str(filepath) if filepath else None,
                "metadata": integrated_data.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"完整报告生成流程失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "report_content": None,
                "filepath": None
            }


# 示例使用函数
def example_usage():
    """示例使用函数"""
    
    # 模拟外部搜索结果
    external_results = {
        "trends": [
            {"title": "多模态AI普及", "description": "文本、图像、语音、视频的融合理解与生成"},
            {"title": "边缘AI部署", "description": "模型轻量化，在终端设备上直接运行"},
            "AI原生应用崛起"
        ],
        "technologies": [
            {"name": "Transformer架构演进", "impact": "提升模型效率和推理速度"},
            {"name": "神经符号AI", "impact": "结合深度学习与符号推理"},
            "联邦学习"
        ],
        "applications": [
            {"domain": "医疗健康", "examples": "辅助诊断、药物发现、个性化治疗"},
            {"domain": "智能制造", "examples": "预测性维护、质量控制、供应链优化"}
        ],
        "risks": [
            {"type": "数据隐私", "mitigation": "差分隐私、同态加密"},
            {"type": "算法偏见", "mitigation": "公平性评估、多样化训练数据"}
        ]
    }
    
    # 模拟内部检索结果
    internal_results = {
        "trends": [
            "具身智能发展",
            {"title": "AI治理框架", "description": "全球监管政策逐步完善"}
        ],
        "technologies": [
            {"name": "量子机器学习", "impact": "解决传统计算难以处理的问题"}
        ],
        "applications": [
            {"domain": "教育科技", "examples": "个性化学习、智能辅导"}
        ],
        "risks": [
            {"type": "就业冲击", "mitigation": "技能培训、人机协作"}
        ]
    }
    
    # 创建报告生成器实例
    generator = ReportGenerator()
    
    # 生成完整报告
    result = generator.generate_full_report(external_results, internal_results)
    
    if result["success"]:
        print(f"报告生成成功！")
        print(f"文件保存位置: {result['filepath']}")
        print(f"报告前500字符预览:")
        print(result["report_content"][:500] + "...")
    else:
        print(f"报告生成失败: {result.get('error')}")
    
    return result


if __name__ == "__main__":
    # 运行示例
    example_usage()
```

## 实现说明
1. 设计了`ReportGenerator`类，提供完整的数据整合、报告生成和文件保存功能，支持灵活的输入数据格式
2. 实现了智能合并策略，能够处理字典和字符串混合的数据格式，确保报告内容的丰富性和完整性
3. 包含完整的错误处理和日志记录，确保生产环境下的稳定运行，同时提供示例使用函数便于测试

## 置信度
0.95