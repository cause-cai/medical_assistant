import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# 中医场景提示词（辨证论治、四诊合参、体质与养生等）
TCM_PROMPTS = {
    "syndrome_differentiation": "你是一名资深中医师，请依据患者的症状、舌象、脉象，进行辨证论治，明确证型，并给出治则治法。",
    "herbal_prescription": "你是一名资深中医师，请根据辨证结果提出合理的中药方剂建议（含君臣佐使），同时给出加减思路与注意事项。",
    "acupoint_tuina": "你是针灸推拿科医生，请根据病机给出针灸/艾灸/推拿取穴建议与操作要点，标注禁忌与注意事项。",
    "diet_lifestyle": "你是中医治未病专家，请给出饮食调理、作息起居、情志疏导、四时养生等生活方式建议。",
    "pediatric_tcm": "你是中医儿科医生，请结合儿童生理病理特点提供中医调护与安全用药建议。",
    "gynecology_tcm": "你是中医妇科医生，请结合月经、带下、孕产相关问题，给出中医辨证与调治建议。",
    "geriatric_tcm": "你是中医老年病专家，请考虑老年人脏腑气血亏虚特征，提供温和稳妥的调理方案。",
    "subhealth": "你是中医治未病与亚健康管理专家，请根据表现提出体质辨识与调理方案。",
    "seasonal_regimen": "你是中医养生家，请依据二十四节气与时令特点，给出顺应节气的起居、饮食与运动建议。",
    "emergency_tcm": "你是中医急症医生，请评估紧急程度，给出可行的中医干预建议，并强调何时必须立即就医。"
}

# 可选场景（编号 -> 名称）
TCM_SCENARIOS = {
    "1": "辨证论治",
    "2": "中药方剂",
    "3": "针灸推拿",
    "4": "饮食与起居",
    "5": "中医儿科",
    "6": "中医妇科",
    "7": "中医老年病",
    "8": "体质与亚健康",
    "9": "时令与节气养生",
    "10": "急症评估(中医视角)"
}

# 示例问题
TCM_SAMPLE_QUESTIONS = {
    "syndrome_differentiation": [
        "反复乏力、食欲差、面色少华，舌淡苔薄白，脉细弱，应如何辨证？",
        "头痛胀痛、胸闷易怒、两胁作痛，舌红苔薄黄，脉弦，应如何论治？"
    ],
    "herbal_prescription": [
        "春季过敏性鼻炎反复发作，推荐思路与方药示例？",
        "胃脘隐痛喜温喜按、纳差，大便溏薄，方药如何加减？"
    ],
    "acupoint_tuina": [
        "颈肩僵硬疼痛，推荐取穴与按揉顺序？",
        "失眠多梦，针灸取穴及艾灸注意事项？"
    ],
    "diet_lifestyle": [
        "脾胃虚弱体质，日常饮食与作息建议？",
        "熬夜导致的口干口苦应如何调理？"
    ],
    "subhealth": [
        "容易怕冷、手脚冰凉、精神不振，可能是哪类体质？",
        "久坐少动导致腰背酸痛，如何通过体质调护？"
    ]
}


class TCMMedicalAssistant:
    def __init__(self, checkpoint_path="./output/Qwen3-0.6B/checkpoint-900"):
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def _select_device_and_dtype(self):
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32

    def load_model(self):
        print("正在加载中医医疗助手模型...")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")

        # 为权重卸载准备目录，减小内存占用
        offload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offload")
        os.makedirs(offload_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            dtype=self.dtype,  # 替换已弃用的 torch_dtype
            local_files_only=True,
            device_map="auto",  # 根据设备自动映射，降低峰值内存
            low_cpu_mem_usage=True,  # 低CPU内存加载
            offload_folder=offload_dir,
            attn_implementation="eager",  # 兼容性更好的注意力实现
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成！使用设备: {self.device}")

    def predict(self, messages, max_new_tokens=512):
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response

    def ask_question(self, question, scenario_type="syndrome_differentiation", max_tokens=512):
        if scenario_type not in TCM_PROMPTS:
            scenario_type = "syndrome_differentiation"

        creativity_note = (
            "在给出结论时兼顾中医整体观与个体化，"
            "可提出一到两个具有创造性的调理思路（如节气加减、体质化加味、情志疏导配合），"
            "务必标注禁忌与需线下就诊的情形。"
        )

        messages = [
            {"role": "system", "content": f"{TCM_PROMPTS[scenario_type]} {creativity_note}"},
            {"role": "user", "content": question},
        ]

        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type,
            "question": question,
            "response": None,
        })

        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history[-1]["response"] = response
        return response

    def show_scenarios(self):
        print("\n🏥 中医医疗助手 - 可用场景:")
        print("=" * 50)
        for key, value in TCM_SCENARIOS.items():
            print(f"{key:2}. {value}")
        print("=" * 50)

    def show_sample_questions(self, scenario_type_key):
        if scenario_type_key in TCM_SAMPLE_QUESTIONS:
            print(f"\n📋 示例问题:")
            print("-" * 40)
            for i, q in enumerate(TCM_SAMPLE_QUESTIONS[scenario_type_key], 1):
                print(f"{i}. {q}")
            print("-" * 40)

    def interactive_mode(self):
        print("\n🤖 中医医疗助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        while True:
            try:
                self.show_scenarios()
                scenario_choice = input("\n请选择中医场景 (1-10): ").strip()
                if scenario_choice == 'quit':
                    break
                elif scenario_choice == 'help':
                    self.show_help()
                    continue
                elif scenario_choice not in TCM_SCENARIOS:
                    print("❌ 无效选择，请重新输入")
                    continue

                scenario_idx = int(scenario_choice) - 1
                scenario_type = list(TCM_PROMPTS.keys())[scenario_idx]

                self.show_sample_questions(scenario_type)

                question = input(f"\n请输入您的{TCM_SCENARIOS[scenario_choice]}问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue

                print("\n🔄 正在分析您的问题...")
                start_time = time.time()
                response = self.ask_question(question, scenario_type)
                elapsed_time = time.time() - start_time

                print(f"\n💡 中医助手回答 (耗时: {elapsed_time:.2f}秒):")
                print("=" * 60)
                print(response)
                print("=" * 60)

                continue_choice = input("\n是否继续咨询？(y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '否']:
                    break
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用中医医疗助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue

    def show_help(self):
        print("\n📖 中医医疗助手使用帮助:")
        print("=" * 50)
        print("1. 选择中医场景 (1-10)")
        print("2. 输入您的问题（可含舌象、脉象、症状、体质等）")
        print("3. 获得辨证、方药/针灸、与生活方式建议")
        print("\n💡 提示:")
        print("- 本助手仅提供参考建议，不能替代线下专业诊疗")
        print("- 急重症或持续加重请立即就医")
        print("- 输入 'quit' 退出程序")
        print("=" * 50)

    def save_conversation(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tcm_conversation_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"💾 对话历史已保存到: {filename}")

    def batch_questions(self, questions_file):
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print(f"📝 开始批量处理 {len(questions)} 个问题...")
            results = []
            for i, q in enumerate(questions, 1):
                print(f"\n处理第 {i}/{len(questions)} 个问题...")
                response = self.ask_question(
                    q.get('question', ''),
                    q.get('scenario', 'syndrome_differentiation'),
                    q.get('max_tokens', 512)
                )
                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'syndrome_differentiation'),
                    "response": response
                })
            output_file = f"tcm_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 批量处理完成！结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="中医医疗助手 - 基于Qwen3-0.6B的智能中医咨询系统")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="./output/Qwen3-0.6B/checkpoint-900",
                        help="模型检查点路径")
    parser.add_argument("--question", "-q", type=str,
                        help="直接询问问题（需要配合 --scenario 使用）")
    parser.add_argument("--scenario", "-s", type=str,
                        default="syndrome_differentiation",
                        choices=list(TCM_PROMPTS.keys()),
                        help="中医场景类型")
    parser.add_argument("--max-tokens", "-m", type=int,
                        default=512,
                        help="最大生成token数")
    parser.add_argument("--batch", "-b", type=str,
                        help="批量处理问题文件（JSON格式）")
    parser.add_argument("--save-history", action="store_true",
                        help="保存对话历史")

    args = parser.parse_args()

    assistant = TCMMedicalAssistant(args.checkpoint)
    assistant.load_model()

    if args.batch:
        assistant.batch_questions(args.batch)
    elif args.question:
        print(f"🤖 中医医疗助手回答:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        assistant.interactive_mode()

    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
