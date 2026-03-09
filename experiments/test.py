import torch
from transformers import AutoProcessor

def find_vla_action_range(model_id="Qwen/Qwen3-VL-8B-Instruct"):
    print(f"正在分析模型: {model_id} ...")
    
    try:
        # 1. 加载处理器和分词器
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        
        # 2. 检索包含关键标识的 Token
        # 注意：Nora 这种 VLA 模型通常使用 <robot_action_数字> 这种格式
        action_map = {}
        for token_str, token_id in vocab.items():
            if "robot_action_" in token_str:
                try:
                    # 提取编号，确保排序正确
                    num = int(token_str.split('_')[-1].replace('>', ''))
                    action_map[num] = token_id
                except ValueError:
                    continue

        if not action_map:
            print("\n[错误] 未在词表中找到 'robot_action_' 系列 Token。")
            print("提示：如果这是原生 Qwen3 而不是 Nora-Qwen3，它可能还没有内置动作 Token。")
            print("以下是词表末尾的 10 个 Token，请核对是否为新增动作 Token：")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
            for i in range(10):
                print(f"  ID: {sorted_vocab[i][1]} -> '{sorted_vocab[i][0]}'")
            return

        # 3. 计算结果
        sorted_ids = sorted(action_map.values())
        min_id = sorted_ids[0]
        max_id = sorted_ids[-1]
        total_count = len(sorted_ids)

        # 4. 验证连续性
        is_continuous = (max_id - min_id + 1) == total_count

        print("\n" + "="*50)
        print("🚀 Qwen3-VLA Token 检测成功！")
        print("-"*50)
        print(f"起始 ID (min_id): {min_id}")
        print(f"结束 ID (max_id): {max_id}")
        print(f"Token 总数量:   {total_count}")
        print(f"是否连续空间:   {'✅ 是' if is_continuous else '❌ 否（请检查词表是否被污染）'}")
        print("="*50)

        # 5. 直接生成可用的代码片段
        print("\n请将以下代码复制并替换到你的训练脚本中：\n")
        print(f"# --- Qwen3 Action Token Config ---")
        print(f"action_token_min = {min_id}")
        print(f"action_token_max = {max_id}")
        print(f"# ----------------------------------")

    except Exception as e:
        print(f"\n[运行异常]: {e}")
        print("请确保已安装最新版 transformers: pip install --upgrade transformers")

if __name__ == "__main__":
    model_path = "declare-lab/nora" # 示例
    find_vla_action_range(model_path)