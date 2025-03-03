# core/llm_agent.py
import os
import torch
import logging
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from omegaconf import DictConfig
from tqdm import tqdm

class LLMReasoningEngine:
    def __init__(self, config: DictConfig):
        self.config = config.llm
        self.device = torch.device(config.system.device)
        self.tokenizer, self.model = self._load_model()
        self.generation_config = self._get_generation_config()
        
        # 初始化提示模板
        self.templates = {
            'recommend_reason': self._load_template('recommend_reason'),
            'semantic_score': self._load_template('semantic_score')
        }
        
        logging.info(f"LLM引擎初始化完成，设备: {self.device}")

    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载预训练模型和分词器"""
        model_path = self.config.model_path
        precision = self.config.get('precision', 'fp16')
        
        # 量化配置
        load_kwargs = {'device_map': 'auto'} if self.config.quantized else {}
        if self.config.quantized:
            load_kwargs.update({
                'load_in_4bit': True,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': "nf4",
                'bnb_4bit_compute_dtype': torch.bfloat16
            })
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **load_kwargs
        )
        
        if not self.config.quantized and precision == 'fp16':
            model = model.half()
            
        return tokenizer, model.to(self.device)

    def _get_generation_config(self) -> GenerationConfig:
        """配置生成参数"""
        return GenerationConfig(
            max_new_tokens=self.config.generation.max_new_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            repetition_penalty=self.config.generation.repetition_penalty,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def _load_template(self, template_name: str) -> str:
        """加载提示模板"""
        template_path = os.path.join(
            self.config.template_dir,
            f"{template_name}.txt"
        )
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def generate_reasons(self, candidates: List[Dict], user_profile: Dict) -> List[Dict]:
        """批量生成推荐理由"""
        processed = []
        for candidate in tqdm(candidates, desc="生成推荐理由"):
            try:
                prompt = self._build_recommend_prompt(candidate, user_profile)
                reason = self._generate_text(prompt)
                processed.append({
                    **candidate,
                    'reason': reason,
                    'semantic_score': self._score_semantic(reason, user_profile)
                })
            except Exception as e:
                logging.error(f"生成失败: {str(e)}")
                processed.append({**candidate, 'reason': "推荐理由生成中...", 'semantic_score': 0.5})
        return processed

    def _build_recommend_prompt(self, candidate: Dict, user: Dict) -> str:
        """构建推荐理由生成提示"""
        return self.templates['recommend_reason'].format(
            user_history=", ".join(user['history'][-3:]),
            item_title=candidate['title'],
            item_abstract=candidate['abstract'],
            user_interests=", ".join(user['interests'])
        )

    def _score_semantic(self, text: str, user: Dict) -> float:
        """计算文本语义相关性得分"""
        prompt = self.templates['semantic_score'].format(
            text=text,
            user_interests=", ".join(user['interests'])
        )
        score_text = self._generate_text(prompt, max_tokens=10)
        try:
            return float(score_text.strip().split())
        except:
            return 0.5

    def _generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """执行文本生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 动态调整生成长度
        gen_config = self.generation_config
        if max_tokens:
            gen_config = gen_config.update(max_new_tokens=max_tokens)
        
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config
        )
        return self.tokenizer.decode(outputs[len(inputs):], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], batch_size: int = 4) -> List[str]:
        """批量生成加速"""
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="批量生成"):
            batch = prompts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
            
            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            results.extend(decoded)
        return results

if __name__ == "__main__":
    from configs import load_config
    
    # 示例配置
    config = load_config(["llm=glm"])
    llm_engine = LLMReasoningEngine(config)
    
    # 模拟输入
    test_candidate = {
        'item_id': 'N123',
        'title': '人工智能新突破',
        'abstract': '研究人员开发出新型神经网络架构...',
        'features': np.random.rand(2816)
    }
    test_user = {
        'user_id': 'U456',
        'history': ['N789', 'N101'],
        'interests': ['机器学习', '算法优化']
    }
    
    # 生成推荐理由
    result = llm_engine.generate_reasons([test_candidate], test_user)
    print(f"\n生成结果示例:\n{result['reason']}\n语义得分: {result['semantic_score']:.2f}")