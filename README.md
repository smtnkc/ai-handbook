# ai-notebook

## Terms

* AGI: Artificial General Intelligence 
* MMLU: Massive Multitask Language Understanding
* RLHF: Reinforcement learning from human feedback
* [Retrieval-Augmented Generation (RAG)](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
* [Hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
* [Catastrophic interference (forgetting)](https://en.wikipedia.org/wiki/Catastrophic_interference)
* [Stochastic Parrot](https://en.wikipedia.org/wiki/Stochastic_parrot)
* Modalities: Text, Image, Video, Audio
* 5-shot: Few-shot training with five samples
* CoT@32: Chain of thought prompting with 32 samples
* MoE: Mixture of experts
* [Data contamination vs Task contamination](https://cobusgreyling.medium.com/llm-performance-over-time-task-contamination-a69fde87dd86)
* Downstream tasks
* Pre-training vs intermediate training (Domain Adaptation)
* Transfer learning
* Fine-tuning
* BLEU score: Metric for translation
* ROUGE score: Metric for text summarization
* Perplexity: Metric for MLM

## Tools
* [ChatGPT](https://chat.openai.com/)
* [Gemini](https://gemini.google.com/app) (Successor of BARD)
* [AlphaCode 2](https://deepmind.google/discover/blog/competitive-programming-with-alphacode/) (Programming tool powered by Gemini)
* [Midjourney](https://www.midjourney.com/home) (Image generator) 
* [Magnific.ai](https://magnific.ai/) (Image upscaler)
* [Lisa AI](https://lisaai.app/) (Artistic image generator)

## AIOps
* [Vercel](https://vercel.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Chroma](https://www.trychroma.com/)
* [Open-WebUI](https://docs.openwebui.com/)
* [Raycast](https://www.raycast.com/)

## Libraries & Frameworks
* [LoRA](https://huggingface.co/docs/diffusers/en/training/lora)
* [DSPy](https://dspy-docs.vercel.app/)
* [Ollama](https://ollama.com/)
* [LM Studio](https://lmstudio.ai/)
* [Cody](https://meetcody.ai/)
* [Localpilot](https://github.com/danielgross/localpilot)
* [Rawdog](https://github.com/AbanteAI/rawdog)
* [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
* [DreamBooth](https://dreambooth.github.io/)

## Transformer Models

* Encoder-only :arrow_right: Classification or Extractive Summarization
  * [BERT](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
  * [RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta) 
* Decoder-only :arrow_right: Prompty-based (autoregressive) text generation
  * [GPT](https://huggingface.co/docs/transformers/en/model_doc/openai-gpt)
  * [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)
  * [Mixtral](https://huggingface.co/docs/transformers/en/model_doc/mixtral)
  * [LLaMa](https://huggingface.co/docs/transformers/main/en/model_doc/llama)
  * [PaLM](https://ai.google/discover/palm2/)
* Encoder + Decoder :arrow_right: Translation or Abstractive Summarization
  * [BART](https://huggingface.co/docs/transformers/en/model_doc/bart)
  * [T5](https://huggingface.co/docs/transformers/en/model_doc/t5)
  * [PEGASUS](https://huggingface.co/docs/transformers/en/model_doc/pegasus) 

<img src="https://raw.githubusercontent.com/smtnkc/ai-notebook/main/transformer_models.png" alt="transformer_models" width="720"/>

## LLMs Evaluation

<img src="https://raw.githubusercontent.com/smtnkc/ai-notebook/main/llms_evaluation.png" alt="llms_evaluation" width="720"/>

## Overused Words

> holistic, multifaceted, nuanced, comprehensive, meticulously, landscape, realm, delve, tapestry, seamlessly, unleash, embark, unwavering, unraveling, testament, leveraging, fostering, cultivate, it’s important to…
 
## References
* [Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)
* [Encoder-Only vs Decoder-Only vs Encoder-Decoder Transformer](https://vaclavkosar.com/ml/Encoder-only-Decoder-only-vs-Encoder-Decoder-Transfomer)
* [BART Text Summarization vs. GPT-3 vs. BERT: An In-Depth Comparison](https://www.width.ai/post/bart-text-summarization)
* [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
* [Domain Adaptation with HuggingFace MLM](https://www.kaggle.com/code/hinepo/domain-adaptation-with-mlm)
* [Training BERT from Scratch on Your Custom Domain Data](https://medium.com/@shankar.arunp/training-bert-from-scratch-on-your-custom-domain-data-a-step-by-step-guide-with-amazon-25fcbee4316a)
* [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
