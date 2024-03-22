# AI Handbook

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
* [AI Mirror Test](https://x.com/joshwhiton/status/1770870738863415500)
* [AI Agentic Workflows](https://twitter.com/AndrewYNg/status/1770897666702233815)
* [Foundation models](https://aws.amazon.com/tr/what-is/foundation-models)
* Downstream tasks
* Pre-training vs intermediate training (Domain Adaptation)
* Transfer learning
* Fine-tuning
* BLEU score: Metric for translation
* ROUGE score: Metric for text summarization
* Perplexity: Metric for MLM

## Tools, Frameworks, Libraries

#### AI assistants
* [ChatGPT](https://chat.openai.com/) (A chatbot developed by OpenAI)
* [Gemini](https://gemini.google.com/app) (Successor of BARD)
* [AlphaCode 2](https://deepmind.google/discover/blog/competitive-programming-with-alphacode/) (Programming tool powered by Gemini)
* [SciSpace](https://typeset.io/) (AI chat for scientific PDFs)
* [JSTOR](https://www.jstor.org/) (AI chat for scientific PDFs)
* [Cody](https://meetcody.ai/) (Customly trainable AI assistant for businesses)
* [Rawdog](https://github.com/AbanteAI/rawdog) (CLI assistant that responds by generating and auto-executing a Python script)
* [Devin](https://www.cognition-labs.com/) (AI Software Engineer by Cognition)
* [LaVague](https://github.com/lavague-ai/LaVague) (Browser interaction and task automation)
* [Chat with RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/) (Locally personalized LLM by NVIDIA - 35Gb)
* [EagleX](https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b) (Attention-free transformer LLM based on the RWKV-v5 architecture)
* [Sora](https://openai.com/sora) (Creating video from text by OpenAI)
* [Open-Sora](https://github.com/hpcaitech/Open-Sora) (Open-source version of Sora)

#### LLM Frameworks
* [LoRA](https://huggingface.co/docs/diffusers/en/training/lora) (Lightweight training technique that reduces the number of trainable parameters)
* [DSPy](https://dspy-docs.vercel.app/) (Solves the fragility in LLM apps by replacing prompting with programming and compiling) [[More Info](https://towardsdatascience.com/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9)]
* [LangChain](https://www.langchain.com/) (Build, observe, and deploy LLM‑powered apps easily)
* [LlamaIndex](https://www.llamaindex.ai/) (Turn your enterprise data into production-ready LLM applications)
* [Ollama](https://ollama.com/) (Get up and running with large language models, locally)
* [Phoenix](https://phoenix.arize.com/) (For AI observability and evaluation)
* [LM Studio](https://lmstudio.ai/) (Discover, download, and run local LLMs)
* [NeoSync](https://github.com/nucleuscloud/neosync) (To create synthetic data or anonymize sensitive data for fine-tuning or model training)
* [Langfuse](https://langfuse.com/) (Open Source LLM Engineering Platform)
* [FastText](https://fasttext.cc/) (Library for efficient text classification and representation learning)
* [PhiData](https://github.com/phidatahq/phidata) (A toolkit for building AI Assistants with function calling and connecting LLMs to external tools)
* [ScreenAI](https://blog.research.google/2024/03/screenai-visual-language-model-for-ui.html) (A visual language model for UI and visually-situated language understanding)
* [MLX](https://github.com/ml-explore/mlx) (An array framework for machine learning research on Apple silicon)

#### Computer Vision
* [Midjourney](https://www.midjourney.com/home) (Image generator) 
* [Magnific.ai](https://magnific.ai/) (Image upscaler)
* [Lisa AI](https://lisaai.app/) (Artistic image generator)
* [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) (Text-to-image model from StabilityAI)
* [DreamBooth](https://dreambooth.github.io/) (A fine-tuning model for stable diffusion model)
* [VideoPoet](https://sites.research.google/videopoet/) (A large language model for zero-shot video generation)
* [Supervision by Roboflow](https://github.com/roboflow/supervision) (A Python package of reusable computer vision tools)
* [YOLO](https://github.com/ultralytics/ultralytics) (Object detection, segmentation, pose estimation, tracking, and classification)
* [LLaVa-NeXT](https://huggingface.co/docs/transformers/en/model_doc/llava_next) (An open-source multimodal language model that can take image)

#### Productivity
* [Localpilot](https://github.com/danielgross/localpilot) (Local GitHub Copilot on Macbook)
* [NextFlow](https://www.nextflow.io/) (Reproducible scientific workflows using software containers)
* [nf-core](https://nf-co.re/pipelines) (Bioinformatics pipelines)

#### Resources & Repositories
* [labml](https://nn.labml.ai/) (Repository of annotated research paper implementations)
* [Awesome Open-source Machine Learning for Developers](https://github.com/merveenoyan/awesome-osml-for-devs)
* [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch) (Build a Large Language Model From Scratch)
* [ML Engineering](https://github.com/stas00/ml-engineering) (Machine Learning Engineering Open Book)

#### AI Deployment
* [Vercel](https://vercel.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Chroma](https://www.trychroma.com/)
* [Open-WebUI](https://docs.openwebui.com/)
* [Raycast](https://www.raycast.com/)
* [Streamlit](https://streamlit.io/)
* [NiceGUI](https://nicegui.io/)
* [Transformers.js](https://github.com/xenova/transformers.js) (Run 🤗 Transformers directly in your browser, with no need for a server)
* [WebGPU](https://en.wikipedia.org/wiki/WebGPU) (A JS API that enables webpage scripts to efficiently utilize a device's GPU)
* [WASM](https://webassembly.org/) (A binary instruction format for compiling and executing code in a client-side web browser)
* [Burn](https://github.com/tracel-ai/burn) (Dynamic Deep Learning Framework built using Rust)
* [Pico MLX Server](https://github.com/ronaldmannak/PicoMLXServer) (A GUI to download and start AI models locally on Mac)

## Transformer Models

#### Encoder-only (Classification or Extractive Summarization)
* [BERT](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
* [RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta) 

#### Decoder-only (Prompty-based Autoregressive Text Generation)
* [GPT](https://huggingface.co/docs/transformers/en/model_doc/openai-gpt)
* [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)
* [Mixtral](https://huggingface.co/docs/transformers/en/model_doc/mixtral)
* [LLaMa](https://huggingface.co/docs/transformers/main/en/model_doc/llama)
* [PaLM](https://ai.google/discover/palm2/)

#### Encoder + Decoder (Translation or Abstractive Summarization)
* [BART](https://huggingface.co/docs/transformers/en/model_doc/bart)
* [T5](https://huggingface.co/docs/transformers/en/model_doc/t5)
* [PEGASUS](https://huggingface.co/docs/transformers/en/model_doc/pegasus) 

<img src="https://raw.githubusercontent.com/smtnkc/ai-notebook/main/transformer_models.png" alt="transformer_models" width="720"/>

## LLMs Evaluation

<img src="https://raw.githubusercontent.com/smtnkc/ai-notebook/main/llms_evaluation.png" alt="llms_evaluation" width="720"/>

## Overused Words

> holistic, multifaceted, nuanced, comprehensive, meticulously, landscape, realm, delve, tapestry, seamlessly, unleash, embark, unwavering, unraveling, testament, leveraging, fostering, cultivate, intricate, it’s important to…
 
## References
* [Understanding Deep Learning (Simon J.D. Prince)](https://udlbook.github.io/udlbook/)
* [Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)
* [Encoder-Only vs Decoder-Only vs Encoder-Decoder Transformer](https://vaclavkosar.com/ml/Encoder-only-Decoder-only-vs-Encoder-Decoder-Transfomer)
* [BART Text Summarization vs. GPT-3 vs. BERT: An In-Depth Comparison](https://www.width.ai/post/bart-text-summarization)
* [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
* [Domain Adaptation with HuggingFace MLM](https://www.kaggle.com/code/hinepo/domain-adaptation-with-mlm)
* [Training BERT from Scratch on Your Custom Domain Data](https://medium.com/@shankar.arunp/training-bert-from-scratch-on-your-custom-domain-data-a-step-by-step-guide-with-amazon-25fcbee4316a)
* [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
