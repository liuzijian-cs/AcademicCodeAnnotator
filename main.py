# cite: https://github.com/SmartFlowAI/Llama3-XTuner-CN/blob/main/web_demo.py
# cite: https://github.com/CrazyBoyM/llama3-Chinese-chat

import copy
import warnings
import streamlit as st
import torch
from torch import nn
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # isort: skip
from peft import PeftModel
from utils.base_function import Color

"""
:copy: 供了用于浅复制和深复制可变对象的功能, 代码中使用了copy.deepcopy() 用于创建一个对象及其包含的对象的完整拷贝。
:warnings: 用于发出警告，以提示程序的某些问题，而不是抛出异常。
:streamlit: 用于快速创建和共享数据应用的Python库。
:torch: 是PyTorch的核心库，是一个用于深度学习的库，支持强大的张量计算（类似于NumPy）以及自动计算梯度等功能。
:torch.nn: 这是PyTorch中的子模块，提供了许多建立和训练神经网络所需的工具和层。
:dataclasses: 用于自动添加特殊方法（如__init__() 和 __repr__()）到类中，主要用于快速创建主要用于存储数据的类。
    asdict() 函数将数据类实例转换为字典，便于处理。
:typing: 这个模块支持Python的类型提示，用于在代码中添加变量的类型信息，提高代码的可读性和可维护性。
    Callable, List, Optional 是常用的类型，分别用于函数类型、列表类型和可选类型的注解。
:transformers.generation.utils: 提供了用于文本生成的工具，如LogitsProcessorList和StoppingCriteriaList。
    这些类用于控制生成过程，例如修改logits来引导生成的内容，以及定义何时停止文本生成。
:transformers.utils:transformers库的一个工具模块，包含日志和其他辅助功能。
    其中logging.get_logger()用于获取日志对象，以记录程序运行时的各种信息。
:transformers: AutoTokenizer 和 AutoModelForCausalLM 是transformers库的组件，用于自动加载预训练模型和相应的分词器。
    这些工具主要用于NLP任务，如文本生成或语言模型评估。
:peft: PeftModel 很可能是一个专门的模型或工具，用于加载和运行特定的预训练模型或适配器。这可能是特定于某个项目或库的自定义实现。
:utils.base_function: 本代码自定义的基础功能
    Color 定义了常见的ASCII颜色转义符
"""

logger = logging.get_logger(__name__)  # 初始化日志设置

# 此函数可用于设置其他页面属性，如布局和初始侧边栏状态，但在此代码段中仅设置了页面标题。
st.set_page_config(page_title="Academic Code Annotator (with LLAMA3 😊)")  # Streamlit 页面配置:设置了页面的标题


@dataclass
class GenerationConfig:
    """
    此配置用于聊天，以提供对话的多样性
    """
    max_length: int = 65535  # 定义生成文本的最大长度为 65535 个字符。这是生成过程中可以生成的字符的绝对上限。
    # max_new_tokens: int = 600  # 设置生成调用中新生成的最大令牌数（例如，单词或标点符号）。这有助于控制输出文本的大小。
    top_p: float = 0.8  # 生成文本时的随机采样策略。top_p 为 0.8 表示在每一步，只考虑累积概率质量至少占总概率质量 80% 的最高概率的词汇。
    temperature: float = 0.8  # 控制生成过程的随机性。温度越低，输出越倾向于高概率选项。0.8 是一个使输出既随机又可靠的中间值。
    do_sample: bool = True  # 是否在生成时使用采样策略。设置为 True 表示启用采样，这通常与 top_p 或 temperature 结合以增加输出的多样性。
    repetition_penalty: float = 1.05  # 重复惩罚，该参数用来降低重复词汇的出现概率。1.05 表示对于重复的词，其选择概率会被略微降低。


@torch.inference_mode()  # 装饰器用于优化性能，在此模式下，PyTorch 将不会计算梯度，这对于推理特别有用。
def generate_interactive(
        model,
        tokenizer,
        prompt,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
        List[int]]] = None,
        additional_eos_token_id: Optional[int] = None,
        **kwargs,  #
):
    """
    :param model: 预训练模型
    :param tokenizer: 与模型相配的分词器，用于文本的编码和解码。
    :param prompt: 初始文本提示，用于启动文本生成过程。
    :param generation_config: 一个可选的 GenerationConfig 数据类实例，包含生成文本时的配置。
    :param logits_processor: 用于调整生成的 logits，可以实现自定义的生成策略。
    :param stopping_criteria: 定义何时停止生成文本的条件。
    :param prefix_allowed_tokens_fn: 一个函数，定义哪些token可以在特定位置被生成。
    :param additional_eos_token_id: 额外的结束符token ID，用于扩展停止生成的条件。
    :param kwargs: 其他传递给模型的关键字参数。
    """
    # 将文本提示（prompt）转换为模型可以理解的格式（通常是token IDs）。return_tensors='pt' 指示分词器返回 PyTorch 张量。
    inputs = tokenizer([prompt], return_tensors='pt')
    # 计算输入的token数量（即输入文本的长度）。
    input_length = len(inputs['input_ids'][0])
    # 循环遍历所有输入张量，将它们移动到 GPU 上，以便模型可以在 GPU 上进行计算。
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    # 从 inputs 字典中提取 input_ids 张量，这是后续生成过程的主输入。
    input_ids = inputs['input_ids']
    # 这行代码提取 input_ids 张量的维度，其中 input_ids_seq_length 是序列的长度。
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    # 检查是否提供了生成配置 (generation_config)。如果没有提供，就使用模型的默认生成配置。
    if generation_config is None:
        generation_config = model.generation_config
    # 使用 deepcopy 来复制生成配置。这确保了原始配置不会在函数中被修改，使得这个函数具有纯功能性质（不改变外部状态）。
    generation_config = copy.deepcopy(generation_config)
    # 更新 generation_config 对象，将任何额外的关键字参数（kwargs）合并到配置中。这允许调用者根据需要自定义生成过程。
    model_kwargs = generation_config.update(**kwargs)
    # 提取开始符 (BOS) 和结束符 (EOS) 的 token ID。这些token用于标识生成文本的开始和结束。
    bos_token_id, eos_token_id = (
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    # 检查eos_token_id是否为整数，并将其转换为列表。这是为了处理生成过程中可能需要的多个结束符。
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    # 这行代码检查是否提供了额外的结束符 token ID (additional_eos_token_id)。
    # 如果提供了，它将被添加到 eos_token_id 列表中。这允许动态扩展文本生成的结束条件。
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    # 检查是否没有通过 kwargs 显式提供 max_length 参数，并且在 generation_config 中已经设置了默认的 max_length。
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    # 如果满足 has_default_max_length 且 max_new_tokens 未设置，将发出警告。
    # 这说明用户依赖于过时的配置方法来限制生成长度，这种做法在未来的版本中将不再支持。
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    # 如果 max_new_tokens 被设置，它将根据输入 ID 的序列长度调整 max_length 的值。这保证生成的长度与输入长度和新生成的 token 数量相适应。
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
                                       input_ids_seq_length
        if not has_default_max_length:
            # 如果 max_length 和 max_new_tokens 都被设置，将发出警告，告知用户 max_new_tokens 将优先使用，并推荐查阅相关文档了解更多信息。
            logger.warning(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )
    # 最后这部分检查输入的长度是否超过了设置的最大长度 (max_length)。
    # 如果是这样，将记录一条警告，指出这可能导致意外的行为，并建议增加 max_new_tokens 的值以避免这种情况。
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    # 下面这两行确保 logits_processor 和 stopping_criteria 都被正确地初始化。如果它们没有被外部提供（即为 None），
    # 则使用默认的 LogitsProcessorList 和 StoppingCriteriaList 来初始化。这些类来自 transformers 库，提供了基础的处理和停止机制。
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    # 这一行调用模型的内部方法 _get_logits_processor 来获取或配置 logits 处理器。
    # 该方法可能基于提供的 generation_config、输入的长度、输入 IDs、以及任何前缀允许的 token 函数来调整或增强传入的 logits_processor。
    # 这是为了确保 logits 处理器能够适应特定的生成任务和配置。
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )
    # 类似地，这一行调用模型的 _get_stopping_criteria 方法来配置或获取停止生成的条件。
    # 这可以根据 generation_config 和已有的 stopping_criteria 进行调整，确保生成过程能在适当的时机停止。
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    # 这一行获取一个 logits_warper，它是用于调整 logits 以改变生成概率分布的工具。这通常用于实现如温度调节或 top-k sampling 等高级生成技巧。
    logits_warper = model._get_logits_warper(generation_config)
    # 初始化一个与 input_ids 相同大小的 tensor，用于跟踪哪些序列尚未完成。所有元素初始设置为 1（表示序列仍在生成中）。
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # 初始化 scores 为 None，可能稍后用于存储调试或分析目的的分数或概率。
    scores = None
    # 开始一个无限循环，直到遇到中断条件才停止，通常是当文本生成过程完成或触发停止标准时。
    while True:
        # 根据当前 input_ids 的状态和额外的参数（model_kwargs）准备模型的输入数据。这一步骤通常将输入格式化为符合模型预期输入结构的方式。
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # 使用准备好的输入通过模型进行前向传递，生成下一个可能标记的 logits。return_dict=True 指定输出应该以字典形式返回。
        # output_attentions 和 output_hidden_states 设置为 False，以最小化内存使用，除非需要这些值。
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # 从输出中提取最后一个标记位置的 logits，这包含了模型对下一个标记的预测。
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        # 使用 logits_processor 处理 logits（可以实现防止标记重复的机制），
        # 然后使用 logits_warper（可能实现如温度调节或 top-k 采样等高级生成策略）。
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        # 使用 softmax 函数将处理后的 logits 转换为概率，softmax 函数将 logits 标准化为概率分布。
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # 通过从概率分布中采样（如果 do_sample 为 True）或选择最高概率的标记来决定下一个标记（Token）。
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        # 将新选择的标记连接到下一次迭代的 input_ids 序列中。
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # 根据当前的输出更新下一次迭代的关键字参数，这可能包括更新注意力掩码或其他下一次前向传递所需的状态。
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
        # 更新 unfinished_sequences 以跟踪哪些序列仍在生成中，哪些已完成，基于下一个标记是否与任何 EOS（句末标记）标记匹配。
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())
        # 提取并解码生成的标记为人类可读的文本，同时处理如果最后一个标记是 EOS 标记的可能性。
        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        # 将生成的响应返回给调用者，允许函数产生输出流而不是单批返回。
        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        # 检查是否所有序列都已完成（即在 unfinished_sequences 中全部标记为完成）或是否满足任何外部停止标准，如果是这样，就中断循环。
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


def on_btn_click():
    """
    这个函数定义了一个事件处理器，当相关的按钮被点击时触发。这里的操作是删除 Streamlit 会话状态中存储的 messages。
    该功能通常用于重置应用的状态或清除缓存的数据。
    """
    del st.session_state.messages


@st.cache_resource  # 这个函数使用 @st.cache_resource 装饰器，使得 Streamlit 可以缓存该函数的结果。
def load_model(model_name_or_path, adapter_name_or_path=None, load_in_4bit=False):
    """
    :param model_name_or_path: 指定模型的名称或路径
    :param adapter_name_or_path: 可选，指定适配器的名称或路径
    :param load_in_4bit: 是否在 4 位量化模式下加载模型
    :return: 返回加载的模型和分词器
    """
    # 这段条件代码检查是否需要以 4 位量化模式加载模型。
    # 如果是，它将创建一个量化配置对象 BitsAndBytesConfig，配置了各种量化参数，如计算数据类型和量化类型。
    # 如果不加载 4 位量化模式，则量化配置设置为 None。
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 量化过程中使用的计算数据类型。
            bnb_4bit_use_double_quant=True,  # 这个参数启用双重量化策略。在降低位宽的同时，尽可能保留更多的信息。
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,  # 指示权重是否以 FP16 格式存储。
        )
    else:
        quantization_config = None
    # 这段代码使用 transformers 库中的 AutoModelForCausalLM.from_pretrained，加载一个预训练的因果语言模型causal language model）。
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,  # 指定模型的名称或模型文件的路径。
        load_in_4bit=load_in_4bit,  # 指定是否在 4 位量化模式下加载模型, 可以显著减少模型的内存占用。
        trust_remote_code=True,  # 在加载远程或自定义模型时，是否信任和执行模型文件中包含的自定义代码。
        low_cpu_mem_usage=False,  # 这个选项在加载模型时减少 CPU 内存的使用。适用于内存资源受限的环境。
        torch_dtype=torch.float16,  # 指定模型使用的 PyTorch 数据类型。
        device_map='auto',  # 指定模型应该加载到哪个设备上。'auto' 表示自动选择最合适的设备。
        quantization_config=quantization_config  # 提供量化配置，这是通过 BitsAndBytesConfig 或其他相关配置类设置的。
    )
    # 如果提供了适配器路径或名称，则加载一个 PeftModel，这是一种支持适配器的模型，可以用来增强或调整原有模型的行为。
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # 将模型设置为评估模式，这通常在进行预测或推理时需要，以确保模型的某些训练特定行为（如 Dropout）被禁用。
    model.eval()
    # 加载与模型相对应的分词器 (AutoTokenizer)，用于将文本转换为模型可以理解的格式。
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    return model, tokenizer


def prepare_generation_config():
    """
    这段代码定义了一个函数 prepare_generation_config()，用于在 Streamlit 应用中配置和显示文本生成相关的超参数。
    该函数利用 Streamlit 的 UI 组件在侧边栏中创建一个交互式的控制面板。
    :return: 返回这个配置对象，使其可以在其他部分的应用中用于控制文本生成行为。
    """
    with st.sidebar:  # 语句指定接下来的 Streamlit 组件将显示在应用的侧边栏中。
        # TODO 修改这部分逻辑，加入pdf文件的上传、下载等，并编写对应的逻辑
        st.title('超参数面板')
        # st.text_area 创建了一个文本输入区域，用户可以在其中输入或修改预设的提示文本。
        system_prompt_content = st.text_area('系统提示词',
                                             "你是一个有创造的超级人工智能assistant,名字叫Llama3-Chinese,拥有全人类的所有知识。"
                                             "你喜欢用幽默风趣的语言回复用户，但你更喜欢用准确、深入的答案。"
                                             "你需要结合中国的文化和聊天记录中的上文话题理解和推测用户真正意图，按要求正确回复用户问题。"
                                             "注意使用恰当的文体和格式进行回复，尽量避免重复文字和重复句子，且单次回复尽可能简洁深邃。"
                                             "你关注讨论的上下文，深思熟虑地回复用户"
                                             "如果你不知道某个问题的含义，请询问用户，并引导用户进行提问。"
                                             "当用户说继续时,请接着aissistant上一次的回答进行继续回复。",
                                             height=200,  # height=200 设置了文本区域的高度。
                                             # key='system_prompt_content' 为这个 UI 组件定义了一个唯一的键值，可以用于后续操作。
                                             key='system_prompt_content'
                                             )  # TODO USELESS
        # 超参数滑块，st.slider 组件允许用户交互式地选择文本生成的各种参数
        max_new_tokens = st.slider('最大回复长度', 100, 8192, 660, step=8)  # 控制生成的最大长度。
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)  # 设置采样的 softmax 概率阈值，用于控制文本多样性。
        temperature = st.slider('温度系数', 0.0, 1.0, 0.7, step=0.01)  # 调节随机性的大小，影响生成文本的一致性和多样性。
        repetition_penalty = st.slider("重复惩罚系数", 1.0, 2.0, 1.07, step=0.01)  # 用于降低重复内容的发生。
        st.button('重置聊天', on_click=on_btn_click)  # 创建一个按钮，当被点击时触发 on_btn_click 函数，该函数可以用来重置聊天状态或清除会话数据。

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                         top_p=top_p,
                                         temperature=temperature,
                                         repetition_penalty=repetition_penalty,
                                         )
    return generation_config


system_prompt = '<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n'
user_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
robot_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n{robot}<|eot_id|>'
cur_query_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
"""
:system_prompt: 系统提示模板。它用来封装一些系统级的信息或指令，可能用于控制或指导生成模型的行为。
    使用 <<SYS>> 和 <</SYS>> 标签来明确地标识包围的内容是系统级的。
    {content} 是一个占位符，用于插入实际的系统信息或指令。这样的设计使得模板可以灵活地用于不同的系统信息。
    模板包含换行符 (\n) 以保持清晰的格式化输出。
:user_prompt: 用户输入的格式模板。它标记了一个用户发言的开始。
:robot_prompt: 类似于 user_prompt，助手或机器人回应的格式模板。
:cur_query_prompt: 这是一个用于当前查询的复合提示模板，它结合了用户的发言和助手的响应
"""


def combine_history(prompt):
    """
    整合聊天历史记录并构造用于文本生成系统的完整输入。
    该函数采用当前的用户输入（prompt）和会话历史，生成一个格式化的文本字符串，该字符串包含了所有先前的对话以及当前的查询。
    :param prompt: 当前的用户输入
    :return: 所有先前的对话以及当前的查询
    """
    # 从 Streamlit 的会话状态中获取 messages 列表，这里假设 messages 是一个字典列表，每个字典包含聊天消息的内容和角色（用户或机器人）。
    messages = st.session_state.messages
    # 初始化一个空字符串 total_prompt，用于累积整个对话的内容。
    total_prompt = ''
    # 遍历 messages 列表中的每个消息。
    for message in messages:
        # 提取每个消息的内容 (cur_content) 和角色，并根据角色使用适当的格式模板（之前定义的 user_prompt 或 robot_prompt）来格式化消息。
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    # 从 Streamlit 的会话状态中获取 system_prompt_content，这可能是预定义的或者是用户在某个界面元素中输入的系统级提示。
    system_prompt_content = st.session_state.system_prompt_content
    # 使用 system_prompt 模板格式化系统提示。
    system = system_prompt.format(content=system_prompt_content)
    # 将系统提示、累积的对话内容以及使用 cur_query_prompt 模板格式化的当前用户输入拼接在一起，形成最终的 total_prompt。
    total_prompt = system + total_prompt + cur_query_prompt.format(user=prompt)

    return total_prompt


def main(model_name_or_path, adapter_name_or_path):
    print(f'{Color.B}[Academic Code Annotator]{Color.RE}Loading model...')
    # 调用 load_model 函数加载指定的模型和分词器。这里，load_in_4bit=False 表示不使用 4 位量化加载模型。
    model, tokenizer = load_model(model_name_or_path, adapter_name_or_path=adapter_name_or_path, load_in_4bit=True)
    print(f'{Color.B}[Academic Code Annotator]{Color.RE}{Color.G}Load model successful!{Color.RE}')

    # 设置 Streamlit 页面标题
    st.title('Llama3-Chinese')  # TODO

    # 调用 prepare_generation_config 函数来设置并获取文本生成的配置参数。
    generation_config = prepare_generation_config()  # TODO 滑块貌似只在这里做了变化，没有做到动态更新

    # 初始化聊天历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 显示历史聊天消息
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('解释一下Vue的原理'):  # 使用 st.chat_input 获取用户的输入。
        # 使用 st.chat_message 显示用户的输入。
        with st.chat_message('user'):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)  # 调用 combine_history 函数将当前输入与历史消息组合，准备发送给模型。
        # 将用户消息添加到会话状态的 messages 列表。
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        # 使用 st.chat_message 来为机器人的回复创建一个消息容器。
        with st.chat_message('robot'):
            message_placeholder = st.empty()
            # 使用 generate_interactive 函数生成回复，期间通过 message_placeholder 实时更新显示的内容。
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=128009,
                    **asdict(generation_config),
            ):
                # 在聊天消息容器中显示机器人响应
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # 完成生成后，将机器人的最终回复添加到会话状态。
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
        })
        # 在生成过程结束后清空 CUDA 缓存，以管理 GPU 内存使用。
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # 导入 Python 的系统模块 sys，它包含了与 Python 解释器和它的环境有关的功能，比如命令行参数。
    import sys
    # sys.argv 是一个列表，包含了命令行参数。sys.argv[0] 是脚本名，sys.argv[1] 通常是第一个参数，这里被用来指定模型的名称或路径。
    model_name_or_path = sys.argv[1]
    # 这里检查 sys.argv 的长度是否大于等于 3，以确定是否有第二个命令行参数提供（即 sys.argv[2]）。如果有，将其作为适配器的名称或路径。
    if len(sys.argv) >= 3:
        adapter_name_or_path = sys.argv[2]
    else:
        adapter_name_or_path = None
    # 调用主函数
    main(model_name_or_path, adapter_name_or_path)
