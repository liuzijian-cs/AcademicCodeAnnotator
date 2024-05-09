# Academic Code Annotator

![](./files/pic/readme_1.webp)

Academic Code Annotator 基于LLAMA3的学术代码解读器。上传pdf格式的论文和python文件，根据论文的内容自动注释代码并导出。

# 快速开始：

- 学术代码解读：
```shell
streamlit run web_aca.py /path/to/model
```

- 仅聊天
```shell
streamlit run web_chat.py /path/to/model
```

## 目录结构：

|- files: 基础文件\
|- tmp: 临时目录，用于上传下载文件\
|- utils: 工具包\
    |- PDFplumber: PDF文件读取算法\
    |- base_function.py: 基础功能\
|- web_aca.py: Academic Code Annotator (with LLAMA3)\
|- web_chat.py: 基础聊天页面(with LLAMA3)

## 模型下载：
| Model                                 | Download                                                                          |
|---------------------------------------|-----------------------------------------------------------------------------------|
| Meta-Llama-3-8B                       | [😊ModelScope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B/summary) |
| Meta-Llama-3-8B-Instruct (Recommend*) |[😊ModelScope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/summary)|
|Llama-3-Chinese-8B-Instruct|[😊ModelScope](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct/summary)|

## 已知缺陷：
- 此项目的PDF读取逻辑为基于规则的读取逻辑，因此可能仅适用于部分期刊的论文（且仅限单栏）；
- 此项目的微调策略较为简单，仅在代码层面进行了提示词拼接，如使用其他微调策略可能效果更加~；


## 运行结果：
1. 文献分析：

![](./files/pic/readme_result_2.png)

2. 代码上传：

![](./files/pic/readme_result_3.png)


3. 代码下载：

![](./files/pic/readme_result_4.png)

## 团队成员：

<a href="https://github.com/liuzijian-cs/AcademicCodeAnnotator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liuzijian-cs/AcademicCodeAnnotator" />
</a>

<a href="https://github.com/liuzijian-cs/AcademicCodeAnnotator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OneMiTang/AcademicCodeAnnotator" />
</a>