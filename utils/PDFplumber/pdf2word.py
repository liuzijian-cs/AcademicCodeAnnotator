import pdfplumber
from .pdf_read_dir import getDir
# import pdf_read_dir as prd


# 过长标题截断处理
def truncation(chapter_name):
    if len(chapter_name) > 15:
        chapter_name = chapter_name[0: 15]

    return chapter_name


def get_chapter(filepath):
    # getDir()函数返回三个值
    # file_dir_txt_path：文章完整目录的txt文件的文件路径
    # directory_str：文章完整目录的str类型变量
    # first_directory：文章一级目录的数组
    _, _, first_dir = getDir(filepath)

    chapter1_name = ''
    chapter1_next_name = ''
    chapter2_name = ''
    chapter2_next_name = ''
    chapter1_str = ''
    chapter2_str = ''

    for index, d in enumerate(first_dir):
        if d.lower().find('intro') != -1:
            chapter1_name = d.lower()
            chapter1_next_name = first_dir[index + 1].lower()
        elif d.lower().find('metho') != -1:
            chapter2_name = d.lower()
            chapter2_next_name = first_dir[index + 1].lower()

    # 过长标题截断处理
    chapter1_name = truncation(chapter1_name)
    chapter1_next_name = truncation(chapter1_next_name)
    chapter2_name = truncation(chapter2_name)
    chapter2_next_name = truncation(chapter2_next_name)

    print(chapter1_name)
    print(chapter1_next_name)
    print(chapter2_name)
    print(chapter2_next_name)

    with pdfplumber.open(filepath) as pdf:
        word = ''
        for p in pdf.pages:
            str = p.extract_text(x_tolerance=1)  # 调整x_tolerance参数，避免单词粘滞
            # 去除paper页号
            str = str.rstrip(str[-1])
            str = str.rstrip(str[-1])
            word += str
        word_low = word.lower()

        print(word_low.find(chapter2_name))
        print(word_low.find(chapter2_next_name))

        if chapter1_name != '':
            if chapter1_next_name != '':
                chapter1_str = word[word_low.find(chapter1_name): word_low.find(chapter1_next_name)]
            else:
                chapter1_str = word[word_low.find(chapter1_name): -1]
        if chapter2_name != '':
            if chapter2_next_name != '':
                chapter2_str = word[word_low.find(chapter2_name): word_low.find(chapter2_next_name)]
            else:
                chapter2_str = word[word_low.find(chapter2_name): -1]

    return chapter1_str, chapter2_str
