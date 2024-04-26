import os
import sys
from PyPDF2 import PdfReader


# 查找指定的字符出现次数
def find_char(str1, char):
    cs = 0
    for c in str1:
        if c == char:
            cs += 1
    return cs


directory_str = ''
first_directory = []
# first = True

def bookmark_listhandler(list):
    global directory_str
    global first_directory
    global first
    for message in list:
        if isinstance(message, dict):
            title = message['/Title'].strip()
            if title.startswith("Chapter"):
                directory_str += '\n' + title + '\n'
                first_directory.append(title)
            elif title[0:2] in ("序章", "前言") or title.startswith("序"):
                directory_str += '\n' + title + '\n'
                first_directory.append(title)
            elif title.startswith("第") and title.split()[0][-1] == "章":
                directory_str += '\n' + title + '\n'
                first_directory.append(title)
            elif title.startswith("第") and title.split()[0][-1] == "节":
                directory_str += '  ' + title + '\n'
            elif title.startswith("第"):
                directory_str += '\n' + title + '\n'
                first_directory.append(title)
            elif title[0] in ('一', '二', '三', '四', '五', '六', '七', '八', '九', '十'):
                directory_str += '    ' + title + '\n'
            elif title[0] in "1234567890":
                cs = find_char(title, '.')
                directory_str += '  ' * cs + title + '\n'
                # 只存一级标题
                if cs == 0:
                    first_directory.append(title)
                #     first = False
                # elif first == True:
                #     first_directory.append(title)
            else:
                directory_str += '        ' + title + '\n'
                # if first == True:
                #     first_directory.append(title)
        else:
            bookmark_listhandler(message)


def getDir(filepath):
    global directory_str
    global first_directory
    # global first

    # 初始化变量
    directory_str = ''
    first_directory = []
    # first = True

    if not os.path.exists(filepath):
        print(f"{filepath} is not exists.")
        sys.exit(2)

    fn, ext = os.path.splitext(filepath)
    if ext.lower() != '.pdf':
        print("Please specify a valid pdf file")
        sys.exit(3)

    with open(filepath, 'rb') as f1:
        pdf = PdfReader(f1)
        # 检索文档中存在的文本大纲,返回的对象是一个嵌套的列表
        bookmark_listhandler(pdf.outline)

    if len(directory_str) > 0:
        fname = fn.split('\\')[-1]
        file_dir_txt_path = fname + '_Dir' + '.txt'
        with open(file_dir_txt_path, 'w', encoding='utf-8') as fp:
            fp.write(directory_str)
    else:
        print("it no directory.")

    return file_dir_txt_path, directory_str, first_directory
