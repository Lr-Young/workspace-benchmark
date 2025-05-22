
from numpy import place
import numpy as np
from openai import OpenAI  
import pandas as pd
import re
import yaml
import json
import tqdm
from pathlib import Path
import time
import hashlib
import requests
import dashscope
from http import HTTPStatus
import json

from spec_retriever import *

from prompt import *

api_key = "sk-3gZ5uUx0IFUWKCeiC7Fa223b5dE34840Aa8cF5D06574Df09"  # 请替换为您的 API Key  
api_base = "http://maas-api.cn-huabei-1.xf-yun.com/v1"  
client = None


API_KEY = 'sk-b93acb44de69499bb467ca611357a412'
dashscope.api_key = API_KEY


def llm(prompt: str) -> str:
    global client
    if not client:
        client = OpenAI(api_key=api_key,base_url=api_base)  
    try:
        response = client.chat.completions.create(
            model="xdeepseekv3",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.7,
            max_tokens=4096,
            extra_headers={"lora_id": "0"},  
            stream_options={"include_usage": True},
            logprobs=True,
            top_logprobs=20
        )

        return str(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return ''
    

def en_to_zh(text: str):
    # 百度翻译API配置信息
    APP_ID = '20241202002217278'
    SECRET_KEY = 'U4a4zoQNEWesL6pbr4Zx'

    url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    salt = str(time.time())
    sign = hashlib.md5((APP_ID + text + salt + SECRET_KEY).encode('utf-8')).hexdigest()
    params = {
        'q': text,
        'from': 'en',
        'to': 'zh',
        'appid': APP_ID,
        'salt': salt,
        'sign': sign
    }
    response = requests.get(url, params=params)
    result = response.json()

    # 添加错误处理和日志记录
    if 'trans_result' in result:
        translations = [item['dst'] for item in result['trans_result']]
        # 将翻译结果按行拼接起来
        translated_text = '\n'.join(translations)
        return translated_text
    else:
        # 打印错误信息和完整的API响应
        print(f"翻译API响应错误: {result}")
        return text  # 如果翻译失败，返回原文
    

def str_to_embedding(text: str) -> list:
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=text)
    if resp.status_code == HTTPStatus.OK:
        embedding = resp.get('output').get('embeddings')[0].get('embedding')
        return embedding
    else:
        print(resp)
        return None


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def text_cosine_similarity(text1: str, text2: str):
    return cosine_similarity(str_to_embedding(text1), str_to_embedding(text2))

def parse_placeholders():
    df = pd.read_excel('question_type.xlsx', header=None)
    placeholders = set()
    pattern = r'(\[[^\]]*\])'
    for i in range(len(df)):
        match = re.findall(pattern, df.iloc[i, 0])
        for e in match:
            placeholders.add(e)
    print(len(placeholders))
    for e in placeholders:
        print(e)
    return placeholders


def write_placeholders_yaml(path: str):
    placeholders = parse_placeholders()
    data = dict()
    for e in placeholders:
        data[f'{e}'] = []
    # with open(path, 'w', encoding='utf-8') as f:
    #     yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

# write_placeholders_yaml('placeholders.yaml')

def retrieve_question():
    df = pd.read_excel('questions.xlsx')
    
    column_names = df.columns.tolist()
    urls = [column_names[i].split(' ')[0] for i in range(2, 6)]
    data = {}
    data['question_num'] = 0
    data['questions'] = []
    i = 0
    while i < len(df):
        question_type = df.iloc[i, 0]
        if pd.isna(df.iloc[i, 1]):
            i += 1
            for j in range(4):
                data['questions'].append({
                    'question': question_type,
                    'repository url': urls[j],
                    'deepseek-r1': '',
                    'cursor-context': '',
                    'context': '',
                    'answer': '',
                })
                data['question_num'] += 1
            continue
        placeholders = {}
        placeholders[df.iloc[i, 1]] = [df.iloc[i, j] for j in range(2, 6)]
        i += 1
        while i < len(df) and pd.isna(df.iloc[i, 0]):
            placeholders[df.iloc[i, 1]] = [df.iloc[i, j] for j in range(2, 6)]
            i += 1
        for j in range(4):
            question:str = question_type
            for k, v in placeholders.items():
                print(question)
                question = question.replace(f'[{k}]', f'{v[j]}')
            data['questions'].append({
                'question': question,
                'repository url': urls[j],
                'deepseek-r1': '',
                'cursor-context': '',
                'context': '',
                'answer': '',
            })
            data['question_num'] += 1

    questions = data.get("questions", [])

    df = pd.DataFrame(questions)

    output_path = 'data.xlsx'
    df.to_excel(output_path, index=False)

    return data
        

def extract_answer_from_think(text: str):
    # 使用正则表达式匹配答案部分
    if not re.search(r'<think>', text) or not re.search(r'</think>', text):
        return text.strip()
    text = text.strip()
    pattern = r'<think>.*?</think>\s*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        print("extract_answer_from_think wrong!\n" + text)
        exit(1)

def extract_answer_and_points(raw_answer: str):
    raw_answer = raw_answer.strip()
    pattern1 = r'### Canonical Answer'
    pattern2 = r'### Evaluation Dimensions \(Total: 10 points\)'
    
    if not re.search(pattern1, raw_answer):
        pattern1 = r'Canonical Answer'
    if not re.search(pattern2, raw_answer):
        pattern2 = r'Evaluation Dimensions \(Total: 10 points\)'
    
    match1 = re.search(pattern1, raw_answer)
    match2 = re.search(pattern2, raw_answer)

    if not match1:
        print("extract_answer_and_points wrong!\n" + pattern1 + " not found\n" + raw_answer)
        exit(1)
    if not match2:
        print("extract_answer_and_points wrong!\n" + pattern2 + " not found\n" + raw_answer)
        exit(1)
    
    answer = raw_answer.split(match2.group(0))[0].split(match1.group(0))[1].strip()
    points = raw_answer.split(match2.group(0))[1].strip()

    return answer, points


def extract_point_and_evaluation(evaluation_outcome: str):
        evaluation_outcome = evaluation_outcome.strip()
        pattern = r'Candidate Answer Score: (\[)?([\d\.]+)(\])?/10'
        match = re.search(pattern, evaluation_outcome)
        
        if not match:
            print("extract_point_and_evaluation wrong!\n" + evaluation_outcome)
            exit(1)
        point = str(match.group(2))
        evaluation = evaluation_outcome.split(match.group(0))[1].strip()
        return point, evaluation


def generate_rewrited_answer(file_name):
    df = pd.read_excel(file_name, header=None)
    df[5] = ['' for i in range(len(df))]
    df[6] = ['' for i in range(len(df))]
    for i in tqdm.tqdm(range(len(df))):
        answer = df.iloc[i, 3]
        points = df.iloc[i, 4]
        prompt = get_rewrite_answer_prompt(answer, points)
        df.iloc[i, 5] = prompt
        df.iloc[i, 6] = llm(prompt)
    df.to_excel(file_name, index=False, header=None)


def generate_evaluation_prompt(file_name):
    df = pd.read_excel(file_name, header=None)
    df[8] = ['' for i in range(len(df))]
    for i in tqdm.tqdm(range(len(df))):
        question = df.iloc[i, 0]
        answer = df.iloc[i, 3]
        points = df.iloc[i, 4].strip().split('\n')
        candidate_answer = df.iloc[i, 7]
        prompt = get_evaluation_prompt(question, answer, points, candidate_answer)
        result = llm(prompt)
        df.iloc[i, 8] = result
    df.to_excel(file_name, index=False, header=None)


def generate_evaluation(file_name):
    df = pd.read_excel(file_name)
    titles = ['deepseek-R1', 'deepseek-V3', 'gpt-4o']
    for i in tqdm.tqdm(range(len(df))):
        question = df['question'].iloc[i]
        answer = df['answer'].iloc[i]
        points = df['points'].iloc[i].strip().split('\n')
        for title in titles:
            print(f'generating {title} evaluation')
            candidate_answer = df[title].iloc[i]
            prompt = get_evaluation_prompt(question, answer, points, candidate_answer)
            df.loc[i, title + '-evaluation-prompt'] = prompt
            df.loc[i, title + '-score'] = llm(prompt)
    
    df.to_excel(file_name, index=False)


def strip_points(file_name, xlsx_name):
    df = pd.read_excel(xlsx_name)
    file = Path(file_name)
    content = ''
    write_content = ''
    cnt = 0
    with file.open('r', encoding='utf-8') as f:
        for line in f:
            pattern1 = r'\([\d\.]+ point(s)?\):'
            match1 = re.search(pattern1, line)
            pattern2 = r'\d+\.'
            match2 = re.search(pattern2, line)
            if match1 and match2:
                if match2.group() == '1.' and content != '':
                    df.iloc[cnt, 2] = content
                    write_content += content + '\n\n\n'
                    cnt += 1
                    content = ''
                content += match2.group() + ' ' + line.split(match1.group())[1].strip() + ' ' + match1.group().strip(':') + '\n'
            else:
                print('error match')
        df.iloc[cnt, 2] = content
                
    with file.open('w', encoding='utf-8') as f:
        f.write(write_content)
    print(content)
    print(cnt)

def reset_column(data_frame: pd.DataFrame, column_title: str):
    data_frame[column_title] = ['' for _ in range(len(data_frame))]


def insert_column_right(data_frame: pd.DataFrame, column_title: str, new_title: str):
    target_index = data_frame.columns.get_loc(column_title)
    new_column_data = ['' for _ in range(len(data_frame))]
    data_frame.insert(target_index + 1, new_title, new_column_data)


def delete_column(data_frame: pd.DataFrame, column_title: str):
    if column_title in data_frame.columns:
        data_frame.drop(column_title, axis=1, inplace=True)


def translate_column(data_frame: pd.DataFrame, column_title: str):
    translate_title = column_title + '-translate'
    if column_title in data_frame.columns:
        delete_column(data_frame, translate_title)
        insert_column_right(data_frame, column_title, translate_title)
        for i in range(len(data_frame)):
            content = data_frame.loc[i, column_title]
            translated_content = en_to_zh(content)
            data_frame.loc[i, translate_title] = translated_content



# 生成 从原始答案到标准答案的提示词
def generate_dataset_answer_prompt(file_name):
    df = pd.read_excel(file_name)
    reset_column(df, 'answer_prompt')
    for i in tqdm.tqdm(range(len(df))):
        question = df.loc[i, 'question']
        think_answer = df.loc[i, 'think_answer']
        answer_prompt = get_answer_points_prompt(question, extract_answer_from_think(think_answer))
        df.loc[i, 'answer_prompt'] = answer_prompt
    df.to_excel(file_name, index=False)


# 生成 各个待评估 LLM 的原始答案的评估提示词和评估结果
def generate_dataset(file_name, model_names: list[str], reset: bool):
    df = pd.read_excel(file_name)
    if reset:
        for model_name in model_names:
            if not model_name in df.columns:
                print(model_name + " is not in column titles")
                exit(1)
            prompt_column = model_name + '-evaluation-prompt'
            output_column = model_name + '-evaluation-output'
            evaluation_column = model_name + '-evaluation'
            scroe_column = model_name + '-score'
            delete_column(df, prompt_column)
            delete_column(df, output_column)
            delete_column(df, evaluation_column)
            delete_column(df, scroe_column)
            insert_column_right(df, model_name, evaluation_column)
            insert_column_right(df, model_name, scroe_column)
            insert_column_right(df, model_name, output_column)
            insert_column_right(df, model_name, prompt_column)
    for i in tqdm.tqdm(range(len(df))):
        question = df.loc[i, 'question']
        raw_answer = df.loc[i, 'raw_answer']
        answer, points = extract_answer_and_points(raw_answer)
        df.loc[i, 'answer'] = answer
        df.loc[i, 'points'] = points
        
        for model_name in tqdm.tqdm(model_names):
            if pd.isna(df.loc[i, model_name]):
                continue
            candidate_answer = df.loc[i, model_name]
            candidate_answer = extract_answer_from_think(candidate_answer)
            evaluation_prompt = get_evaluation_prompt(question, answer, points.strip().split('\n'), candidate_answer)
            evaluation_outcome = llm(evaluation_prompt)
            point, evaluation = extract_point_and_evaluation(evaluation_outcome)

            df.loc[i, model_name + '-evaluation-prompt'] = evaluation_prompt
            df.loc[i, model_name + '-evaluation-output'] = evaluation_outcome
            df.loc[i, model_name + '-score'] = float(point)
            df.loc[i, model_name + '-evaluation'] = evaluation
            df.to_excel(file_name, index=False)
    
    df.to_excel(file_name, index=False)


def get_average_point(file_name, model_names: list[str]):
    df = pd.read_excel(file_name)
    points = dict()
    for model_name in model_names:
        if model_name not in df.columns:
            print(f"{model_name} is not a column in table!\n")
            exit(1)
        if model_name + '-score' not in df.columns:
            print(f"{model_name}-score is not a column in table!\n")
            exit(1)
        sum = float(0)
        for i in range(len(df)):
            sum += float(df.loc[i, model_name + '-score'])
        points[model_name] = sum / len(df)
    return points


def get_average_embedding(file_name, model_names: list[str]):
    df = pd.read_excel(file_name)
    points = dict()
    for model_name in model_names:
        if model_name not in df.columns:
            print(f"{model_name} is not a column in table!\n")
            exit(1)
        if model_name + '-embedding-similarity' not in df.columns:
            print(f"{model_name}-embedding-similarity is not a column in table!\n")
            exit(1)
        sum = float(0)
        for i in range(len(df)):
            sum += float(df.loc[i, model_name + '-embedding-similarity'])
        points[model_name] = sum / len(df)
    return points


def get_model_point(file_name, model_names: list[str]):
    df = pd.read_excel(file_name)
    points = dict()
    for model_name in model_names:
        scores = list()
        if model_name not in df.columns:
            print(f"{model_name} is not a column in table!\n")
            exit(1)
        if model_name + '-score' not in df.columns:
            print(f"{model_name}-score is not a column in table!\n")
            exit(1)
        for i in range(len(df)):
            scores.append(float(df.loc[i, model_name + '-score']))
        points[model_name] = scores
    return points


def generate_translation(file_name, model_names: list[str]):
    df = pd.read_excel(file_name)
    for model_name in model_names:
        translate_column(df, model_name)
        translate_column(df, model_name + '-evaluation-output')
    for column_title in ['answer', 'points']:
        translate_column(df, column_title)
        
    file_path = Path(file_name)
    dir_path = file_path.parent
    output_file_name = dir_path.joinpath(file_path.stem + '-translated' + file_path.suffix)
    df.to_excel(output_file_name, index=False)

def violin_plot(model_scores: dict[str, list]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    data = []

    for [k,v] in model_scores.items():
        avg_score = sum(v) / len(v)
        for i, score in enumerate(v):
            data.append({
                'Model': k,
                'Evaluation_Point': f'Point_{i + 1}',
                'Score': score,
                'Average_Score': avg_score
            })

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 绘制小提琴图
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Model', y='Score', data=df)
    plt.title('Evaluation Scores Distribution for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()


def generate_embedding_points(file_name: str, model_names: list[str]):
    df = pd.read_excel(file_name)
    for model_name in model_names:
        if not model_name in df.columns:
            print(model_name + " is not in column titles")
            exit(1)
        embedding_column = model_name + '-embedding-similarity'
        delete_column(df, embedding_column)
        insert_column_right(df, model_name, embedding_column)
    for i in tqdm.tqdm(range(len(df))):
        answer = df.loc[i, 'answer']
        for model_name in model_names:
            candidate_answer = df.loc[i, model_name]
            candidate_answer = extract_answer_from_think(candidate_answer)
            df.loc[i, model_name + '-embedding-similarity'] = text_cosine_similarity(answer, candidate_answer)
    df.to_excel(file_name, index=False)


def get_questions(file_name: str):
    df = pd.read_excel(file_name)
    questions = []
    for i in range(len(df)):
        questions.append(df.loc[i, 'question'])
    return questions


def calculate_context(file_name: str, cursor_output_file: str):
    with open(cursor_output_file, 'r') as f:
        text = f.read()
    questions, refs, answers = retrieve_spec(text)
    df = pd.read_excel(file_name)

    reset_column(df, 'cursor context')
    reset_column(df, 'recall')
    reset_column(df, 'precision')
    reset_column(df, 'f1')
    reset_column(df, 'soft-recall')
    reset_column(df, 'soft-precision')
    reset_column(df, 'soft-f1')
    
    for i in range(len(questions)):
        if pd.isna(df.loc[i, 'context']):
            continue
        context = df.loc[i, 'context']
        gt_ref = FileRange.from_list(context.strip().split('\n'))
        ref = refs[i]
        if len(ref) == 0:
            continue
        
        cursor_context = ''
        for e in ref:
            cursor_context += str(e) + '\n'
        df.loc[i, 'cursor context'] = cursor_context

        df.loc[i, 'recall'] = calculate_recall(gt_ref, ref, False)
        df.loc[i, 'precision'] = calculate_precision(gt_ref, ref, False)
        df.loc[i, 'f1'] = calculate_f1(gt_ref, ref, False)
        df.loc[i, 'soft-recall'] = calculate_recall(gt_ref, ref, True)
        df.loc[i, 'soft-precision'] = calculate_precision(gt_ref, ref, True)
        df.loc[i, 'soft-f1'] = calculate_f1(gt_ref, ref, True)
    
    df.to_excel(file_name, index=False)


# model_scores = dict()

# model_scores['model_A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# model_scores['model_B'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# model_scores['model_C'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
# model_scores['model_D'] = [10, 10, 10, 10, 9]
# model_scores['model_E'] = [5, 5, 5, 1]


# violin_plot(model_scores)

# generate_dataset_answer_prompt('xlsx/tmp.xlsx')
file = 'xlsx/benchmark-cutlass-80-context.xlsx'
models = ['huawei-workspace-agents']
# models = ['huawei-workspace-agents']
# generate_dataset('xlsx/benchmark-cutlass-20.xlsx', models, False)

# print(get_average_point('xlsx/benchmark-cutlass-20.xlsx', models))
# print(get_model_point('xlsx/benchmark-cutlass-20.xlsx', models))

# violin_plot(get_model_point('xlsx/benchmark-cutlass-20.xlsx', models))


# generate_translation('xlsx/benchmark-cutlass-20.xlsx', models)
# generate_embedding_points('xlsx/benchmark-cutlass-20.xlsx', models)


# generate_dataset_answer_prompt(file)

# file = 'xlsx/benchmark-cutlass-80.xlsx'
# models = ['deepseek-R1', 'deepseek-V3', 'gpt-4o', 'claude-3.7-sonnet']
# # generate_dataset(file, models, True)
# violin_plot(get_model_point(file, models))

# generate_dataset(file, models, True)

calculate_context(file, 'tmp.txt')


