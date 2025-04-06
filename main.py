
from numpy import place
from openai import OpenAI  
import pandas as pd
import re
import yaml
import json
import tqdm
from pathlib import Path

from prompt import *

api_key = "sk-3gZ5uUx0IFUWKCeiC7Fa223b5dE34840Aa8cF5D06574Df09"  # 请替换为您的 API Key  
api_base = "http://maas-api.cn-huabei-1.xf-yun.com/v1"  
client = None

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
    
    match1 = re.search(pattern1, raw_answer)
    match2 = re.search(pattern2, raw_answer)

    if not match1:
        print("extract_answer_and_points wrong!\n" + raw_answer)
        exit(1)
    if not match2:
        print("extract_answer_and_points wrong!\n" + raw_answer)
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


def generate_dataset_answer_prompt(file_name):
    df = pd.read_excel(file_name)
    reset_column(df, 'answer_prompt')
    for i in tqdm.tqdm(range(len(df))):
        question = df.loc[i, 'question']
        think_answer = df.loc[i, 'think_answer']
        answer_prompt = get_answer_points_prompt(question, extract_answer_from_think(think_answer))
        df.loc[i, 'answer_prompt'] = answer_prompt
    df.to_excel(file_name, index=False)


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

        reset_column(df, 'answer')
        reset_column(df, 'points')
    for i in tqdm.tqdm(range(len(df))):
        if i < 19:
            continue
        question = df.loc[i, 'question']
        raw_answer = df.loc[i, 'raw_answer']
        answer, points = extract_answer_and_points(raw_answer)
        df.loc[i, 'answer'] = answer
        df.loc[i, 'points'] = points
        
        for model_name in tqdm.tqdm(model_names):
            candidate_answer = df.loc[i, model_name]
            candidate_answer = extract_answer_from_think(candidate_answer)
            evaluation_prompt = get_evaluation_prompt(question, answer, points.strip().split('\n'), candidate_answer)
            evaluation_outcome = llm(evaluation_prompt)
            point, evaluation = extract_point_and_evaluation(evaluation_outcome)

            df.loc[i, model_name + '-evaluation-prompt'] = evaluation_prompt
            df.loc[i, model_name + '-evaluation-output'] = evaluation_outcome
            df.loc[i, model_name + '-score'] = point
            df.loc[i, model_name + '-evaluation'] = evaluation
            df.to_excel(file_name, index=False)
    
    df.to_excel(file_name, index=False)

# generate_dataset_answer_prompt('xlsx/tmp.xlsx')
generate_dataset('xlsx/tmp.xlsx', ['deepseek-R1', 'deepseek-V3', 'gpt-4o', 'claude-3.7-sonnet'], False)
