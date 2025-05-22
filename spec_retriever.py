import re
import xml.etree.ElementTree as ET


class FileRange:
    pattern = r'([^:]+):\s*(\d+)~(\d+)'
    pattern2 = r'([^:]+)::\s*(\d+)~(\d+)'

    def init_from_elements(self, file: str, start: int, end: int):
        self.file: str = file
        self.start: int = int(start)
        self.end: int = int(end)

    def init_from_str(self, str_repr: str):
        obj = re.match(self.pattern, str_repr.strip())
        if not obj:
            obj = re.match(self.pattern2, str_repr.strip())
            if not obj:
                # raise Exception(f'FileRange parse error: {str_repr}')
                self.file: str = 'not a file'
                self.start: int = -1
                self.end: int = -1
                return
        self.file: str = obj.group(1)
        self.start: int = int(obj.group(2))
        self.end: int = int(obj.group(3))
    

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and type(args[0] == str):
            self.init_from_str(args[0])
        elif len(args) == 3 and type(args[0] == str and type(args[1]) == int and type(args[2]) == int):
            self.init_from_elements(*args)

    def __str__(self):
        return 'File: ' + self.file + ':' + str(self.start) + '~' + str(self.end)
    
    def __eq__(self, other):
        if isinstance(other, FileRange):
            return self.file == other.file and self.start == other.start and self.end == other.end
        return False
    
    def cover(self, other):
        if not isinstance(other, FileRange):
            raise Exception("cover parameter must be of FileRange type")
        return self.file == other.file and self.start <= other.start and self.end >= other.end
    
    def soft_match(self, other):
        if not isinstance(other, FileRange):
            raise Exception("soft_match parameter must be of FileRange type")
        return self.file == other.file
    
    @classmethod
    def from_list(cls, strs: list[str]):
        ret = []
        for str in strs:
            ret.append(FileRange(str))
        return ret

def extract_file_lines(text):
    pattern = r'\|\s*`([^`]+)`\s*\|\s*L(\d+)-(\d+)\s*\|'
    matches = re.findall(pattern, text)
    
    results: list[FileRange] = []
    for match in matches:
        if len(match) != 3:
            a = 1
        results.append(FileRange(*match))
    
    return results

def retrieve_spec(text: str):

    text = text.strip()

    questions = []
    refs = []
    answers = []

    user_split = '_**User**_'
    output_split = '_**Assistant**_'
    segment_split = '\n---'

    conversations = text.split(user_split)[1:]

    for conversation in conversations:
        conversation = conversation.strip()
        if len(conversation.split(output_split)) == 1:
            a = 1
        input = conversation.split(output_split)[0].strip()
        output = conversation.split(output_split)[1].strip()

        segments = output.split(segment_split)[:-1]

        questions.append(input.split(segment_split)[0].strip())

        ref = []
        for i in range(len(segments) - 1):
            ref.extend(extract_file_lines(segments[i]))
        
        refs.append(ref)
        answers.append(segments[len(segments) - 1].strip())

    return questions, refs, answers


def get_tp(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    tp = 0
    for g in groundtruth:
        for c in candiate:
            if (soft and g.soft_match(c)) or (not soft and c.cover(g)):
                tp += 1
                break
    return tp


def get_fn(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    return len(groundtruth) - get_tp(groundtruth, candiate, soft)


def get_fp(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    fp = len(candiate)
    for c in candiate:
        for g in groundtruth:
            if (soft and g.soft_match(c)) or (not soft and c.cover(g)):
                fp -= 1
                break
    return fp


def calculate_recall(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    tp = get_tp(groundtruth, candiate, soft)
    fn = get_fn(groundtruth, candiate, soft)
    return tp / (tp + fn)


def calculate_precision(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    tp = get_tp(groundtruth, candiate, soft)
    fp = get_fp(groundtruth, candiate, soft)
    return tp / (tp + fp)


def calculate_f1(groundtruth: list[FileRange], candiate: list[FileRange], soft: bool=True):
    recall = calculate_recall(groundtruth, candiate, soft)
    precision = calculate_precision(groundtruth, candiate, soft)
    if recall + precision == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))


if __name__ == '__main__':
    with open('tmp.txt', 'r') as f:
        text = f.read()

    questions, refs, answers = retrieve_spec(text)
    print('questions: ', questions)
    print('refs: ', refs)
    print('answers: ', answers)

    empty_cnt = 0

    with open('tmp_output.txt', 'w') as f:
        for i in range(len(questions)):
            f.write(f'question {i}:\n')
            f.write(questions[i] + '\n')
            f.write(f'ref {i}:\n')
            for j in refs[i]:
                f.write(str(j))
                f.write('\n')
            f.write(f'answer {i}:\n')
            f.write(answers[i] + '\n')
            
            if len(refs[i]) == 0:
                empty_cnt += 1
        
    print(empty_cnt)
    print(len(questions))

