import datetime, requests
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from grader import grade_answer
from datasets import load_dataset


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"
MAX_TOKEN_LENGTH = 16384
SAMPLE_SIZE = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Mathematical Benchmarks")
    parser.add_argument("--model_path", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path to the model or tokenizer")
    parser.add_argument("--benchmark", type=str, default="amc23",
                        choices=["amc23", "math500", "aime24", "gpqa"],
                        help="The benchmark dataset to evaluate")
    parser.add_argument("--port", type=str, default="42000",
                        help="SGLang server URL")
    return parser.parse_args()

args = parse_args()

def generate_completions(prompts:list, url:str, max_new_tokens = 16384, temperature = 0.9, top_k = 50, top_p = 0.9, n = 1, stop = None):

    data = {
        "text": prompts,
        "sampling_params": {
            "max_new_tokens": max_new_tokens, 
            'temperature': temperature, 
            "top_k": top_k,
            "top_p": top_p,
            'n': n,
        }
    }
    if stop is not None:
        data["sampling_params"]["stop"] = stop
        data['sampling_params']['no_stop_trim'] = True
        
    raw = requests.post(url+"/generate", json=data).json()
    responses = []
    try:
        if type(raw) is not list:
            responses.append(raw['text'])
        else:
            for row in raw:
                responses.append(row['text'])
    except:
        print("Error")
        log_path = "sglang_infer_error.log"
        with open(log_path, "a") as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Data: {data}\n")
            f.write(f"Raw: {raw}\n\n\n")
        return [""] * len(prompts)

    return responses

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    return remove_boxed(boxed)


MODEL_PATH = args.model_path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
def generate(question, tokenizer, n, port=args.port):
    url = f"http://localhost:{port}"
    chat = [
        {"role": "user", "content": f"{question}\n\n{SYSTEM_PROMPT}"}  
    ]
    
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    out = generate_completions([prompt], url, max_new_tokens=MAX_TOKEN_LENGTH, n=1, temperature = 0, top_k=-1, top_p=1)
    if n == 1:
        return out[0]
    return out

from datasets import load_dataset

amc23 = load_dataset("math-ai/amc23", split="test") 

math500 = load_dataset("HuggingFaceH4/MATH-500", split="test") 
math500 = math500.rename_column("problem", "question")

aime24 = load_dataset("math-ai/aime24", split="test") 
aime24 = aime24.rename_column("problem", "question")
aime24 = aime24.rename_column("solution", "answer")

gpqa = load_dataset("fingertap/GPQA-Diamond", split="test")



def eval(data):
    num_correct = 0
    tok_len = 0
    info = []

    for i in tqdm(range(len(data))):
        question = data[i]['question']
        gt = str(data[i]['answer'])

        response = generate(question, tokenizer, 1, port=args.port)
        answer = extract_solution(response)
        
        if grade_answer(answer, gt):
            num_correct += 1
        tok_len += len(tokenizer.encode(response))
        print(f"accuracy: {num_correct / (i + 1)} ({num_correct} / {i + 1})")
        print("avg_tok_len:", tok_len / (i + 1))
        info.append({
            "question": question,
            "ground_truth": gt,
            "response": response,
            "extracted_answer": answer,
            "is_correct": grade_answer(answer, gt)
        })
        
    acc = num_correct / len(data)
    avg_len = tok_len / len(data)
    
    return acc, avg_len
            
            
            
            
# --- Execution ---
if __name__ == "__main__":
    print(f"Evaluating Model: {args.model_path} on Benchmark: {args.benchmark}")
    
    if args.benchmark == "amc23":
        data = amc23
    elif args.benchmark == "math500":
        data = math500
    elif args.benchmark == "aime24":
        data = aime24
    elif args.benchmark == "gpqa":
        data = gpqa
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    acc, avg_len = eval(amc23)
    print(f"{args.benchmark}: Accuracy: {acc}, Average Token Length: {avg_len}")
    
