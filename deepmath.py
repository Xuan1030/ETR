import torch
import numpy as np

def remove_boxed(s):
    # print(s)
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    print(s)
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


def momentum_entropy_reward(entropy, alpha=0.4, beta=0.8, gamma=0.9):
        """
        Momentum-based recursive reward encouraging smooth entropy decrease.

        entropy: list[float] or torch.Tensor — 每句的entropy
        return: float — 越稳定下降reward越高
        """
        if not isinstance(entropy, torch.Tensor):
            entropy = torch.tensor(entropy, dtype=torch.float32)
        n = len(entropy)
        if n < 2:
            return 0.0

        R_prev = 0.0
        total_R = 0.0

        for t in range(1, n):
            dE = entropy[t] - entropy[t - 1]
            reward_t = (
                gamma * R_prev
                + alpha * torch.clamp(-dE, min=0)   # 熵下降→加分
                - beta * torch.clamp(dE, min=0)    # 熵上升→扣分
            )
            total_R += reward_t
            R_prev = reward_t

        return float(total_R)


def compute_score(solution_str, ground_truth, extra_info):
    sentence_level_entropy = extra_info["step_entropy"]
    trimmed = sentence_level_entropy[:torch.where(sentence_level_entropy != 0)[0][-1] + 1] if (sentence_level_entropy != 0).any() else torch.tensor([])
    if len(trimmed) == 0:
        return 0.0

    '''Step Entropy Trend Reward Version'''
    inertia_reward = momentum_entropy_reward(trimmed, gamma=0.99)
    correct_reward = 1.0 if extract_solution(solution_str) == ground_truth else 0.0

    if correct_reward == 0.0:
        return -1.0
    else:
        return correct_reward + 0.005 * inertia_reward 


 