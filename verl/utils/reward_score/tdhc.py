# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # 严格模式：精确匹配"匹配结果:是"或"匹配结果:否"的格式 匹配结果后的冒号是英文半角
        match = re.search(r"匹配结果:(是|否)", solution_str)
        if match is None:
            final_answer = None
        else:
            final_answer = match.group(1)
    elif method == "flexible":
        # 灵活模式：尝试在文本中查找"是"或"否"
        if "是" in solution_str or "否" in solution_str:
            # 查找最后一个出现的"是"或"否"
            is_present = solution_str.rfind("是")
            not_present = solution_str.rfind("否")
            if is_present > not_present:
                final_answer = "是"
            elif not_present > is_present:
                final_answer = "否"
            else:
                final_answer = None
        else:
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    print(f"Extracted answer: {answer}, Ground truth: {ground_truth}")
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


def my_reward_fn(solution_str, _ground_truth=None):
  return len(solution_str)/100