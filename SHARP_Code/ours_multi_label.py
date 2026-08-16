import os
import json
import re
import pandas as pd
import glob
import csv
import requests

import time
import logging

import helper.utils as utils
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


logger = logging.getLogger(__name__)
MAX_RETRIES = 3
BASE_SLEEP_SECONDS = 5

def encode_image_base64(img_path):
    return utils.encode_image_base64(img_path)



def call_llm(texts, images):

    ENDPOINT = "ENTER YOUR ENDPOINT HERE"
    API_KEY = os.environ.get("WRITE_API_KEY")

    headers = {
        "Content-type": "application/json",
        "api-key": API_KEY
    }


    parts = []
    for img in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    for text in texts:
        parts.append({
            "type": "text",
            "text": text
        })

    messages = [{ "role": "user", "content": parts}]
    body = {"messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(ENDPOINT, headers=headers, json=body, timeout=120)

            # print(response)
            # print("BODY:", response.text[:1000])

            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.warning(f"[Attempt {attempt}] LLM error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BASE_SLEEP_SECONDS * 2 ** (attempt - 1))
            else:
                logger.error("All retries failed.")
    return ""






def _call_gpt_parts(parts):
    # Call Gemini with already-built multimodal parts and return text.
    ENDPOINT = "ENTER YOUR ENDPOINT HERE"
    API_KEY = os.environ.get("WRITE_API_KEY")

    headers = {
        "Content-type": "application/json",
        "api-key": API_KEY
    }

    messages = [{ "role": "user", "content": parts}]
    body = {"messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(ENDPOINT, headers=headers, json=body, timeout=120)     
            response.raise_for_status()
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.warning(f"[Attempt {attempt}] LLM error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BASE_SLEEP_SECONDS * 2 ** (attempt - 1))
            else:
                logger.error("All retries failed.")
    return ""


def _build_hypothesis_episode_parts(image_label_pairs):
    # Build episode parts in the same spirit as hypthesis_generator.py, using Gemini inlineData.
    parts = []
    for i, img_pair in image_label_pairs.items():
        eid = f"ep_{int(i) + 1:03d}" if isinstance(i, (int, np.integer)) or str(i).isdigit() else f"ep_{i}"

        parts.append({
        "type": "text",
        "text": f"\n=== EPISODE {eid} ==="
        })

        parts.append({
        "type": "text",
        "text": f"[EP={eid} SECTION=input IDX=0 TYPE=img ASSET=image LABEL=image media_type=image/png]"
        })

        parts.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_pair[0]}"}
        })

        parts.append({
        "type": "text",
        "text": f"[EP={eid} SECTION=output IDX=0 TYPE=text ASSET=label LABEL=label]"
        })

        parts.append({
        "type": "text",
        "text": str(img_pair[1])
        })
    return parts


def _normalize_hypotheses(hypotheses):
    normalized = []
    for h in hypotheses or []:
        if isinstance(h, dict):
            text = str(h.get("hypothesis", "")).strip()
            if text:
                normalized.append({
                    "type": h.get("type", "new"),
                    "original": h.get("original", ""),
                    "hypothesis": text,
                    "evidences": h.get("evidences", [])
                })
        elif isinstance(h, str) and h.strip():
            normalized.append({"type": "new", "hypothesis": h.strip(), "evidences": []})
    return normalized

def _normalize_factors(factors):
    normalized = []
    seen = set()
    for factor in factors or []:
        text = str(factor).strip()
        key = text.lower()
        if text and key not in seen:
            normalized.append(text)
            seen.add(key)
    return normalized


def extract_factors_for_subquestion(sub_question, image_label_pairs, prev_factors=None):
    # Factor extraction stage from hypthesis_generator.py, adapted to Gemini inlineData.
    prev_factors = _normalize_factors(prev_factors or [])
    prev_factors_text = ", ".join(prev_factors) if prev_factors else "N/A"

    prompt = f"""
You are an AI that extracts important features/factors for deriving output labels from input images
from multiple episodes (input image -> output label examples).

# Atomic sub-question / missing concept to explain
{sub_question}

# Input labels
1: image

# Output labels
1: label

# Task
predict label from image, focusing on the atomic sub-question above

# Previously extracted factors
{prev_factors_text}

# Important multimodal input reference rules
- Images are provided as inlineData in the message content.
- Reference tags in the text correspond to the image that immediately follows them.
- Always match reference tags with images and extract objective facts.

# Rules to follow
- Objectively observe each episode and identify important features from the relationship between input and output.
- Extract factors that help answer the atomic sub-question and derive the output label from the image.
- Express each factor as a single word or in the form "Y of X" such as "weight of item".
- Factors should be abstract features behind observed facts, not the facts themselves.
- If images are included, analyze visual features such as color, shape, arrangement, material, texture, size, count, and spatial relation when relevant.
- List each factor individually; do not combine them.
- Only add new factors that do not overlap with existing ones; merge similar factors.

# Examples of appropriate factors
- Size
- Color
- Shape
- Material
- Count of objects
- Position of object
- Texture
- Weight of item

# Examples of inappropriate factors
- There is something red in the image
- Heavy items are on top
- A person is riding

# Output format: JSON only
{{
  "new_factors": ["factor1", "factor2", "factor3"]
}}

Now extract factors step by step.
"""
    parts = _build_hypothesis_episode_parts(image_label_pairs)
    parts.append({
    "type": "text",
    "text": prompt
    })
    
    response = _call_gpt_parts(parts)
    print(response, "\n")
    result = safe_extract_json(response)
    new_factors = _normalize_factors(result.get("new_factors", []))
    merged = _normalize_factors(prev_factors + new_factors)
    return merged, new_factors


def generate_hypotheses_for_subquestion(sub_question, image_label_pairs, factors=None, prev_hypotheses=None, validation_log=None):
    # Generate hypotheses using extracted factors as axes. 
    prev_hypotheses = _normalize_hypotheses(prev_hypotheses or [])
    prev_hypotheses_text = "\n".join([
        f"{i+1}. {h.get('hypothesis', h)}" for i, h in enumerate(prev_hypotheses)
    ]) if prev_hypotheses else "N/A"
    validation_text = validation_log if validation_log else "N/A"
    factors = _normalize_factors(factors or [])
    factors_text = ", ".join(factors) if factors else "N/A"

    prompt = f"""
You are an AI that extracts hypotheses/rules for deriving output labels from input images
using multiple episodes (input image -> output label examples).

# Atomic sub-question / missing concept to explain
{sub_question}

# Input labels
1: image

# Output labels
1: label

# Existing hypotheses
{prev_hypotheses_text}

# Previous validation results
{validation_text}

# Extracted factors (axes for hypothesis generation)
{factors_text}

# Important multimodal input reference rules
- Images are provided as inlineData in the message content.
- Reference tags in the text correspond to the image that immediately follows them.
- Always match reference tags with images and extract objective facts.

# Rules to follow
- Do not create hypotheses immediately. First extract objective facts from each episode.
- Prioritize observed facts from the given episodes over common sense.
- Generate hypotheses that answer the atomic sub-question and predict output labels from input images based on the extracted factors.
- Hypotheses should be concise, executable rules, preferably in the form \"if X then Y\".
- For each hypothesis, list supporting episode_id values as evidences.
- Prefer modifying/merging existing hypotheses over proliferating similar hypotheses.
- Review invalidated hypotheses and revise them.

# Output format: JSON only
{{
  "reason": ["..."],
  "hypotheses": [
    {{
      "type": "keep|modify|new",
      "original": "only when type is modify",
      "hypothesis": "concise executable hypothesis",
      "evidences": ["ep_001", "ep_002"]
    }}
  ]
}}

Now generate hypotheses step by step.
"""
    parts = _build_hypothesis_episode_parts(image_label_pairs)
    parts.append({
    "type": "text",
    "text": prompt
    })
    response = _call_gpt_parts(parts)
    print(response, "\n")
    result = safe_extract_json(response)
    return _normalize_hypotheses(result.get("hypotheses", []))


def validate_hypotheses_for_subquestion(sub_question, image_label_pairs, hypotheses):
    # Validate generated hypotheses against all episodes and keep only valid ones.
    hypotheses = _normalize_hypotheses(hypotheses)
    if not hypotheses:
        return [], "No hypotheses"

    hypotheses_text = "\n".join([
        f"{i}. {h.get('hypothesis', h)}" for i, h in enumerate(hypotheses)
    ])

    prompt = f"""
You are an AI that validates whether generated hypotheses function correctly across all episodes.

# Atomic sub-question / missing concept
{sub_question}

# Hypotheses to validate
{hypotheses_text}

Important: use the exact zero-based hypothesis_index shown above. For example, the first hypothesis has hypothesis_index 0.

# Validation rules
- A hypothesis is valid only if it is consistent with the facts of ALL episodes.
- A hypothesis is invalid if any of the following apply:
  - It contradicts any episode.
  - It is inconsistent overall.
  - It contains reasoning based on unstated, irrelevant, or common-sense-only.
  - Conditions are joined by \"or\" and only one condition is actually supported.
- First verify objective facts from each episode.
- Avoid judgments based on common sense; prioritize observed facts from the given episodes.
- Make strict binary judgments. Partial correctness is invalid.

# Output format: JSON only
{{
  "validations": [
    {{
      "hypothesis_index": 0,
      "hypothesis": "hypothesis text",
      "is_valid": true,
      "reason": "validation reason"
    }}
  ]
}}

Now validate the hypotheses step by step.
"""
    parts = _build_hypothesis_episode_parts(image_label_pairs)
    parts.append({
    "type": "text",
    "text": prompt
    })
    response = _call_gpt_parts(parts)
    print(response, "\n")
    result = safe_extract_json(response)

    validations = result.get("validations", [])
    valid_indices = set()
    for v in validations:
        idx = v.get("hypothesis_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if v.get("is_valid", False):
            valid_indices.add(idx)

    valid_hypotheses = [h for i, h in enumerate(hypotheses) if i in valid_indices]
    validation_log = "\n".join([
        f"{v.get('hypothesis', '')} => {'valid' if v.get('is_valid', False) else 'invalid'} ({v.get('reason', '')})"
        for v in validations
    ])

    if validations:
        return valid_hypotheses, validation_log
    return hypotheses, "Validation skipped"


def call_llm_hypothesis(sub_ques, image_label_pairs, num_iterations=3, num_factor_iterations=1, max_empty_hypothesis_retries=2):
    # Three-stage process: extract factors -> generate hypotheses -> validate hypotheses.
    # If the model returns {"hypotheses": []}, repeat hypothesis generation before validation.
    current_factors = []
    current_hypotheses = []
    validation_log = None

    for factor_iteration in range(num_factor_iterations):
        print(f"Factor iteration {factor_iteration + 1}: extracting factors for: {sub_ques}\n")
        current_factors, new_factors = extract_factors_for_subquestion(
            sub_question=sub_ques,
            image_label_pairs=image_label_pairs,
            prev_factors=current_factors,
        )
        print(f"Factor iteration {factor_iteration + 1}: new factors: {new_factors}\n")

    for iteration in range(num_iterations):
        generated_hypotheses = []

        for retry_idx in range(max_empty_hypothesis_retries):
            print(
                f"Hypothesis iteration {iteration + 1}, generation attempt {retry_idx + 1}: "
                f"generating hypotheses for: {sub_ques}\n"
            )
            generated_hypotheses = generate_hypotheses_for_subquestion(
                sub_question=sub_ques,
                image_label_pairs=image_label_pairs,
                factors=current_factors,
                prev_hypotheses=current_hypotheses,
                validation_log=validation_log,
            )
            print(
                f"Hypothesis iteration {iteration + 1}, generation attempt {retry_idx + 1}: "
                f"generated {len(generated_hypotheses)} hypotheses\n"
            )

            if generated_hypotheses:
                break

            print(
                f"WARNING: Hypothesis iteration {iteration + 1}, generation attempt {retry_idx + 1}: "
                "empty hypotheses returned; repeating generation.\n"
            )

        if not generated_hypotheses:
            print(
                f"WARNING: Hypothesis iteration {iteration + 1}: no hypotheses generated after "
                f"{max_empty_hypothesis_retries} attempts; keeping previous hypotheses.\n"
            )
            generated_hypotheses = current_hypotheses

        current_hypotheses, validation_log = validate_hypotheses_for_subquestion(
            sub_question=sub_ques,
            image_label_pairs=image_label_pairs,
            hypotheses=generated_hypotheses,
        )
        print(f"Hypothesis iteration {iteration + 1}: {len(current_hypotheses)} hypotheses validated\n")

    factors_summary = ", ".join(current_factors)
    hypothesis_summary = " ".join([h["hypothesis"] for h in current_hypotheses])
    if factors_summary:
        semantic_summary = f"Relevant factors: {factors_summary}. Validated hypotheses: {hypothesis_summary}"
    else:
        semantic_summary = hypothesis_summary

    return json.dumps({
        "factors": current_factors,
        "hypothesis": [h["hypothesis"] for h in current_hypotheses],
        "hypothesis_summary": semantic_summary,
        "validation_log": validation_log or ""
    }, ensure_ascii=False)







def call_llm_final(target_question, semantic_rule, image_label_pairs, target_encoded_img):
    ENDPOINT = "ENTER YOUR ENDPOINT HERE"
    API_KEY = os.environ.get("WRITE_API_KEY")
    headers = {
        "Content-type": "application/json",
        "api-key": API_KEY
    }

    base_prompt = f"""You are an expert visual reasoning model.
You are given a context and target question with target image to answer.

Context: 
{semantic_rule}
"""

    output_prompt = """
Output format:
````json
{{
  "answer": [<object_descriptions>],
  "total": <int_total>,
  "reasoning": <short_reasoning>,
  "uncertain": <yes/no>
}}
````
"""
    parts = []
    parts.append({
    "type": "text",
    "text": base_prompt
    })

    parts.append({
    "type": "text",
    "text": f"Target Question: {target_question}\n"
    })

    parts.append({
    "type": "text",
    "text": "Target Image:"
    })

    parts.append({
    "type": "image_url",
    "image_url": {"url": f"data:image/png;base64,{target_encoded_img}"}
    })

    parts.append({
    "type": "text",
    "text": output_prompt
    })

    # print(parts)

    messages = [{ "role": "user", "content": parts}]
    body = {"messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(ENDPOINT, headers=headers, json=body, timeout=120)

            # print("BODY:", response.text[:1000])

            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.warning(f"[Attempt {attempt}] LLM error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(BASE_SLEEP_SECONDS * 2 ** (attempt - 1))
            else:
                logger.error("All retries failed.")
    return ""








def verify_prompt(target_question, model_reasoning):
    return f"""Target Question: {target_question}
Reasoning: {model_reasoning}

Did the reasoning assume any missing definition or rule not stated in Target Question? Answer only JSON:
{{"assumed_definition":"yes/no","why":"short"}}"""



# ---------- PROMPTS ---------- #
def form_prompt(target_question):

    prompt = f"""You are an expert visual reasoning model.
You are given a image and a target question to answer.

# Target Question:
{target_question}

If the question cannot be answered directly, say there is not enough context and describe what is missing.

# Output Format:
````json
{{
  "answer": [<object_descriptions>],
  "total": <int_total>,
  "reasoning": <short_reasoning>,
  "uncertain": <yes/no>
}}
````
"""
    return prompt





# def subquestion_prompt(target_question, prev_reasoning):
#     prompt = f"""You previously could not answer the target question completely.

# Your task is to generate a atomic sub-question needed to recover the missing concept required to answer the target question.

# IMPORTANT RULES:
# - The sub-question must ask about EXACTLY ONE attribute.
# - Ask only about attributes that is required by the target question.
# - Do not ask multiple subquestions if only one attribute is missing.
# - Do NOT ask about the target image contents.
# - Do NOT ask which objects are present.
# - Do NOT ask about specific object instances.
# - Do NOT ask about complementary labels unless explicitly required by the target question.
# - Prefer general definition/property question over dataset-specific wording.

# Your output should contain only the atomic missing attributes needed to answer the target question.

# # Target Question:
# {target_question}

# # # Previous Reasoning:
# # {prev_reasoning}

# Output format:
# ```json
# {{
#   "sub_question": ["<definition question 1>"]
# }}
# ```
# """
#     return prompt


def subquestion_prompt(target_question, prev_reasoning):
    prompt = f"""You previously could not answer the target question completely.

Your task is to generate the MINIMAL set of atomic sub-questions needed to recover the missing attributes required to answer the target question.

IMPORTANT RULES:
- Each sub-question must ask about EXACTLY ONE attribute.
- Ask only about attributes that are required by the target question.
- Do not ask multiple subquestions if only one attribute is missing.
- Do NOT ask about the target image contents.
- Do NOT ask which objects are present.
- Do NOT ask about specific object instances.
- Do NOT ask about combinations of attributes in one question.
- Do NOT generate paraphrases or duplicate questions.
- Do NOT ask about complementary labels unless explicitly required by the target question.
- Prefer general definition/property questions over dataset-specific wording.

Your output should contain only the atomic missing attributes needed to answer the target question.

# Target Question:
{target_question}

# # Previous Reasoning:
# {prev_reasoning}

Output format:
```json
{{
  "sub_questions": ["<definition question 1>", "<definition question 2>", ...]
}}
````
"""
    return prompt






def merge_semantic_rules(sub_questions, semantic_rules):
    merged_parts = []
    for sq, rule in zip(sub_questions, semantic_rules):
        if rule and rule.strip():
            # merged_parts.append(f"Sub-question: {sq}\nRule: {rule}")
            merged_parts.append(rule)
    # return "\n\n".join(merged_parts)
    return " ".join(merged_parts)




def get_semantic_rule_for_subquestion(sub_question, image_label_pairs, clarification_memory, clarification_embedding, threshold):
    clarification_key = " ".join(sub_question.lower().split())

    # 1) Exact reuse
    if clarification_key in clarification_memory:
        print(f"🔁 Reusing cached semantic rule for: {sub_question}\n")
        return clarification_memory[clarification_key]

    # 2) Embed new question
    sub_question_embedding = get_embedding(sub_question)
    best_match_key = None
    best_score = -1.0

    # 3) Compare with existing embeddings
    for existing_key, existing_emb in clarification_embedding.items():
        score = cosine_similarity(sub_question_embedding, existing_emb)
        if score > best_score:
            best_score = score
            best_match_key = existing_key

    # 4) Reuse if similar enough
    if best_match_key is not None and best_score >= threshold and best_match_key in clarification_memory:
        print(f"🔁 Reusing semantic rule from similar question "
              f"(score={best_score:.3f}) for: {sub_question}\n")
        semantic_rule = clarification_memory[best_match_key]

        clarification_memory[clarification_key] = semantic_rule
        clarification_embedding[clarification_key] = sub_question_embedding
        return semantic_rule

    # 5) Otherwise generate new
    print(f"✨ Generating new semantic rule for: {sub_question}\n")
    hypo_response = call_llm_hypothesis(sub_question, image_label_pairs)
    print(hypo_response, "\n")
    hypo_json = safe_extract_json(hypo_response)

    semantic_rule = hypo_json.get("hypothesis_summary", "")
    clarification_memory[clarification_key] = semantic_rule
    clarification_embedding[clarification_key] = sub_question_embedding

    return semantic_rule





def safe_extract_json(input_string):
    result = utils.extract_json_from_string(input_string)
    if not result:
        try:
            match = re.search(r"\{.*\}", input_string, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
        except Exception:
            result = {}
    return result or {}



def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def get_embedding(text):
    return sentence_transformer_model.encode(text)

THRESHOLD = 0.80
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- MAIN LOOP ---------- #

def test_socratic_query(data_root="dataset"):
    train_csv = os.path.join(data_root, "train/multi_label_train.csv")
    test_csv = os.path.join(data_root, "test/multi_label_test.csv")
    train_img_dir = os.path.join(data_root, "train/images")
    test_img_dir = os.path.join(data_root, "test/images")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)



    final_sum = 0
    total_ques = 0
    clarification_memory = {}
    clarification_embedding = {}

    # image_label_pairs = {}
    # for idx, tr in train_df.iterrows():
    #     img_path = os.path.join(train_img_dir, tr["filename"])
    #     episode_encoded_img = encode_image_base64(img_path)
    #     image_label_pairs[idx] = [episode_encoded_img, 'Anomaly' if tr['is_anomaly'] else 'Normal']

    image_label_pairs = {}
    for idx, tr in train_df.iterrows():
        img_path = os.path.join(train_img_dir, tr["filename"])
        encoded_img = encode_image_base64(img_path)
        anomaly_text = "anomaly" if tr["is_anomaly"] else "normal"
        weight_text = "heavy" if tr["is_heavy"] else "light"
        hard_text = "hard" if tr["is_hard"] else "soft"
        price_text = str(int(tr["price"]))
        formatted_label = f"{anomaly_text}, {weight_text}, {hard_text}, price={price_text}"
        image_label_pairs[idx] = [encoded_img, formatted_label]



    # print("Evidences:\n",evidences,"\n")

    for idx, row in test_df.iterrows():


        total_ques += 1

        max_rounds = 3
        round_idx = 0
        semantic_rule = ""

        query_file = row["filename"]
        query_text = row["textual_feedback"]
        ground_truth_answer = row["ground_truth_answer"]
        correct_context = row["reasoning"]
        background = row["background"]

        print(f"\n=== Test Episode {idx+1}: {query_file} ===")
        print(f"Test Question: {query_text}\n")

        target_question = row["question"]

        # --- Encode query image ---
        img_path = os.path.join(test_img_dir, query_file)
        target_encoded_img = encode_image_base64(img_path)

        # Ask the target question
        prompt_1 = form_prompt(target_question)
        print(prompt_1)

        response_1 = call_llm([prompt_1], [target_encoded_img])
        print(response_1)

        json_1 = safe_extract_json(response_1)

        answer = json_1.get("answer", [])
        total = json_1.get("total", None)



        reasoning = json_1.get("reasoning", "")

        uncertain_flag = str(json_1.get("uncertain", "")).strip().lower() in ["yes", "true"]


        if not uncertain_flag:
            #Extra lines added for more safety to check if the model is making wrong assumptions.
            verify_resp = call_llm([verify_prompt(target_question, reasoning)], [target_encoded_img])
            verify_json = safe_extract_json(verify_resp) or {}
            assumed = str(verify_json.get("assumed_definition","")).strip().lower() in ["yes","true"]
            if assumed:
                print("⚠️ Verifier: model assumed a missing definition → starting clarification loop.\n")
                uncertain_flag = True
        
            else:
                print("Model is confident and did not assume missing definitions — skipping clarification loop.\n")


        while uncertain_flag and round_idx < max_rounds:
            round_idx += 1
            print("Model uncertain — generating sub-question.\n")

            subq_prompt = subquestion_prompt(target_question, reasoning)
            print(subq_prompt)
            subq_response = call_llm([subq_prompt], [])
            print(subq_response)
            subq_json = safe_extract_json(subq_response)

            sub_questions = subq_json.get("sub_questions", "")
            # sub_questions = subq_json.get("sub_question", "")

            for sq_idx, subquestion in enumerate(sub_questions): 
                print(f"{sq_idx}: Sub-question: {subquestion}\n")


            all_semantic_rules = []

            for sub_question in sub_questions:
                one_rule = get_semantic_rule_for_subquestion(sub_question=sub_question, image_label_pairs=image_label_pairs, clarification_memory=clarification_memory, clarification_embedding=clarification_embedding, threshold=THRESHOLD)
                if one_rule:
                    all_semantic_rules.append(one_rule)

            semantic_rule = merge_semantic_rules(sub_questions, all_semantic_rules)

            print("Merged semantic rule:\n", semantic_rule, "\n")

            response_2 = call_llm_final(target_question, semantic_rule, image_label_pairs, target_encoded_img)
            print(response_2)
            
            json_2 = safe_extract_json(response_2)

            answer = json_2.get("answer", [])
            total = json_2.get("total", None)

            reasoning = json_2.get("reasoning", "")
            uncertain_flag = str(json_2.get("uncertain", "")).strip().lower() in ["yes", "true"]


            if not uncertain_flag:
                print("Model confident — stopping clarification loop.\n")
                break

            else:
                print("Continue asking clarification question")
                

        predicted_answer = safe_int(total)

        if predicted_answer is None:
            predicted_answer = 0

        print(f"Predicted Objects: {answer}")

        print(f"Predicted Answer: {predicted_answer}")
        print(f"Predicted Reasoning: {reasoning}\n")

        print(f"Ground Truth Answer: {ground_truth_answer}")
        print(f"Ground Truth Rules: {correct_context}\n")


        if str(predicted_answer) == str(ground_truth_answer):
            print("✅ Correct reasoning achieved!")
            final_sum = final_sum + 1
        else:
            print("❌ Still incorrect — more evidence may be needed.")

        log_data = {
            "filename": query_file,
            "ground_truth_answer": ground_truth_answer,
            "predicted_answer": predicted_answer,
            "accuracy": 1 if str(predicted_answer) == str(ground_truth_answer) else 0,
            "answer": answer,
            "reasoning": reasoning,
            "semantic_rule": semantic_rule if semantic_rule else "",
        }

        log_path = "ours_multi_label.csv"

        # Append or create CSV file
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_data)

    print("Total test questions = ", total_ques)
    print("Accuracy = ",final_sum / total_ques)


if __name__ == "__main__":
    test_socratic_query()
