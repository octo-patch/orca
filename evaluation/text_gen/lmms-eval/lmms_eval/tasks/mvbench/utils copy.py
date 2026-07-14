import os
import re
import string
from pathlib import Path

import PIL
import yaml
from loguru import logger as eval_logger

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

LOCAL_DATA_LIST = {
    "object_interaction": "star/Charades_v1_480",
    "action_sequence": "star/Charades_v1_480",
    "action_prediction": "star/Charades_v1_480",
    "action_localization": "sta/sta_video",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)
DEFAULT_VIDEO_ROOT = Path(os.getenv("MVBENCH_VIDEO_ROOT", "your_data_path")).expanduser()

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

template_config = yaml.safe_load("".join(safe_data))
cache_name = ((template_config.get("dataset_kwargs") or {}).get("cache_dir"))


def _normalize_video_root(root: Path) -> Path:
    root = root.expanduser()
    if (root / "video").exists():
        return root / "video"
    return root


def _candidate_roots():
    roots = []
    candidate_roots = [_normalize_video_root(DEFAULT_VIDEO_ROOT)]
    if cache_name:
        candidate_roots.append(Path(base_cache_dir) / cache_name)

    for root in candidate_roots:
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def _candidate_dataset_folders(sub_task: str):
    # The official MVBench dump and the HF-cache layout use slightly different folder names.
    folders = []
    for folder in (LOCAL_DATA_LIST[sub_task], DATA_LIST[sub_task]):
        if folder not in folders:
            folders.append(folder)

    for folder in list(folders):
        if folder.startswith(("clevrer/", "star/")):
            alt_folder = f"data0613/{folder}"
            if alt_folder not in folders:
                folders.append(alt_folder)

    return folders


def _resolve_visual_path(doc, sub_task: str) -> str:
    for root in _candidate_roots():
        for dataset_folder in _candidate_dataset_folders(sub_task):
            candidate = root / dataset_folder / doc["video"]
            if candidate.exists():
                return str(candidate)
            # Some official MVBench video segment files append timestamp suffixes
            # (for example `clip_xx_start_end.mp4`) while the JSON keeps only the clip stem.
            parent = root / dataset_folder
            if parent.exists() and "." not in doc["video"]:
                matches = sorted(parent.glob(f"{doc['video']}_*"))
                file_matches = [path for path in matches if path.is_file()]
                if file_matches:
                    # Some TVQA clip ids map to multiple timestamped segments in the
                    # local dump. Pick the first sorted file deterministically so
                    # evaluation can proceed without falling back to a non-existent stem.
                    return str(file_matches[0])

    searched_paths = [str(root / dataset_folder / doc["video"]) for root in _candidate_roots() for dataset_folder in _candidate_dataset_folders(sub_task)]
    eval_logger.error(f"Video path does not exist for sub_task={sub_task}, video={doc['video']}. Checked: {searched_paths}")
    return searched_paths[0]


def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = _resolve_visual_path(doc, lmms_eval_specific_kwargs["sub_task"])
    if "start" in doc and "end" in doc:
        video_start = float(doc["start"])
        video_end = float(doc["end"])
        if video_end < video_start:
            video_end = video_start
        return [
            {
                "type": "video",
                "url": video_path,
                "video_start": video_start,
                "video_end": video_end,
            }
        ]
    return [video_path]


def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = _resolve_visual_path(doc, lmms_eval_specific_kwargs["sub_task"])
    frame_path_list = sorted(os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png"))
    sample_frames = (lmms_eval_specific_kwargs or {}).get("sample_frames", 32)
    if sample_frames and len(frame_path_list) > sample_frames:
        sample_indices = [round(i * (len(frame_path_list) - 1) / (sample_frames - 1)) for i in range(sample_frames)] if sample_frames > 1 else [0]
        frame_path_list = [frame_path_list[idx] for idx in sample_indices]
    frame_image_list = [PIL.Image.open(frame_path).convert("RGB") for frame_path in frame_path_list]
    return frame_image_list


def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"

    full_text = "Question:" + doc["question"] + "\nOption:\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
    return full_text


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def mvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    pred = results[0]

    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc["candidates"]):
        if candidate == doc["answer"]:
            gt_option_letter = option_letters[i]
            break

    # Calculate the score using mcq_acc function
    score = mcq_acc(gt_option_letter, pred)

    data_dict = {"pred_answer": pred, "gt_answer": gt_option_letter, "score": score}

    return {"mvbench_accuracy": data_dict}


def mvbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]

    return 100 * total_correct / total_answered if total_answered > 0 else 0
