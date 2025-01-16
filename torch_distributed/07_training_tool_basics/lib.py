import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Mapping, List, Union, Dict, Any, Tuple
import json
import torch
import pytrec_eval
import tqdm

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')


@dataclass
class ScoredDoc:
    qid: str
    pid: str
    rank: int
    score: float = field(default=-1)

def normalize_qa_text(text: str) -> str:
    # TriviaQA has some weird formats
    # For example: """What breakfast food gets its name from the German word for """"stirrup""""?"""
    while text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    return text


def get_question_key(question: str) -> str:
    # For QA dataset, we'll use normalized question strings as dict key
    return question

def load_query_answers(path: str) -> Dict[str, Dict[str, Any]]:
    assert path.endswith('.tsv')

    qid_to_query = {}
    for line in open(path, 'r', encoding='utf-8'):
        query, answers = line.strip().split('\t')
        query = normalize_qa_text(query)
        answers = normalize_qa_text(answers)
        qid = get_question_key(query)
        if qid in qid_to_query:
            print('Duplicate question: {} vs {}'.format(query, qid_to_query[qid]['query']))
            continue

        qid_to_query[qid] = {}
        qid_to_query[qid]['query'] = query
        qid_to_query[qid]['answers'] = list(eval(answers))

    print('Load {} queries from {}'.format(len(qid_to_query), path))
    return qid_to_query

def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    assert path.endswith('.txt')

    # qid -> pid -> score
    qrels = {}
    for line in open(path, 'r', encoding='utf-8'):
        qid, _, pid, score = line.strip().split('\t')
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][pid] = int(score)

    print('Load {} queries {} qrels from {}'.format(len(qrels), sum(len(v) for v in qrels.values()), path))
    return qrels


def load_queries(path: str, task_type: str = 'ir') -> Dict[str, str]:
    assert path.endswith('.tsv')

    if task_type == 'qa':
        qid_to_query = load_query_answers(path)
        qid_to_query = {k: v['query'] for k, v in qid_to_query.items()}
    elif task_type == 'ir':
        qid_to_query = {}
        for line in open(path, 'r', encoding='utf-8'):
            qid, query = line.strip().split('\t')
            qid_to_query[qid] = query
    else:
        raise ValueError('Unknown task type: {}'.format(task_type))

    print('Load {} queries from {}'.format(len(qid_to_query), path))
    return qid_to_query

def trec_eval(qrels: Dict[str, Dict[str, int]],
              predictions: Dict[str, List[ScoredDoc]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    results: Dict[str, Dict[str, float]] = {}
    for query_id, scored_docs in predictions.items():
        results.update({query_id: {sd.pid: sd.score for sd in scored_docs}})

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics

def get_rel_threshold(qrels: Dict[str, Dict[str, int]]) -> int:
    # For ms-marco passage ranking, score >= 1 is relevant
    # for trec dl 2019 & 2020, score >= 2 is relevant
    rel_labels = set()
    for q_id in qrels:
        for doc_id, label in qrels[q_id].items():
            rel_labels.add(label)

    print('relevance labels: {}'.format(rel_labels))
    return 2 if max(rel_labels) >= 3 else 1


def compute_mrr(qrels: Dict[str, Dict[str, int]],
                predictions: Dict[str, List[ScoredDoc]],
                k: int = 10) -> float:
    threshold = get_rel_threshold(qrels)
    mrr = 0
    for qid in qrels:
        scored_docs = predictions.get(qid, [])
        for idx, scored_doc in enumerate(scored_docs[:k]):
            if scored_doc.pid in qrels[qid] and qrels[qid][scored_doc.pid] >= threshold:
                mrr += 1 / (idx + 1)
                break

    return round(mrr / len(qrels) * 100, 4)

def load_msmarco_predictions(path: str) -> Dict[str, List[ScoredDoc]]:
    assert path.endswith('.txt')

    qid_to_scored_doc = {}
    for line in tqdm.tqdm(open(path, 'r', encoding='utf-8'), desc='load prediction', mininterval=3):
        fs = line.strip().split('\t')
        qid, pid, rank = fs[:3]
        rank = int(rank)
        score = round(1 / rank, 4) if len(fs) == 3 else float(fs[3])

        if qid not in qid_to_scored_doc:
            qid_to_scored_doc[qid] = []
        scored_doc = ScoredDoc(qid=qid, pid=pid, rank=rank, score=score)
        qid_to_scored_doc[qid].append(scored_doc)

    qid_to_scored_doc = {qid: sorted(scored_docs, key=lambda sd: sd.rank)
                         for qid, scored_docs in qid_to_scored_doc.items()}

    print('Load {} query predictions from {}'.format(len(qid_to_scored_doc), path))
    return qid_to_scored_doc

def save_preds_to_msmarco_format(preds: Dict[str, List[ScoredDoc]], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as writer:
        for qid in preds:
            for idx, scored_doc in enumerate(preds[qid]):
                writer.write('{}\t{}\t{}\t{}\n'.format(qid, scored_doc.pid, idx + 1, round(scored_doc.score, 3)))
    print('Successfully saved to {}'.format(out_path))