#!/usr/bin/env python3
"""
HireRank AI - offline candidate ranking engine for Redrob Hackathon.
CPU-only, no network, streaming JSONL ranker.

Usage:
  python src/rank_candidates.py --candidates data/candidates.jsonl --output outputs/team_xxx.csv
"""
import argparse, csv, gzip, heapq, json, math, re
from datetime import date, datetime
from pathlib import Path

TODAY = date(2026, 6, 28)

CORE_SKILLS = {
    'python': 5.0, 'embeddings': 5.0, 'embedding': 5.0, 'sentence-transformers': 4.5,
    'bge': 4.5, 'e5': 4.2, 'retrieval': 5.0, 'ranking': 5.0, 'ranker': 5.0,
    'llm': 4.0, 'llms': 4.0, 'rag': 4.2, 'vector search': 5.0, 'semantic search': 5.0,
    'faiss': 5.0, 'pinecone': 5.0, 'weaviate': 5.0, 'qdrant': 5.0, 'milvus': 5.0,
    'opensearch': 4.5, 'elasticsearch': 4.5, 'xgboost': 3.2, 'learning-to-rank': 4.0,
    'ndcg': 4.5, 'mrr': 4.2, 'map': 3.8, 'ab testing': 3.5, 'a/b testing': 3.5,
    'fine-tuning': 3.0, 'finetuning': 3.0, 'lora': 2.5, 'qlora': 2.5, 'peft': 2.5,
    'nlp': 3.2, 'machine learning': 4.0, 'ml': 3.5, 'deep learning': 2.5,
    'model deployment': 3.5, 'mlops': 3.5, 'kubeflow': 2.8, 'bentoml': 2.8,
    'fastapi': 1.8, 'flask': 1.2, 'pytorch': 2.8, 'tensorflow': 2.2,
}
PROD_TERMS = {
    'production': 4.5, 'deployed': 4.0, 'real users': 4.0, 'scale': 2.0, 'latency': 2.5,
    'monitoring': 2.2, 'evaluation': 3.5, 'offline benchmark': 3.5, 'a/b': 3.5,
    'regression': 2.8, 'index refresh': 3.0, 'drift': 3.0, 'quality': 1.5,
    'on-call': 1.2, 'owned': 1.2, 'shipped': 3.0, 'built': 1.0, 'architecture': 1.0,
}
BAD_DOMAIN_TERMS = {'marketing','accounting','civil','mechanical','customer support','sales','seo','photoshop','operations manager','hr manager'}
TITLE_POSITIVE = {'ai engineer','ml engineer','machine learning engineer','applied scientist','data scientist','search engineer','nlp engineer','backend engineer','software engineer'}
LOCATION_GOOD = {'india','pune','noida','bangalore','bengaluru','mumbai','delhi','gurgaon','hyderabad','chennai'}

TOKEN_RE = re.compile(r"[a-z0-9+#.\-/]+")
def norm(s): return (s or '').lower()
def tokens(s): return set(TOKEN_RE.findall(norm(s)))
def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))
def sigmoid(x): return 1/(1+math.exp(-x))

def open_jsonl(path):
    path = str(path)
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8')

def days_since(d):
    try:
        return max(0, (TODAY - datetime.strptime(d, '%Y-%m-%d').date()).days)
    except Exception:
        return 999

def prof_value(p):
    return {'beginner':0.35,'intermediate':0.6,'advanced':0.85,'expert':1.0}.get(norm(p),0.4)

def text_blob(c):
    prof = c.get('profile', {})
    parts = [prof.get('headline',''), prof.get('summary',''), prof.get('current_title',''), prof.get('current_industry','')]
    for h in c.get('career_history',[]):
        parts += [h.get('title',''), h.get('industry',''), h.get('description','')]
    for sk in c.get('skills',[]): parts.append(sk.get('name',''))
    for cert in c.get('certifications',[]): parts += [cert.get('name',''), cert.get('issuer','')]
    for edu in c.get('education',[]): parts += [edu.get('degree',''), edu.get('field_of_study',''), edu.get('institution','')]
    return ' '.join(str(x) for x in parts if x)

def weighted_keyword_score(blob, vocab):
    b = norm(blob)
    total = sum(vocab.values())
    hit = 0.0
    for term, wt in vocab.items():
        if term in b: hit += wt
    return clamp(hit / max(18.0, total * 0.38))

def skill_score(c):
    score = 0.0; max_possible = 0.0; names=[]
    for sk in c.get('skills',[]):
        name = norm(sk.get('name',''))
        names.append(name)
        wt = CORE_SKILLS.get(name, 0.0)
        if wt == 0:
            # partial aliases
            for term, t_weight in CORE_SKILLS.items():
                if term in name or name in term:
                    wt = max(wt, t_weight*0.75)
        if wt:
            months = min(sk.get('duration_months') or 0, 60) / 60
            endorse = min(sk.get('endorsements') or 0, 50) / 50
            score += wt * (0.60*prof_value(sk.get('proficiency')) + 0.25*months + 0.15*endorse)
        max_possible += wt
    # key combo bonuses: retrieval/ranking + vector db + python
    name_text = ' '.join(names)
    if 'python' in name_text: score += 1.5
    if any(x in name_text for x in ['faiss','pinecone','weaviate','qdrant','milvus','elasticsearch','opensearch']): score += 3.0
    if any(x in name_text for x in ['embedding','retrieval','semantic search','rag','ranking']): score += 3.0
    return clamp(score / 18.0)

def experience_score(c):
    p = c.get('profile', {})
    yrs = float(p.get('years_of_experience') or 0)
    # Ideal around 6.5-8, acceptable 4-10
    exp_fit = math.exp(-((yrs-7.0)**2)/(2*2.2**2))
    title = norm(p.get('current_title','')+' '+p.get('headline',''))
    title_fit = 0.0
    for t in TITLE_POSITIVE:
        if t in title: title_fit = max(title_fit, 1.0 if 'ai' in t or 'ml' in t or 'machine' in t else 0.65)
    seniority = 0.15 if any(x in title for x in ['senior','lead','staff','principal','founding']) else 0
    return clamp(0.65*exp_fit + 0.25*title_fit + seniority)

def production_score(c, blob):
    base = weighted_keyword_score(blob, PROD_TERMS)
    hist = c.get('career_history', [])
    current_code = 0.0
    if hist:
        h = hist[0]
        desc = norm(h.get('description','')+' '+h.get('title',''))
        if any(x in desc for x in ['built','implemented','shipped','deployed','designed','owned']): current_code += .35
        if any(x in desc for x in ['pipeline','retrieval','ranking','model','ml','embedding','search','backend','api']): current_code += .45
        if any(x in desc for x in ['manager','marketing','accountant','civil','mechanical','support']) and 'engineer' not in desc: current_code -= .2
    return clamp(0.75*base + 0.25*current_code)

def behavioral_score(c):
    s = c.get('redrob_signals', {})
    active = clamp(1 - days_since(s.get('last_active_date',''))/180)
    response = clamp(float(s.get('recruiter_response_rate') or 0))
    interview = clamp(float(s.get('interview_completion_rate') or 0))
    profile = clamp((s.get('profile_completeness_score') or 0)/100)
    github = s.get('github_activity_score', -1)
    github = 0 if github == -1 else clamp(github/100)
    notice = 1 - clamp((s.get('notice_period_days') or 180)/180)
    relocate = 1 if s.get('willing_to_relocate') else 0
    openwork = 1 if s.get('open_to_work_flag') else 0
    saved = clamp((s.get('saved_by_recruiters_30d') or 0)/20)
    verified = (0.5 if s.get('verified_email') else 0) + (0.5 if s.get('verified_phone') else 0)
    return clamp(.16*active+.15*response+.14*interview+.12*profile+.12*github+.12*notice+.08*relocate+.06*openwork+.03*saved+.02*verified)

def location_score(c):
    p = c.get('profile', {})
    text = norm(p.get('country','')+' '+p.get('location',''))
    if any(x in text for x in ['pune','noida']): return 1.0
    if any(x in text for x in LOCATION_GOOD): return .82
    if c.get('redrob_signals',{}).get('willing_to_relocate'): return .65
    return .2

def trap_penalty(c, blob):
    p = c.get('profile', {})
    yrs = float(p.get('years_of_experience') or 0)
    b = norm(blob)
    pen = 0.0
    # disqualifiers and synthetic trap patterns
    if yrs < 3.5: pen += .18
    if yrs > 11.5: pen += .12
    if any(t in norm(p.get('current_title','')) for t in BAD_DOMAIN_TERMS): pen += .18
    # keyword-stuffer: many core skills but weak profile/career descriptions
    core_hits = sum(1 for k in CORE_SKILLS if k in b)
    prod_hits = sum(1 for k in PROD_TERMS if k in b)
    if core_hits >= 12 and prod_hits <= 2: pen += .18
    if 'research' in b and not any(x in b for x in ['production','deployed','real users','shipped']): pen += .12
    if 'chatgpt' in b and not any(x in b for x in ['retrieval','ranking','embedding','production']): pen += .16
    sig = c.get('redrob_signals', {})
    if not sig.get('open_to_work_flag'): pen += .05
    if (sig.get('notice_period_days') or 0) >= 120: pen += .08
    if days_since(sig.get('last_active_date','')) > 240: pen += .08
    if (sig.get('recruiter_response_rate') or 0) < .15: pen += .05
    return clamp(pen, 0, .55)

def score_candidate(c):
    blob = text_blob(c)
    semantic = weighted_keyword_score(blob, {**CORE_SKILLS, **PROD_TERMS})
    skills = skill_score(c)
    exp = experience_score(c)
    prod = production_score(c, blob)
    beh = behavioral_score(c)
    loc = location_score(c)
    penalty = trap_penalty(c, blob)
    raw = (.24*semantic + .22*skills + .20*prod + .14*exp + .12*beh + .08*loc)
    final = clamp(raw * (1 - penalty))
    features = {'semantic':semantic,'skills':skills,'production':prod,'experience':exp,'behavior':beh,'location':loc,'penalty':penalty}
    return final, features

def best_skill_names(c):
    scored=[]
    for sk in c.get('skills',[]):
        n=sk.get('name','')
        key=norm(n)
        val=0
        for term,wt in CORE_SKILLS.items():
            if term in key or key in term: val=max(val,wt)
        if val: scored.append((val,n))
    return [x[1] for x in sorted(scored, reverse=True)[:4]]

def reason(c, features):
    p=c.get('profile',{}); s=c.get('redrob_signals',{})
    skills=best_skill_names(c)
    yrs=p.get('years_of_experience',0)
    title=p.get('current_title','candidate')
    bits=[]
    if skills: bits.append('strong match on '+', '.join(skills[:3]))
    if features['production']>.55: bits.append('profile shows production/retrieval or ranking work')
    if features['experience']>.55: bits.append(f"{yrs} years fits the senior IC range")
    if s.get('github_activity_score',-1) not in (-1,None) and s.get('github_activity_score',0)>45: bits.append('good GitHub activity')
    concerns=[]
    if s.get('notice_period_days',0)>=90: concerns.append(f"{s.get('notice_period_days')} day notice")
    if not s.get('open_to_work_flag'): concerns.append('not marked open to work')
    if features['penalty']>.2: concerns.append('possible domain/production-fit risk')
    if not bits: bits.append('adjacent technical profile with some useful signals')
    r = f"{title} with {yrs} years; " + '; '.join(bits[:3]) + '.'
    if concerns: r += ' Concern: ' + ', '.join(concerns[:2]) + '.'
    return r[:500]

def rank(candidates_path, out_path, topk=100):
    heap=[]; seen=0
    with open_jsonl(candidates_path) as f:
        for line in f:
            if not line.strip(): continue
            c=json.loads(line); seen+=1
            sc, feat=score_candidate(c)
            item=(sc, c.get('candidate_id',''), c, feat)
            if len(heap)<topk: heapq.heappush(heap, item)
            else:
                # replace if better score or deterministic ID tie
                if (sc, c.get('candidate_id','')) > (heap[0][0], heap[0][1]): heapq.heapreplace(heap, item)
    ranked=sorted(heap, key=lambda x:(-x[0], x[1]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['candidate_id','rank','score','reasoning'])
        last_score=2.0
        for i,(sc,cid,c,feat) in enumerate(ranked,1):
            display_score=round(sc,6)
            if display_score>last_score: display_score=last_score
            last_score=display_score
            w.writerow([cid,i,display_score,reason(c,feat)])
    print(f"Ranked {seen} candidates. Wrote {out_path}")

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--output', default='outputs/team_xxx.csv')
    ap.add_argument('--topk', type=int, default=100)
    args=ap.parse_args()
    rank(args.candidates, args.output, args.topk)
