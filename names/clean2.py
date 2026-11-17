import json,re,sys,collections,random
d=json.load(open(sys.argv[1]))
print(collections.Counter(v.get('mode') for k,v in d.items() if isinstance(v,dict)).most_common(12))
names=set()
for k,v in d.items():
    if k=='sample_spec' or not isinstance(v,dict) or v.get('mode') not in ('chat','responses','completion'): continue
    n=k.split('/')[-1].lower().strip()
    n=re.sub(r'[:@].*$','',n).replace('_','-')
    n=re.sub(r'^(ft:|us\.|eu\.|apac\.|global\.|au\.|jp\.)','',n)
    n=re.sub(r'^(anthropic|amazon|meta|mistral|cohere|ai21|openai|deepseek|google|qwen|writer|moonshot|minimax|zai|nvidia|twelvelabs|stability|luma)\.','',n)  # cloud vendor prefixes
    n=re.sub(r'-\d{4}-\d{2}-\d{2}$','',n); n=re.sub(r'-\d{8}$','',n); n=re.sub(r'-(\d{4})$',lambda m:'' if m.group(1)[:2] in('01','02','03','04','05','06','07','08','09','10','11','12') else m.group(0),n)
    n=re.sub(r'-v\d+$','',n)
    if not re.fullmatch(r'[a-z0-9.\-]{4,24}',n): continue
    if re.search(r'\d{5,}',n): continue
    names.add(n)
names=sorted(names); open('names/chatnames.txt','w').write('\n'.join(names)+'\n')
print(len(names)); random.seed(2); print(random.sample(names,50))
