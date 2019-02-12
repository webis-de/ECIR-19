import re
from pyspark import SparkFiles
from text_reuse_pipline.helpers import utility

def collect_docs(p, lang_detection_model_name=None, lang='en'):
    
    if lang_detection_model_name != None:
        from pyfasttext import FastText
        model_path = SparkFiles.get(lang_detection_model_name)
        model = FastText(model_path)

    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    
    result = []
    lines = list(p)
    indices = [ i for i, line in enumerate(lines) if regex.search(line.strip()) ]
    for i in range(0, len(indices)):
        idx     = indices[i]
        content = lines[idx+1]
        paras   = re.findall('<PAR>(.*?)</PAR>', content, re.DOTALL)
        
        if model:
            #filter only english paras
            langs    = model.predict(paras)
            en_paras = list(filter(lambda p: lang in p[1], zip(paras, langs)))
            paras    = list(map(lambda pair: pair[0], en_paras))

        if paras:
            url = lines[idx].strip()
            result.append((url, paras))
    
    return result

def remove_js_codes(paragraphs):
    import re
    regex = re.compile('function (.*) {.*}')    
    return list(filter(lambda x: not regex.match(x), paragraphs))

def remove_headers(paragraphs, stopwords, puncs, min_tokens, min_stopwords):
    idx    = 0
    main_content = False
    
    def contains_punct(s, puncs):
        for p in puncs:
            if p in s:
                return True

    while not main_content and idx < len(paragraphs):
        p = paragraphs[idx]
        p_tokens = [x.lower() for x in p.split()]
        
        print(p_tokens)
        if len(p_tokens) > min_tokens and len(stopwords.intersection(set(p_tokens))) > min_stopwords and contains_punct(p, puncs):
            main_content = True
            break

        idx+=1

    return paragraphs[idx:]

def remove_footers(paragraphs, stopwords, puncs, min_tokens, min_stopwords):
    idx    = len(paragraphs) -1
    main_content = False
    
    def contains_punct(s, puncs):
        for p in puncs:
            if p in s:
                return True

    while not main_content and idx > 0:
        p = paragraphs[idx]
        p_tokens = [x.lower() for x in p.split()]
        
        if len(p_tokens) > min_tokens and len(stopwords.intersection(set(p_tokens))) > min_stopwords and contains_punct(p, puncs):
            main_content = True
            break

        idx-=1
    
    return paragraphs[0:idx]