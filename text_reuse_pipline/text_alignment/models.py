import json
import subprocess

class PicaPicaAligner(object):

    def __init__(self):
        self._proc = None

    def _start(self):
        #Params for configuring picapica aligner
        self._proc = subprocess.Popen(
           "java -cp picapica.jar org.picapica.textalignment.usage.SparkAligner".split(),
           stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _getProc(self):
        if self._proc is None:
            self._start()
        return self._proc

    def align(self, text1, text2):
        obj = {
            "d1" : text1.strip(),
            "d2" : text2.strip()
        }
        p = self._getProc()
        p.stdin.write(bytes(json.dumps(obj).encode('utf-8')))
        p.stdin.write(bytes('\n'.encode('utf8')))
        p.stdin.flush()
        return json.loads(p.stdout.readline().decode('utf8'))

    def finish(self):
        if self._proc is not None:
            self._proc.kill()

class DocSimilarity(object):
    def __init__(self, wiki_id, other_wiki_ids, body=None):
        self.wiki_id = wiki_id
        self.wiki_body = body
        self.similarity_map = {}

        for t in other_wiki_ids:
            sim_doc_id    = t[0]
            sim_doc_score = t[1]
            self.similarity_map[sim_doc_id] = sim_doc_score 

class DocAlignment(object):
    def __init__(self, wiki_id, wiki_text_length):
        self.wiki_id = wiki_id
        self.wiki_length = wiki_text_length
        self.alignments  = []

    def add_alignment(self, susp_id, susp_length, d_alignments):
        seeds = list(map(lambda seed : {
            'span1_start': seed['span1'][0],
            'span1_length': seed['span1'][1] - seed['span1'][0],
            'span2_start': seed['span2'][0],
            'span2_length': seed['span2'][1] - seed['span2'][0],
        }, d_alignments))

        alignment = {
            'susp_id' : susp_id,
            'length' : susp_length,
            'seed_count' : len(d_alignments),
            'seeds' : seeds
        }

        self.alignments.append(alignment)