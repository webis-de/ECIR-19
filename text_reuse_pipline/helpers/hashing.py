def flip(bit):
    assert bit == "0" or bit == "1"
    return "0" if bit == "1" else "1"

def binlist_to_bstring(blist):
    c = list(map(lambda x: str(x), blist))
    s = ''.join(c)
    return s

def binlist_to_int(blist):
    c = list(map(lambda x: str(x), blist))
    s = ''.join(c)
    return int(s,2)

def H(s, k):
    # invariant: H yields `len(s)`-bit strings that have k-bits flipped
    if len(s) < k:
        return  # produce nothing; can't flip k bits
    if k == 0:
        yield s  # only one n-bit string has Hamming distance 0 from s (itself)
    else:
        for s_k_minus_one_flipped in H(s[:-1], k - 1):
            yield s_k_minus_one_flipped + flip(s[-1])  # flip last bit
        for s_k_flipped in H(s[:-1], k):
            yield s_k_flipped + s[-1]  # don't flip last bit
            
def expand_hash(h, distance=0):
    if distance == 0:
        return [binlist_to_int(h)]
    else:
        neighboring_hashes = list(H(h, distance))
        return map(lambda h: binlist_to_int(h), neighboring_hashes + [h])