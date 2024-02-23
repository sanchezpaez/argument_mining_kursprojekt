

def check_claim_verbs(word: str) -> tuple[str, str]:
    claim_verbs = ['argue', 'claim', 'emphasise', 'contend', 'maintain', 'assert', 'theorize', 'support the view that', 'deny', 'negate',
                   'refute', 'reject', 'challenge', 'strongly believe that', 'counter the view that' 'argument that',
                   'acknowledge', 'consider', 'discover', 'hypothesize', 'object', 'say', 'admit', 'assume', 'decide',
                   'doubt', 'imply', 'observe', 'show', 'agree', 'believe', 'demonstrate', 'emphasize', 'indicate',
                   'point out', 'state', 'allege']
    if word in claim_verbs:
        print(word)
        return word, "claim"
    return word, "no"

