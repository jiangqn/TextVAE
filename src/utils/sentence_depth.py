# import stanza
#
# nlp = stanza.Pipeline('en')
#
# def dfs(graph, root):
#     if len(graph[root]) == 0:
#         return 1
#     else:
#         return max([dfs(graph, child) for child in graph[root]]) + 1
#
# def sentence_depth(sentence):
#     sentence = nlp(sentence)
#     graph = [[] for _ in range(len(sentence.sentences[0].words) + 1)]
#     for word in sentence.sentences[0].words:
#         id = int(word.id)
#         head = int(word.head)
#         graph[head].append(id)
#     return dfs(graph, 0)

import spacy
nlp = spacy.load('en')

def dfs(graph: dict, root: spacy.tokens.token.Token) -> int:
    if not root in graph:
        return 1
    else:
        return max([dfs(graph, node) for node in graph[root]]) + 1

def sentence_depth(sentence: str) -> int:
    doc = nlp(sentence)
    sentence = [s for s in doc.sents][0]
    graph = {}
    root = None
    for word in sentence:
        head = word.head
        if head == word:
            root = head
        else:
            if not head in graph:
                graph[head] = []
            graph[head].append(word)
    return dfs(graph, root)