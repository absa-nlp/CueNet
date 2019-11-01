# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:32:55 2018

@author: NLP
"""

from nltk.parse.stanford import StanfordDependencyParser
import os
#from graphviz import Source
import numpy as np

os.environ["PATH"] += os.pathsep + 'E:/release/bin/'
os.environ['STANFORD_PARSER'] = 'E:/stanford-parser-full-2018-02-27/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'E:/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
#java_path = "C:\Program Files\Java\jdk1.8.0_144\bin\java.exe"
#os.environ['JAVAHOME'] = java_path
#sent = 'it took about 1/2 hours to be served our 2 courses . '
#maxlen = 80

parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def gramer_dependency(sent, maxlen):
    res = list(parser.raw_parse(sent))
#    for row in res[0].triples():
#        print(row)
    dep_tree_dot_repr = [parse for parse in res][0].to_dot()
    
    tree = dep_tree_dot_repr.split('\n')
    for t in tree:
        if t.find('aspect_term') != -1:
            tar_pos = int(t.split(' ')[0])
    
    graph_pos = np.zeros((maxlen,)) + 5
    graph_pos[tar_pos-1] =0
    
    now_pos = [[] for i in range(6)]
    for i in range(6):
        if i == 0:
            now_pos[i].append(tar_pos)
        for t in tree[6:-1]:
            if t.find('->') != -1:
                ts = t.split(' ')
                if int(ts[0]) in now_pos[i]:
                    if graph_pos[int(ts[2])-1] > i+1: 
                        graph_pos[int(ts[2])-1] = i+1
                        if i < 5:
                            now_pos[i+1].append(int(ts[2]))
                if int(ts[2]) in now_pos[i]:
                    if graph_pos[int(ts[0])-1] > i+1: 
                        graph_pos[int(ts[0])-1] = i+1
                        if i< 5:
                            now_pos[i+1].append(int(ts[0]))
    max_key = 0
    word_dict = res[0].nodes
    for t in word_dict:
        if t> max_key:
            max_key = t
    con_updata = ['' for i in range(max_key+1)]
    for key in word_dict:
        if key != 0:
            con_updata[word_dict[key]['address']-1] = word_dict[key]['word']
    
    puct_list = [',','.',':','?','\'','"',';','!','(',')','#','+','-']
    punctuation = [t for t in sent.split(' ') if t in puct_list]
    count = 0
    for i,t in enumerate(con_updata):
        if t in puct_list:
            count += 1
        if t == '':
            try:
                con_updata[i] = punctuation[count]
                count += 1
            except:
                con_updata[i] = '.'
    return graph_pos,con_updata
  
#source = Source(dep_tree_dot_repr, filename="dep_tree", format="png",engine='dot')
#source.view()   
def graph_dist(context, maxlen):
    graph_dis = np.zeros((len(context),maxlen))
    new_context = []
    for i,t in enumerate(context):
        if i % 100 == 0:
            print('i=',i)
        sent = ' '.join(t)
        tar_pos = t.index('aspect_term')
        try:
            graph_pos,con_updata = gramer_dependency(sent, maxlen)
            new_context.append(con_updata)
            graph_dis[i,:] = graph_pos
        except:
            print(i)
            new_context.append(t)
            graph_pos = np.zeros((maxlen,)) + 5
            graph_pos[:len(t)] = 1
            graph_pos[tar_pos] = 0
            graph_dis[i,:] = graph_pos
    return graph_dis,new_context

