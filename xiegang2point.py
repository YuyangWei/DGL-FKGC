#!/usr/bin/python3

import json

def read_triples(path):
    triples = []
    with open(path) as fin:
        for line in fin:
            triple = []
            h, r, t = line.strip().split('\t')
            triple.append(h)
            triple.append(r)
            triple.append(t)
            triples.append(triple)
    return triples

def get_json(path):
    json_data = {}
    with open(path,'r',encoding='utf-8')as fp:
        json_data = json.load(fp)
    return json_data   

def task_replace_point(dic):
    replaced_dic = {}
    for k,v in dic.items():
        k_replace = k.replace("/", ":")
        triples_replace = []
        for triple in v:
            triple_replace = []
            triple_replace.append(triple[0].replace("/", ":"))
            triple_replace.append(triple[1].replace("/", ":"))
            triple_replace.append(triple[2].replace("/", ":"))
            triples_replace.append(triple_replace)
        replaced_dic[k_replace] = triples_replace
    return replaced_dic

def element_replace_point(dic):
    replaced_dic = {}
    for k,v in dic.items():
        k_replace = k.replace("/", ":")
        replaced_dic[k_replace] = v
    return replaced_dic

def rel2candidates_replace_point(dic):
    replaced_dic = {}
    for k,v in dic.items():
        k_replace = k.replace("/", ":")
        element_replace = []
        for element in v:
            element_replace.append(element.replace("/", ":"))
        replaced_dic[k_replace] = element_replace
    return replaced_dic

def e1rel_e2_replace_point(dic):
    replaced_dic = {}
    for k,v in dic.items():
        k_replace = k.replace("/", ":")
        element_replace = []
        for element in v:
            element_replace.append(element.replace("/", ":"))
        replaced_dic[k_replace] = element_replace
    return replaced_dic

def graph_replace_point(graph):
    replaced_graph = []
    for triple in graph:
        replaced_triple = []
        replaced_triple.append(triple[0].replace("/", ":"))
        replaced_triple.append(triple[1].replace("/", ":"))
        replaced_triple.append(triple[2].replace("/", ":"))
        replaced_graph.append(replaced_triple)
    return replaced_graph

def write_json(path, dic):
    json_str = json.dumps(dic)
    with open(path, 'w') as json_file:
        json_file.write(json_str)

def write_triples(path, triples):
    with open(path, 'w') as file:
        for triple in triples:
            file.write(triple[0])
            file.write("\t")
            file.write(triple[1])
            file.write("\t")
            file.write(triple[2])
            file.write("\n")

def main(path_dir, write_dir):
    train_path = path_dir + "train_tasks.json"
    test_path = path_dir + "test_tasks.json"
    dev_path = path_dir + "dev_tasks.json"    
    ent_path = path_dir + "ent2ids"
    rel_path = path_dir + "relation2ids"
    path_graph = path_dir + "path_graph"
    rel2candidates_path = path_dir + "rel2candidates.json"
    rel2candidates_all_path = path_dir + "rel2candidates_all.json"
    e1rel_e2_path = path_dir + "e1rel_e2.json"
    known_rels_path = path_dir + "known_rels.json"

    train_task = get_json(train_path)
    test_task = get_json(test_path)
    dev_task = get_json(dev_path)  
    known_rels = get_json(known_rels_path) 
    ent = get_json(ent_path)  
    rel = get_json(rel_path)  
    graph = read_triples(path_graph)  
    rel2candidates = get_json(rel2candidates_path)  
    rel2candidates_all = get_json(rel2candidates_all_path)  
    e1rel_e2 = get_json(e1rel_e2_path)  

    #print("rel",rel)
    #print("train_task",train_task.keys())    

    train_task = task_replace_point(train_task)
    test_task = task_replace_point(test_task)
    dev_task = task_replace_point(dev_task)
    known_rels = task_replace_point(known_rels)

    ent2id = element_replace_point(ent)
    rel2id = element_replace_point(rel)

    graph = graph_replace_point(graph)
   
    rel2candidates = rel2candidates_replace_point(rel2candidates)
    rel2candidates_all = rel2candidates_replace_point(rel2candidates)
    e1rel_e2 = e1rel_e2_replace_point(e1rel_e2)
    

    write_train_path = write_dir + "train_tasks.json"
    write_test_path = write_dir + "test_tasks.json"
    write_dev_path = write_dir + "dev_tasks.json"    
    write_ent_path = write_dir + "ent2ids"
    write_rel_path = write_dir + "relation2ids"
    write_path_graph = write_dir + "path_graph"
    write_rel2candidates_path = write_dir + "rel2candidates.json"
    write_rel2candidates_all_path = write_dir + "rel2candidates_all.json"
    write_e1rel_e2_path = write_dir + "e1rel_e2.json"
    write_known_rels_path = write_dir + "known_rels.json"
   
    write_json(write_train_path,train_task)
    write_json(write_test_path,test_task)
    write_json(write_dev_path,dev_task)
    write_json(write_ent_path,ent2id)
    write_json(write_rel_path,rel2id)
    write_triples(write_path_graph,graph)
    write_json(write_rel2candidates_path,rel2candidates)
    write_json(write_rel2candidates_all_path,rel2candidates_all)
    write_json(write_e1rel_e2_path,e1rel_e2)
    write_json(write_known_rels_path,known_rels)

if __name__ == '__main__':
    path_dir = "./FB15k-One/"
    write_dir = "./FB15k/"
    main(path_dir, write_dir)
