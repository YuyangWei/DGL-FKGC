import json

def get_json(path):
    json_data = {}
    with open(path,'r',encoding='utf-8')as fp:
        json_data = json.load(fp)
    return json_data   

def element_replace_point(dic):
    replaced_dic = {}
    for k,v in dic.items():
        k_replace = k.replace("/", ":")
        replaced_dic[k_replace] = v
    return replaced_dic

def write_json(path, dic):
    json_str = json.dumps(dic)
    with open(path, 'w') as json_file:
        json_file.write(json_str)

def main(path_dir, write_dir):
    id2real_path = path_dir + "FB2RealName.json"
    id2real = get_json(id2real_path)

    replace_id2real = element_replace_point(id2real)

    write_json(write_dir+"FB2RealName.json", replace_id2real)

if __name__ == '__main__':
    path_dir = "./FB15k-One/"
    write_dir = "./FB15k-One/"
    main(path_dir, write_dir)