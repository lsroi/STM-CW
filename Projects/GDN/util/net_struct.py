import glob


def get_feature_map(dataset):
    """
        读取指定数据集的特征列表文件，返回特征名称列表
    """
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list
# graph is 'fully-connect'
def get_fc_graph_struc(dataset):
    """
        构建全连接图结构（每个节点与其他所有节点相连）
        构建全连接图，适用于无先验知识的场景。
    """
    feature_file = open(f'./data/{dataset}/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    
    return struc_map

def get_prior_graph_struc(dataset):
    """
        基于领域知识构建特定数据集的图结构（非全连接）
        WADI 数据集：若两个特征名称的第一个字符相同（如1_Pressure和1_Temp），则相连。
        SWAT 数据集：若两个特征名称的倒数第三个字符相同（如FIT101和LIT101中的1），则相连。
        基于领域知识构建稀疏图，可减少计算量并提高模型性能
    """
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    
    return struc_map


if __name__ == '__main__':
    get_graph_struc()
 