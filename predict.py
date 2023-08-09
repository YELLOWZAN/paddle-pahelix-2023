import numpy as np
import pandas as pds
import json
import sys
import pickle
from threading import Thread, Lock

import paddle as pd
import paddle.nn as nn
from rdkit.Chem import AllChem
import pgl
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'env/PaddleHelix/'))

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d


mutex = Lock()


# TODO 1: 测试集数据预处理
def transfer_smiles_to_graph_(compound_name_to_smiles_dict):
    n = len(compound_name_to_smiles_dict)
    global p
    index = 0
    while True:
        mutex.acquire()
        if p >= n:
            mutex.release()
            break
        index = p
        p += 1
        mutex.release()

        compound_name = list(compound_name_to_smiles_dict.keys())[index]
        smiles = compound_name_to_smiles_dict[compound_name]
        mutex.acquire()
        print(index, ':', round(index/n*100, 2),'%', smiles)
        mutex.release()
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)
        except:
            print("Invalid smiles!", compound_name,smiles)
            continue

        global compound_name_to_graph_dict
        mutex.acquire()
        compound_name_to_graph_dict[compound_name] = molecule_graph
        mutex.release()


def data_preprocess(test_csv):
    df = pds.read_csv(test_csv)
    CID = [cid for cid in range(len(df))]
    SMILES = df['SMILES'].to_list()

    compound_name_to_smiles_dict = dict(zip(CID, SMILES))

    global compound_name_to_graph_dict
    compound_name_to_graph_dict = {}
    global p
    p = 0
    thread_count = 4
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=transfer_smiles_to_graph_, args=(compound_name_to_smiles_dict,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data_list = []
    for compound_name in compound_name_to_graph_dict.keys():
        data_item = {}
        data_item['compound_name'] = compound_name
        data_item['graph'] = compound_name_to_graph_dict[compound_name]
        data_item['class'] = 0
        data_list.append(data_item)

    pickle.dump(data_list, open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'class_data_list.pkl'), 'wb'))

# TODO 2: 预测测试集
class DownstreamCollateFn(object):
    """CollateFn for downstream model"""
    def __init__(self, 
            atom_names, 
            bond_names, 
            bond_float_names,
            bond_angle_float_names,
            task_type,
            is_inference=False):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.task_type = task_type
        self.is_inference = is_inference

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])
    
    def __call__(self, data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        compound_class_list=[]
        for data in data_list:
            compound_class_list.append(data['class'])
            data=data['graph']
            ab_g = pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)

        return atom_bond_graph, bond_angle_graph, np.array(compound_class_list,dtype=np.float32)


def get_data_loader():
    bs = 256
    test_data_list = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'class_data_list.pkl'), 'rb'))
    shuffle_ = False

    test_dataset = InMemoryDataset(test_data_list)
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    collate_fn = DownstreamCollateFn(
            atom_names=atom_names, 
            bond_names=bond_names,
            bond_float_names=bond_float_names,
            bond_angle_float_names=bond_angle_float_names,
            task_type='regr',is_inference=True)

    test_data_loader = test_dataset.get_data_loader(batch_size=bs, num_workers=1, shuffle=shuffle_, collate_fn=collate_fn)
    return test_data_loader


class RRP(nn.Layer):
    def __init__(self, encoder):
        super(RRP, self).__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 2, weight_attr=nn.initializer.KaimingNormal()),
            nn.Softmax()
        )

    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        x = graph_repr
        x = self.mlp(x)
        return x


def evaluate(model_version, result_csv):
    data_loader = get_data_loader()

    compound_encoder_config = json.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'env/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json'), 'r'))
    encoder = GeoGNNModel(compound_encoder_config)
    model = RRP(encoder=encoder)

    model.set_state_dict(pd.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/'+model_version+'.pkl')))

    model.eval()
    all_result = []
    for batch_index, (atom_bond_graph, bond_angle_graph, compound_class) in enumerate(data_loader):
        output = model(atom_bond_graph, bond_angle_graph)
        result = output.cpu().numpy()[:, 1].tolist()
        all_result.extend(result)

    pkl = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'class_data_list.pkl'), 'rb'))
    CIDs = []
    for i in pkl:
        CIDs.append(i['compound_name'])
    df = pds.DataFrame(data=CIDs, columns=['CID'])
    df['pred'] = all_result
    df_sorted = df.sort_values(by='CID')
    df_sorted.to_csv(result_csv, index=False)


if __name__ == '__main__':
    test_csv = sys.argv[1]  # 测试集路径
    result_csv = sys.argv[2]  # 结果文件路径

    data_preprocess(test_csv)  # 处理测试集数据
    evaluate(model_version='gem', result_csv=result_csv)  # 预测测试集
