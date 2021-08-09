import os
from datetime import time

import matplotlib
import numpy
import numpy as np
import skfmm

from joblib.numpy_pickle_utils import xrange
import networkx as nx
from torch.autograd.grad_mode import F
from torch.optim import optimizer

import model
from model import gat


def vgn_feature(prob_map,delta,geo_dist_thresh,temp_graph_path):
    seg_size_str = '%.2d_%.2d'%(delta,geo_dist_thresh)
    cur_save_gra_savepath = os.path.join(temp_graph_path,seg_size_str+'.graph_res')

    airwayness = prob_map

    im_y = airwayness.shape[0]
    im_x = airwayness.shape[1]
    y_quan = range(0,im_y,delta)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,delta)
    x_quan = sorted(list(set(x_quan) | set([im_x])))

    max_val = []
    max_pos = []

    for yi in xrange(len(y_quan)-1):
        for xi in xrange(len(x_quan)-1):
            cur_patch = airwayness[y_quan[yi]:y_quan[yi+1],x_quan[xi]:x_quan[xi+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[yi]+cur_patch.shape[0]/2,x_quan[xi]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[yi] + temp[0], x_quan[xi] + temp[1]))
    graph = nx.Graph()

    print("vertex success")

    # add nodes
    nodeidlist=[]
    for node_idx, (node_y,node_x) in enumerate(max_pos):
        nodeidlist.append(node_idx)
        graph.add_node(node_idx,kind='MP',y=node_y,x=node_x,label=node_idx)

    speed = airwayness[:,:,0]
    speed = numpy.squeeze(speed)
    speed[ speed < 0.00001 ] = 0.00001
    nodelist = graph.nodes
    for i,n in enumerate(nodelist):
        phi = np.ones_like(speed)
        phi[int(graph.nodes[i]['y']),int(graph.nodes[i]['x'])] = -1
        if speed[int(graph.nodes[i]['y']),int(graph.nodes[i]['x'])] == 0:
            continue
        neig = speed[max(0,int(graph.nodes[i]['y'])-1):min(yi,int(graph.nodes[i]['y'])+2),\
                    max(0,int(graph.nodes[i]['x'])-1):min(xi,int(graph.nodes[i]['x'])+2)]
        if np.mean(neig)<0.1:
            continue
        tt = skfmm.travel_time(phi,speed,narrow=geo_dist_thresh)
        for n_id in nodeidlist[i + 1:]:
            n_comp = nodelist[n_id]
            geo_dist = tt[int(n_comp['y']), int(n_comp['x'])]  # travel time
            if geo_dist < geo_dist_thresh:
                graph.add_edge(n, n_id, weight=geo_dist_thresh / (geo_dist_thresh + geo_dist))
    nx.write_gpickle(graph,cur_save_gra_savepath)
    print(graph)
    graph.clear()
    return max_pos

def train(epoch,fea):
    t = time.time()
    feat = vgn_feature(img, 4, 10, path)
    model.train()
    optimizer.zero_grad()
    features = fea
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()


if __name__ == '__main__':
    img = matplotlib.image.imread(r"/Users/samanthawang/Documents/segmentationvgn/DRIVE_2/train_img/8.png")
    path = r"/Users/samanthawang/Documents/segmentationvgn/DRIVE_2/new"


    print(in_feat)
    gat(in_feat,8,1,0.5,0.5,5)