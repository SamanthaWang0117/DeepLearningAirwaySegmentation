import os
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import skfmm
# import torch
import skimage
from joblib.numpy_pickle_utils import xrange
import networkx as nx
# from torchvision.utils import save_image
from tqdm import tqdm 

def vgn_feature(prob_map,delta,geo_dist_thresh,min_coord=None,max_coord=None):
    vesselness = np.squeeze(prob_map)

    # Note: the prob_map has shape [n_ch, n_feat, y, x], where n_ch=1 and n_feat=1 and y and x are the image
    # y and x coordinates, respectively
    im_y = vesselness.shape[-1]
    im_x = vesselness.shape[-2]
    y_quan = range(0,im_y,delta)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,delta)
    x_quan = sorted(list(set(x_quan) | set([im_x])))

    max_val = []
    max_pos = []

    for yi in xrange(len(y_quan)-1):
        for xi in xrange(len(x_quan)-1):
            cur_patch = vesselness[y_quan[yi]:y_quan[yi+1], x_quan[xi]:x_quan[xi+1]]
            tmp_max_val = np.amax(cur_patch)
            if tmp_max_val < 0.1:
                max_val.append(0)
                max_pos.append((y_quan[yi]+cur_patch.shape[0]/2,x_quan[xi]+cur_patch.shape[1]/2))
            else:

                max_val.append(np.amax(cur_patch))
                result = np.asarray(np.where(cur_patch == np.amax(cur_patch)))
                max_pos.append((int(y_quan[yi] + result[0, 0]), int(x_quan[xi] + result[1, 0])))
    graph = nx.Graph()

    # add nodes
    nodeidlist=[]
    for node_idx, (node_y,node_x) in enumerate(max_pos):
        nodeidlist.append(node_idx)
        graph.add_node(node_idx,kind='MP',y=node_y,x=node_x,label=node_idx)

    speed = vesselness
    speed = numpy.squeeze(speed)
    speed[ speed < 0.00001 ] = 0.00001
    nodelist = graph.nodes
    for i,n in enumerate(nodelist):
        phi = np.ones_like(speed)
        y_coord = int(graph.nodes[i]['y'])
        x_coord = int(graph.nodes[i]['x'])
        if min_coord is not None:
            if (y_coord < min_coord) or (x_coord < min_coord):
                continue
        if max_coord is not None:
            if (y_coord > max_coord) or (x_coord > max_coord):
                continue
        phi[y_coord, x_coord] = -1
        if speed[y_coord,x_coord] == 0:
            continue
        neig = speed[max(0,y_coord-1):min(yi,y_coord+2),\
                    max(0,x_coord-1):min(xi,x_coord+2)]
        if np.mean(neig)<0.1:
            continue
        tt = skfmm.travel_time(phi,speed,narrow=geo_dist_thresh)
        for n_id in nodeidlist[i + 1:]:
            n_comp = nodelist[n_id]
            geo_dist = tt[int(n_comp['y']), int(n_comp['x'])]  # travel time
            if geo_dist < geo_dist_thresh:
                ### Let's not use the weight= option, to keep things simple ###
                graph.add_edge(n, n_id)
    return graph

