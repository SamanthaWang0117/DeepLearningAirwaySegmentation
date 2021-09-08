import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

from model import infer_module, GraphAttentionLayer, vggNet
from vertex_sampling_and_edge import vgn_feature
from load_air_data import Datasets


def plot_edges_on_prob_map(base_dir, prob_map, graph, fig_suffix=""):
    # Code to plot prob_map, drawing arrows where connected nodes are in the image
    fig = plt.figure(figsize=(10,10),dpi=300)
    im = plt.imshow(np.squeeze(prob_map.detach().numpy()))
    fig.colorbar(im)
    nodelist = graph.nodes
    num_edges = 0
    print("plotting nodes on probability map..")
    for (n,data) in graph.nodes(data=True):
        plt.plot(int(data['x']), int(data['y']),color='w',marker='o', markersize=0.5)
    print("plotting edges on probability map..")
    for (i,n) in graph.adjacency():
        if len(n) > 0:
            tmp_node1 = nodelist[i]
            x1 = tmp_node1['x']
            y1 = tmp_node1['y']
            for i_node in n.keys():
                tmp_node2 = nodelist[i_node]
                x2 = tmp_node2['x']
                y2 = tmp_node2['y']
                plt.arrow(x1,y1,x2-x1,y2-y1,color='w',width=1e-4)
                num_edges += 1
    print(str(num_edges) + " edges plotted on prob_map")
    annot_prob_map_path = os.path.join(base_dir,"Example_prob_map_edges" + fig_suffix +  ".png")
    plt.savefig(annot_prob_map_path)
    print("Annotated prob_map saved to file: " + annot_prob_map_path)

# Base directory
base_dir = "C:\\Users\\eyjolfur\\Documents\\UCL_MScProject2021_Offline\\Samantha_DeepLearningAirwaySegmentation-main\\samantha_vgn"

# We save the graphs for all training images to a file in the training directory
graph_dir = os.path.join(base_dir,"DRIVE_2","training")
graph_file = os.path.join(graph_dir, "graph_training.pickle")
# We use the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vggNet(num_classes = 1)
criterion = nn.BCELoss()
data_path = os.path.join(base_dir,"DRIVE_2","training")
delta = 16
dist_thresh = 50
print(data_path)
train_data = Datasets(path=data_path)

# We set batch_size to 1 to make things simpler when applying the model/creating a graph for a single image
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

# Load best model - the name of your model might be "best_model.pth"
best_model_path = os.path.join(base_dir,"best_model_testdice.pth")
model.load_state_dict(torch.load(best_model_path))
model = model.eval()
print("Model loaded.")

# We create vertices and edges for im based on prob_map (converted to a Numpy array)
# We store the graphs in a dict, whose keys will be the name of each image
graphs_dict = {}

if not os.path.exists(graph_file):
    print("Creating graphs for training images..")
    # Load images and labels and create a graph for each image
    for i, train_input in enumerate(train_loader):
        # Apply the pre-trained CNN model to the example image to get the probability map, intermediate CNN features,
        # and the concatenated feature matrix of the CNN module
        prob_map, cnn_feat, concat_feat = model(train_input['img'])
        prob_map = torch.nn.Sigmoid()(prob_map)

        print("Image index: " + str(i))
        print("Creating graph with delta=" + str(delta) + " and dist_thresh=" + str(dist_thresh))
        # We set min_coord=50 and max_coord=462 to ignore any vertices in the outermost 50 pixels of the image (any
        # vertices with x or y coordinate less than min_coord are ignored when analysing connectiveness in the graph, and
        # same for vertices with x or y coordinate greater than max_coord - this can save time during graph construction)
        tmp_graph = vgn_feature(prob_map.detach().numpy(), delta, dist_thresh, min_coord=50, max_coord=462)
        # Print out number of nodes and edges
        print("Graph created.")
        print("no. of nodes: " + str(tmp_graph.number_of_nodes()))
        print("no. of edges: " + str(tmp_graph.number_of_edges()))
        graphs_dict[train_input['img_name'][0]] = tmp_graph
        #plot_edges_on_prob_map(base_dir, prob_map, tmp_graph, fig_suffix="_training_" + str(i))


    # Save the graphs list to file
    with open(graph_file, 'wb') as f:
        pickle.dump(graphs_dict, f)
    print("Graphs created for training dataset and saved to file: " + graph_file)
    # We use an example graph to visualize the results
    graph = tmp_graph
else:
    with open(graph_file, 'rb') as f:
        graphs_dict = pickle.load(f)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    train_input = next(iter(train_loader))
    graph = graphs_dict[train_input['img_name'][0]]
    prob_map, cnn_feat, concat_feat = model(train_input['img'][0].unsqueeze(0))
    prob_map = torch.nn.Sigmoid()(prob_map)

# Print out the max value of train_input['img']
print("Max value of image: " + str(np.max(train_input['img'][0].detach().cpu().numpy())))
print("Min value of image: " + str(np.min(train_input['img'][0].detach().cpu().numpy())))




# We save the graphs for all training images to a file in the training directory
graph_dir = os.path.join(base_dir,"DRIVE_2","test")
graph_file = os.path.join(graph_dir, "graph_test.pickle")
data_path = os.path.join(base_dir,"DRIVE_2","test")
print(data_path)
test_data = Datasets(path=data_path)

# We set batch_size to 1 to make things simpler when applying the model/creating a graph for a single image
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

# We create vertices and edges for im based on prob_map (converted to a Numpy array)
# We store the graphs in a dict, whose keys will be the name of each image
graphs_dict = {}

if not os.path.exists(graph_file):
    print("Creating graphs for training images..")
    # Load images and labels and create a graph for each image
    for i, test_input in enumerate(test_loader):
        # Apply the pre-trained CNN model to the example image to get the probability map, intermediate CNN features,
        # and the concatenated feature matrix of the CNN module
        prob_map, cnn_feat, concat_feat = model(test_input['img'])
        prob_map = torch.nn.Sigmoid()(prob_map)

        print("Image index: " + str(i))
        print("Creating graph with delta=" + str(delta) + " and dist_thresh=" + str(dist_thresh))
        # We set min_coord=50 and max_coord=462 to ignore any vertices in the outermost 50 pixels of the image (any
        # vertices with x or y coordinate less than min_coord are ignored when analysing connectiveness in the graph, and
        # same for vertices with x or y coordinate greater than max_coord - this can save time during graph construction)
        tmp_graph = vgn_feature(prob_map.detach().numpy(), delta, dist_thresh, min_coord=50, max_coord=462)
        # Print out number of nodes and edges
        print("Graph created.")
        print("no. of nodes: " + str(tmp_graph.number_of_nodes()))
        print("no. of edges: " + str(tmp_graph.number_of_edges()))
        graphs_dict[test_input['img_name'][0]] = tmp_graph
        #plot_edges_on_prob_map(base_dir, prob_map, tmp_graph, fig_suffix="_test_" + str(i))

    # Save the graphs list to file
    with open(graph_file, 'wb') as f:
        pickle.dump(graphs_dict, f)
    print("Graphs created for test dataset and saved to file: " + graph_file)
    # We use an example graph to visualize the results
    graph = tmp_graph
else:
    with open(graph_file, 'rb') as f:
        graphs_dict = pickle.load(f)
    test_input = next(iter(test_loader))
    graph = graphs_dict[train_input['img_name'][0]]
    prob_map, cnn_feat, concat_feat = model(train_input['img'][0].unsqueeze(0))
    prob_map = torch.nn.Sigmoid()(prob_map)

# Print out the max value of train_input['img']
print("Max value of image: " + str(np.max(train_input['img'][0].detach().cpu().numpy())))
print("Min value of image: " + str(np.min(train_input['img'][0].detach().cpu().numpy())))


