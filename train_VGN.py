from copy import copy
import numpy
import skfmm
from matplotlib import pyplot as plt
from load_air_data import Datasets
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision.utils import save_image
import os
import networkx as nx
import pickle
from joblib.numpy_pickle_utils import xrange
import numpy as np
from model import GraphAttentionLayer, infer_module, vggNet
from copy import deepcopy

def dice_coef(y, y_pred):
    gt = np.ndarray.flatten(copy(y))
    pred = np.ndarray.flatten((copy(y_pred)))
    return (2 * np.sum(gt * pred)) / (np.sum(gt) + np.sum(pred))


def create_gnn_feats(concat_feat, graph, nfeat, delta=16):
    adj_matrix = nx.adjacency_matrix(graph)
    im_y = concat_feat.shape[-2]
    im_x = concat_feat.shape[-1]
    y_quan = range(0, im_y, delta)
    x_quan = range(0, im_x, delta)
    # For the GAT layer, we create an input feature matrix of shape [n_vertices, nfeat]
    gnn_feat = np.zeros((adj_matrix.shape[-1], nfeat))
    node_id = 0
    nodelist = graph.nodes
    for yi in xrange(len(y_quan) - 1):
        for xi in xrange(len(x_quan) - 1):
            tmp_node = nodelist[node_id]
            gnn_feat[node_id, :] = concat_feat[:, int(tmp_node['y']), int(tmp_node['x'])]
            node_id += 1
    return gnn_feat


def extract_vertex_gt_label(graph, gt_label):
    # Extract probabilities at vertices' locations, based on ground-truth label.
    #
    # Inputs:
    # graph [networkx.Graph]: Graph for the current image.
    # gt_label [numpy array]: Ground truth labels for image.
    im_y = concat_feat.shape[-2]
    im_x = concat_feat.shape[-1]
    y_quan = range(0, im_y, delta)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0, im_x, delta)
    x_quan = sorted(list(set(x_quan) | set([im_x])))
    gnn_node_prob = np.zeros((1, len(y_quan) - 1, len(x_quan) - 1))
    nodelist = graph.nodes
    node_id = 0
    for yi in xrange(len(y_quan) - 1):
        for xi in xrange(len(x_quan) - 1):
            tmp_node = nodelist[node_id]
            gnn_node_prob[:, yi, xi] = np.mean(gt_label[0, int(tmp_node['y']), int(tmp_node['x'])])
            node_id += 1

    return gnn_node_prob


if __name__ == "__main__":
    # We created the graphs using delta=16, dist_thresh=50
    delta = 16
    dist_thresh = 50
    # Base directory
    base_dir = "/Users/samanthawang/Downloads/segmentationvgn"
    data_path = os.path.join(base_dir, "DRIVE_2/training")
    test_path = os.path.join(base_dir, "DRIVE_2/test")
    graphs_path = os.path.join(data_path, "graph_training.pickle")
    test_graphs_path = os.path.join(test_path, "graph_test.pickle")
    vgn_result = os.path.join(base_dir, "DRIVE_2/vgn_results/training")
    vgn_test_result = os.path.join(base_dir, "DRIVE_2/vgn_results/test")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # Load the graphs that have been pre-created for the training set
    with open(graphs_path, 'rb') as f:
        # The graphs are saved into a dict object, with the key being the name of the image
        graphs_dict = pickle.load(f)

    # Load the graphs that have been pre-created for the test set
    with open(test_graphs_path, 'rb') as f:
        # The graphs are saved into a dict object, with the key being the name of the image
        test_graphs_dict = pickle.load(f)

    train_data = Datasets(path=data_path)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    test_data = Datasets(path=test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # Training and model parameters
    dropout = 0.5
    alpha = 0.2
    epoch = 200
    n_classes = 1
    nfeat = 64

    # Load pre-trained CNN model
    cnn_model = vggNet(num_classes=1)
    best_model_path = os.path.join(base_dir, "best_model_testdice_Sep3.pth")
    cnn_model.load_state_dict(torch.load(best_model_path))

    # Optional freezing of CNN parameters by setting param.requires_grad = False in the below for loop
    for param in cnn_model.parameters():
        param.requires_grad = False
    cnn_model = cnn_model.to(device)
    print("Pre-trained CNN model loaded")
    # BCEWithLogitsLoss() is used for generating the output in range (0,1)
    cnn_criterion = nn.BCEWithLogitsLoss()

    gat_model = GraphAttentionLayer(nfeat, 32, dropout, alpha).to(device)
    gat_criterion = nn.BCELoss()

    infer_model = infer_module(nfeat, dropout, alpha, device, base_dir).to(device)
    
    infer_criterion = nn.BCEWithLogitsLoss()

    # We sum together the lists of the parameters of the CNN, GAT and inference modules,
    # see: https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603
    optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(gat_model.parameters())
                                 + list(infer_model.parameters()), lr=5e-5)
    best_train_loss = 100
    best_train_dice = 0
    best_test_loss = 100
    best_test_dice = 0
    mean_train_loss_list = []
    mean_train_dice_list = []
    mean_test_loss_list = []
    mean_test_dice_list = []

    for i in range(epoch):
        cnn_loss_list = []
        gat_loss_list = []
        infer_loss_list = []
        loss_list = []
        dice_train_list = []

        cnn_model = cnn_model.train()
        gat_model = gat_model.train()
        infer_model = infer_model.train()

        for i_batch, batch in enumerate(train_loader):
            # Get output from CNN, GAT and inference separately + calculate losses on each
            # sum losses and call .backward() on the summed loss
            # (see: https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603)
            img = batch['img'].type(torch.FloatTensor).to(device)
            name = batch['img_name']
            labels = batch['label'].type(torch.FloatTensor).to(device)

            # Get the CNN probability map and CNN features
            prob_map, cnn_interm_feat, concat_feat = cnn_model(img)

            # print(prob_map.shape)
            cnn_loss = cnn_criterion(prob_map, labels)
            # print(graphs_dict)
            # Acquire the input features to the GNN using the graph and the concatenated feature matrix from the CNN
            tmp_graph = graphs_dict[name[0]]
            n_vertices = tmp_graph.number_of_nodes()
            # We assume that the number of vertices is equal in the y and x dimension
            n_vertices_per_dim = np.sqrt(n_vertices).astype(int)
            # GNN takes as input a tensor of size [batch_size, n_vertices, nfeat]
            gnn_feats = create_gnn_feats(concat_feat[0].detach().cpu().numpy(), tmp_graph, nfeat)
            gnn_feats = torch.from_numpy(gnn_feats).to(device)
            adj_mat = torch.from_numpy(nx.adjacency_matrix(tmp_graph).toarray()).to(device)

            # gat_output is the 32-feature feature matrix, gat_prob is the per-vertex probability from the GAT on
            # which we calculate the loss for the GAT
            gat_output, gat_prob = gat_model(gnn_feats, adj_mat)

            ### The "tensorization" of gat_output ###
            # reshape the gat output
            gat_output = gat_output.permute(1, 0)
            # then, reshape gat_output from [out_features, n_vertices] to
            # [out_features, n_vertices_per_dim, n_vertices_per_dim]
            gat_output = torch.reshape(gat_output, (32, n_vertices_per_dim, n_vertices_per_dim))
            # finally, add extra dimension to gat_output, final shape
            # is [1, out_feat, n_vertices_per_dim, n_vertices_per_dim]
            gat_output = torch.unsqueeze(gat_output, 0)

            
            # Create ground truth vertex to calculate loss
            node_gt_prob = extract_vertex_gt_label(tmp_graph, labels[0].detach().cpu().numpy())
            node_gt_prob = torch.from_numpy(node_gt_prob).float().to(device)
            gat_loss = gat_criterion(gat_prob, node_gt_prob)

            # Apply the inference module
            infer_out = infer_model(cnn_interm_feat, gat_output)
            infer_loss = infer_criterion(infer_out, labels)

            # Sum the losses from CNN, GAT, inference modules
            sum_loss = (cnn_loss + gat_loss + infer_loss)
            optimizer.zero_grad()
            
            sum_loss.backward()
            optimizer.step()

            loss_list.append(sum_loss.item())
            cnn_loss_list.append(cnn_loss.item())
            gat_loss_list.append(gat_loss.item())
            infer_loss_list.append(infer_loss.item())

            infer_thresh = torch.sigmoid(infer_out).detach().cpu().numpy()
            infer_thresh[infer_thresh >= 0.5] = 1.
            infer_thresh[infer_thresh < 0.5] = 0.
            cnn_out_thresh = torch.sigmoid(prob_map).detach().cpu().numpy()
            cnn_out_thresh[cnn_out_thresh >= 0.5] = 1.
            cnn_out_thresh[cnn_out_thresh < 0.5] = 0.
            labels = labels.detach().cpu().numpy()
            dice_train = dice_coef(labels, infer_thresh)
            dice_train_list.append(dice_train)
            # saving graph every 150 image
            if i_batch % 150 == 0:
                print("Epoch %i of %i, training batch %i loss = %.5f" % (i, epoch, i_batch, sum_loss.item()))
                print("cnn_loss = %.5f, gat_loss = %.5f, infer_loss = %.5f" % (
                    np.mean(cnn_loss_list), np.mean(gat_loss_list), np.mean(infer_loss_list)))
                save_image(torch.sigmoid(infer_out).detach().cpu(),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_VGN_train_pred.png"), normalize=True)
                save_image(torch.from_numpy(infer_thresh),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_VGN_train_pred_thresh.png"), normalize=True)
                save_image(torch.sigmoid(prob_map).detach().cpu(),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_CNN_train_pred.png"), normalize=True)
                save_image(torch.from_numpy(cnn_out_thresh),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_CNN_train_pred_thresh.png"), normalize=True)
                save_image(torch.from_numpy(labels),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_VGN_train_gt.png"), normalize=True)
                save_image(img.detach().cpu(),
                           os.path.join(vgn_result, f"ep{i}_{i_batch}_VGN_train_im.png"), normalize=True)

        mean_train_loss_list.append(np.mean(loss_list))
        mean_train_dice_list.append(np.mean(dice_train_list))
        print(f"[{epoch + 1}, {i + 1}], training loss: {np.mean(loss_list):.4f}")
        print(f"[{epoch + 1}, {i + 1}], train Dice: {np.mean(dice_train_list):.4f}")
        
        # Testing part
        with torch.no_grad():
            test_loss_list = []
            test_dice_list = []
            cnn_model = cnn_model.eval()
            gat_model = gat_model.eval()
            infer_model = infer_model.eval()

            for i_batch, data in enumerate(test_loader):
                images = data['img']
                labels = data['label']
                images = images.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                test_prob_map, cnn_interm_feat, concat_feat = cnn_model(images)

                cnn_loss_test = cnn_criterion(test_prob_map, labels)
      
                # Acquire the input features to the GNN using the graph and the concatenated feature matrix from the CNN
                tmp_graph = graphs_dict[name[0]]
                n_vertices = tmp_graph.number_of_nodes()
                # We assume that the number of vertices is equal in the y and x dimension
                n_vertices_per_dim = np.sqrt(n_vertices).astype(int)
                # GNN takes as input a tensor of size [batch_size, n_vertices, nfeat]
                gnn_feats = create_gnn_feats(concat_feat[0].detach().cpu().numpy(), tmp_graph, nfeat)
                gnn_feats = torch.from_numpy(gnn_feats).to(device)
                adj_mat = torch.from_numpy(nx.adjacency_matrix(tmp_graph).toarray()).to(device)

                # gat_output is the 32-feature feature matrix, gat_prob is the per-vertex probability from the GAT on
                # which we calculate the loss for the GAT
                gat_output, gat_prob = gat_model(gnn_feats, adj_mat)

                ### The "tensorization" of gat_output ###
                gat_output = gat_output.permute(1, 0)
                gat_output = torch.reshape(gat_output, (32, n_vertices_per_dim, n_vertices_per_dim))
                gat_output = torch.unsqueeze(gat_output, 0)

                # We extract the ground-truth vertex vessel/airway probability at each vertex location to calculate loss
                # of the GAT model
                node_gt_prob = extract_vertex_gt_label(tmp_graph, labels[0].detach().cpu().numpy())
                node_gt_prob = torch.from_numpy(node_gt_prob).float().to(device)
                gat_loss_test = gat_criterion(gat_prob, node_gt_prob)

                # Apply the inference module
                infer_out = infer_model(cnn_interm_feat, gat_output)
                infer_loss_test = infer_criterion(infer_out, labels)

                infer_thresh = torch.sigmoid(infer_out).detach().cpu().numpy()
                infer_thresh[infer_thresh >= 0.5] = 1.
                infer_thresh[infer_thresh < 0.5] = 0.
                # Sum the losses from CNN, GAT, inference modules
                sum_loss_test = (cnn_loss_test + gat_loss_test + infer_loss_test)
                test_loss_list.append(sum_loss_test.item())
                labels = labels.detach().cpu().numpy()
                dice_test = dice_coef(labels, infer_thresh)
                test_dice_list.append(dice_test)
                # save relative graphs every 50 image
                if i_batch % 50 == 0:
                    print("Test batch %i of %i" % (i_batch, len(test_loader)))
                    save_image(torch.sigmoid(infer_out).detach().cpu(), os.path.join(vgn_test_result, f"ep{i}_{i_batch}_test_pred.png"),
                                normalize=True)
                    save_image(torch.from_numpy(infer_thresh),
                        os.path.join(vgn_test_result, f"ep{i}_{i_batch}_test_pred_thresh.png"), normalize=True)
                    save_image(torch.from_numpy(labels), os.path.join(vgn_test_result, f"ep{i}_{i_batch}_test_gt.png"),
                                normalize=True)
                    save_image(images.detach().cpu(), os.path.join(vgn_test_result, f"ep{i}_{i_batch}_test_im.png"),
                                normalize=True)
        mean_test_loss_list.append(np.mean(test_loss_list))
        mean_test_dice_list.append(np.mean(test_dice_list))

        if np.mean(test_loss_list) < best_test_loss:
            best_test_loss = np.mean(test_loss_list)
            torch.save(deepcopy(cnn_model.state_dict()), r'./best_VGN_cnn_model_testloss.pth')
            torch.save(deepcopy(gat_model.state_dict()), r'./best_VGN_gat_model_testloss.pth')
            torch.save(deepcopy(infer_model.state_dict()), r'./best_VGN_infer_model_testloss.pth')
        if np.mean(test_dice_list) > best_test_dice:
            best_test_dice = np.mean(test_dice_list)
            torch.save(deepcopy(cnn_model.state_dict()), r'./best_VGN_cnn_model_testdice.pth')
            torch.save(deepcopy(gat_model.state_dict()), r'./best_VGN_gat_model_testdice.pth')
            torch.save(deepcopy(infer_model.state_dict()), r'./best_VGN_infer_model_testdice.pth')

        print(f"[{epoch + 1}, {i + 1}], test loss: {np.mean(test_loss_list):.4f}")
        print(f"[{epoch + 1}, {i + 1}], test Dice: {np.mean(test_dice_list):.4f}")

        plt.figure(1)
        plt.plot(range(len(mean_train_loss_list)), np.array(mean_train_loss_list))
        plt.plot(range(len(mean_test_loss_list)), np.array(mean_test_loss_list))
        plt.legend(["Training", "Testing"])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(r".\VGN_Loss_vs_epochs_graph.png")
        plt.ylim((0,10))
        plt.close()

        plt.figure(2)
        plt.plot(range(len(mean_train_dice_list)), np.array(mean_train_dice_list))
        plt.plot(range(len(mean_test_dice_list)), np.array(mean_test_dice_list))
        plt.legend(["Training", "Testing"])
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.ylim((0,1))
        plt.savefig(r".\VGN_Dice_vs_epochs_graph.png")
        plt.close()
        # Save mean training/test loss/dice to .txt file
        with open(r'./VGN_Train_Loss.txt', 'w') as f:
            for item in mean_train_loss_list:
                f.write("%f\n" % item)
        with open(r'./VGN_Train_Dice.txt', 'w') as f:
            for item in mean_train_dice_list:
                f.write("%f\n" % item)
        with open(r'./VGN_Test_Loss.txt', 'w') as f:
            for item in mean_test_loss_list:
                f.write("%f\n" % item)
        with open(r'./VGN_Test_Dice.txt', 'w') as f:
            for item in mean_test_dice_list:
                f.write("%f\n" % item)
    print('Finish all')


