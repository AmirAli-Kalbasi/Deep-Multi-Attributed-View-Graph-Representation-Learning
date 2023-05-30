import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, A_hat, X):
        return self.linear(torch.matmul(A_hat, X))

        
class MagCAE:
    def __init__(self, A, X, hidden_dim, alpha, gamma, lambda_, num_epochs, test_edges, test_edges_false):
        self.A = A
        self.X = X
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_epochs = num_epochs
        self.gcn = [GCN(X.shape[1], hidden_dim) for _ in range(X.shape[2])]
        self.view_weights = nn.Parameter(torch.ones(X.shape[2]))
        self.optimizer = optim.Adam(list(self.view_weights) + [list(gcn.parameters()) for gcn in self.gcn], lr=alpha)
        self.A_hat = self.normalize_adjacency_matrix(A)
        self.test_edges = test_edges
        self.test_edges_false = test_edges_false

    def normalize_adjacency_matrix(self, A):
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        return D_inv_sqrt @ A @ D_inv_sqrt

    def single_attributed_view_proximity(self, X_m):
        return np.exp(-self.gamma * pairwise_distances(X_m, metric='euclidean'))

    def multi_attributed_view_proximity(self, S_a):
        return np.prod(S_a, axis=0)

    def embedding_proximity(self, Z):
        return np.exp(-1/Z.shape[1] * pairwise_distances(Z, metric='euclidean'))

    def reconstruction_loss(self, A, A_prime):
        return np.sum(np.square(A - A_prime)) / (A.shape[0] ** 2)

    def similarity_loss(self, S_v, S_e):
        return np.sum(np.abs(S_v - S_e)) / (S_v.shape[0] ** 2)

    def total_loss(self, L_res, L_sim):
        return L_res + self.lambda_ * L_sim

    def train(self):
        for epoch in range(self.num_epochs):
            for m in range(self.X.shape[2]):
                X_m = self.X[:, :, m]
                S_a_m = self.single_attributed_view_proximity(X_m)
                Z_m = self.gcn[m](self.A_hat, torch.from_numpy(X_m).float())
                self.optimizer.zero_grad()
                loss = self.total_loss(self.reconstruction_loss(self.A, self.A_hat @ Z_m @ Z_m.T), self.similarity_loss(self.multi_attributed_view_proximity(S_a_m), self.embedding_proximity(Z_m)))
                loss.backward()
                self.optimizer.step()
 

    def link_prediction(self, Z, test_edges, test_edges_false):
        scores = []
        for edge in test_edges:
            score = torch.sigmoid(torch.sum(Z[edge[0]] * Z[edge[1]]))
            scores.append(score.item())
        for edge in test_edges_false:
            score = torch.sigmoid(torch.sum(Z[edge[0]] * Z[edge[1]]))
            scores.append(score.item())
        return scores

    def node_classification(self, Z, labels, train_indices, test_indices):
        clf = SVC(kernel='linear')
        clf.fit(Z[train_indices], labels[train_indices])
        predictions = clf.predict(Z[test_indices])
        cv_scores = cross_val_score(clf, Z, labels, cv=10)
        return f1_score(labels[test_indices], predictions, average='micro'), f1_score(labels[test_indices], predictions, average='macro'), np.mean(cv_scores)

    def visualize_node_embeddings(self, Z):
        tsne = TSNE(n_components=2)
        Z_2d = tsne.fit_transform(Z)
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1])
        plt.show()



    def gcn_hidden_layer_dimension_analysis(self, p_values):
        performances = []
        for p in p_values:
            self.gcn = GCN(self.X.shape[1], int(p * self.X.shape[1]))
            self.train()
            Z = self.gcn(self.A_hat, torch.from_numpy(self.X).float()).detach().numpy()
            performance = self.link_prediction(Z, self.test_edges, self.test_edges_false)
            performances.append(performance)
        return performances

    def coefficient_sensitivity_analysis(self, lambda_values):
        performances = []
        for lambda_ in lambda_values:
            self.lambda_ = lambda_
            self.train()
            Z = self.gcn(self.A_hat, torch.from_numpy(self.X).float()).detach().numpy()
            performance = self.link_prediction(Z, self.test_edges, self.test_edges_false)
            performances.append(performance)
        return performances


    def plot_performance(self, performances, parameter_values, parameter_name):
        plt.plot(parameter_values, performances)
        plt.xlabel(parameter_name)
        plt.ylabel('Performance')
        plt.show()


def load_dataset(dataset_name):
    # Load the dataset
    cites = np.loadtxt(f'{dataset_name}.cites', dtype=int)
    content = np.loadtxt(f'{dataset_name}.content', dtype=str)

    # Extract the adjacency matrix
    nodes = np.unique(cites)
    node_map = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for cite in cites:
        i, j = node_map[cite[0]], node_map[cite[1]]
        A[i, j] = 1
        A[j, i] = 1  # Assuming the graph is undirected

    # Extract the node attributes and labels
    attribute_dict = {row[0]: row[1:-1] for row in content}
    label_dict = {row[0]: row[-1] for row in content}
    X = np.array([attribute_dict[node] for node in nodes])
    labels = np.array([label_dict[node] for node in nodes])

    # Split the attributes into three views
    num_attributes = X.shape[1]
    view_size = num_attributes // 3
    X1 = X[:, :view_size]
    X2 = X[:, view_size:2*view_size]
    X3 = X[:, 2*view_size:]
    X = np.stack([X1, X2, X3], axis=2)

    # Split the dataset into training, validation, and testing sets
    train_indices, test_indices = train_test_split(np.arange(A.shape[0]), test_size=0.2)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25)

    # Generate positive and negative edges for link prediction
    positive_edges = np.array(np.nonzero(A)).T
    negative_edges = np.array(np.nonzero(A == 0)).T
    np.random.shuffle(negative_edges)
    negative_edges = negative_edges[:positive_edges.shape[0]]

    # Split the edges into training, validation, and testing sets
    train_edges, test_edges = train_test_split(positive_edges, test_size=0.2)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.25)
    train_edges_false, test_edges_false = train_test_split(negative_edges, test_size=0.2)
    train_edges_false, val_edges_false = train_test_split(train_edges_false, test_size=0.25)

    return A, X, labels, train_indices, val_indices, test_indices, train_edges, val_edges, test_edges, train_edges_false, val_edges_false, test_edges_false


import scipy.io
import os

def load_dataset2(name):
    # Construct the paths to the .mat files
    rating_path = os.path.join(name, 'rating.mat')
    trust_network_path = os.path.join(name, 'trustnetwork.mat')

    # Load the dataset
    rating = scipy.io.loadmat(rating_path)['rating']
    trust_network = scipy.io.loadmat(trust_network_path)['trustnetwork']

    # Extract the adjacency matrix from the trust network
    nodes = np.unique(trust_network)
    node_map = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for edge in trust_network:
        i, j = node_map[edge[0]], node_map[edge[1]]
        A[i, j] = 1
        A[j, i] = 1  # Assuming the graph is undirected

    # Use the ratings as node attributes
    attribute_dict = {row[0]: row[1:] for row in rating}
    X = np.array([attribute_dict.get(node, np.zeros(rating.shape[1] - 1)) for node in nodes])

    # Split the attributes into three views
    # In this case, since all our attributes are the same (ratings), all three views will be the same
    X = np.stack([X, X, X], axis=2)

    # Use the node IDs as labels because the dataset does not include any labels
    labels = nodes

    # Split the dataset into training, validation, and testing sets
    train_indices, test_indices = train_test_split(np.arange(A.shape[0]), test_size=0.2)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25)

    # Generate positive and negative edges for link prediction
    positive_edges = np.array(np.nonzero(A)).T
    negative_edges = np.array(np.nonzero(A == 0)).T
    np.random.shuffle(negative_edges)
    negative_edges = negative_edges[:positive_edges.shape[0]]

    # Split the edges into training, validation, and testing sets
    train_edges, test_edges = train_test_split(positive_edges, test_size=0.2)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.25)
    train_edges_false, test_edges_false = train_test_split(negative_edges, test_size=0.2)
    train_edges_false, val_edges_false = train_test_split(train_edges_false, test_size=0.25)

    return A, X, labels, train_indices, val_indices, test_indices, train_edges, val_edges, test_edges, train_edges_false, val_edges_false, test_edges_false



def main():
    # Load the dataset
    A, X, labels, train_indices, val_indices, test_indices, train_edges, val_edges, test_edges, train_edges_false, val_edges_false, test_edges_false = load_dataset('cora')
    
    # Initialize the MagCAE
    magcae = MagCAE(A, X, hidden_dim=64, alpha=0.01, gamma=0.1, lambda_=1, num_epochs=100, test_edges=test_edges, test_edges_false=test_edges_false)
    
    # Train the MagCAE
    magcae.train()

    # Perform link prediction
    scores = magcae.link_prediction(test_edges, test_edges_false)


    # Compute the true labels for the edges
    true_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Compute the AUC score
    auc = roc_auc_score(true_labels, scores)
    print(f'Link prediction AUC: {auc}')
    

    # Compute the Average Precision score
    ap = average_precision_score(true_labels, scores)
    print(f'Link prediction AP: {ap}')

    # Perform node classification
    micro_f1, macro_f1, cv_score = magcae.node_classification(labels, train_indices, test_indices)
    print(f'Node classification Micro-F1: {micro_f1}, Macro-F1: {macro_f1}, Cross-validation score: {cv_score}')

    # Perform GCN hidden layer dimension analysis
    p_values = [0.1, 0.2, 0.3]
    performances = magcae.gcn_hidden_layer_dimension_analysis(p_values)
    magcae.plot_performance(performances, p_values, 'p')

    # Perform coefficient sensitivity analysis
    lambda_values = [0.1, 0.5, 1, 1.5, 2]
    performances = magcae.coefficient_sensitivity_analysis(lambda_values)

    magcae.plot_performance(performances, lambda_values, 'lambda')

    # Visualize node embeddings
    magcae.visualize_node_embeddings()
    
if __name__ == "__main__":
    main()
