#! /usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from anytree import NodeMixin, PostOrderIter, PreOrderIter
from copy import deepcopy
from util import ME_test, PR_test, s_softmax, calc_pair_mat
from subprocess import check_call


class OncoNode(NodeMixin):

    def __init__(self, genes=[], f=1, parent=None, children=[]):
        super().__init__()
        self.genes = genes
        self.f = f
        self.parent = parent
        self.children = children

    @property
    def s(self):
        return(len(self.genes))

    @property
    def is_root(self):
        # Is it the root?
        if self.parent is None and not(self.is_ps):
            return(True)
        else:
            return(False)

    @property
    def is_ps(self):
        # Is it the set of passengers?
        if self.parent is None and self.f == 0:
            return(True)
        else:
            return(False)

    @property
    def is_leaf(self):
        # Is it a leaf node?
        if len(self.children) == 0:
            return(True)
        else:
            return(False)

    @property
    def is_simple(self):
        # Is it a simple node (including a single gene, in the first layer, with no child)
        if self.parent is None:
            return(False)
        elif len(self.children) == 0 and self.parent.is_root and len(self.genes) == 1:
            return(True)
        else:
            return(False)


class OncoTree():

    def __init__(self, nodes, pfp, pfn, single_error=True, mut_rates=None):
        self.nodes = nodes
        tmp = []
        for node in nodes:
            tmp.extend(node.genes)
        self.genes = tmp
        self.genes.sort()
        self.n_genes = len(tmp)
        for node in self.nodes:
            if node.is_root:
                self.root = node
        self.single_error = single_error
        if self.single_error:
            self.pfp = (pfp+pfn)/2
            self.pfn = (pfp+pfn)/2
        else:
            self.pfp = pfp
            self.pfn = pfn
        if mut_rates is None:
            self.mut_rates = np.ones(self.n_genes, dtype=int)
        else:
            self.mut_rates = mut_rates

    @classmethod
    def linear_from_dataset(cls, dataset_dic, weights_dic=None, coeff=0.5, pfp=None, pfn=None, single_error=True, error_estimation=False):
        # --- Initial tree with linear structure adapted to the dataset ---
        # The driver sets have a linear structure where the avg mut freq of each driver set -
        #   - is lower than its parent by a factor less than "coeff"
        frames = [dataset_dic[key] for key in dataset_dic.keys()]
        dataset = np.concatenate(frames, axis=0)
        if weights_dic is not None:
            frames = [weights_dic[key] for key in weights_dic.keys()]
            weights = np.concatenate(frames, axis=0)
        else:
            weights = np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
        n_muts = []
        for gene in range(dataset.shape[1]):
            n_muts.append(np.sum(weights[dataset[:, gene] == 1]))
        n_muts = np.array(n_muts)
        mmg = np.argsort(n_muts)[::-1]  # Most frequently Mutated Genes, MMG
        root_node = OncoNode(genes=[], f=1)
        nodes = [root_node, OncoNode(genes=[mmg[0]], f=1, parent=root_node)]
        for i in range(1, len(n_muts)):
            if n_muts[mmg[i]] < coeff*np.mean(n_muts[nodes[-1].genes]):
                nodes.append(OncoNode(genes=[mmg[i]], parent=nodes[-1], f=1))
            else:
                nodes[-1].genes.append(mmg[i])
        # Hardcoding the pfp and pfn ratio
        if (pfp is None) or (pfn is None):
            pfp = np.min(n_muts)*0.5
            pfn = np.min(n_muts)*0.5
        prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error, mut_rates=n_muts)
        prog_mo = prog_mo.assign_f_values(dataset_dic, weights_dic)
        if error_estimation:
            prog_mo = prog_mo.assign_error_values(dataset_dic, weights_dic)
        return(prog_mo)

    @classmethod
    def star_from_dataset(cls, dataset_dic, weights_dic=None, pfp=None, pfn=None, single_error=True, error_estimation=False):
        # --- Star tree adapted to the dataset ---
        frames = [dataset_dic[key] for key in dataset_dic.keys()]
        dataset = np.concatenate(frames, axis=0)
        if weights_dic is not None:
            frames = [weights_dic[key] for key in weights_dic.keys()]
            weights = np.concatenate(frames, axis=0)
        else:
            weights = np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
        n_muts = []
        for gene in range(dataset.shape[1]):
            n_muts.append(np.sum(weights[dataset[:, gene] == 1]))
        n_muts = np.array(n_muts)
        root_node = OncoNode(genes=[], f=1)
        nodes = [root_node]
        for gene in range(dataset.shape[1]):
            nodes.append(OncoNode(parent=root_node, genes=[gene], f=0.5))
        # Hardcoding the pfp and pfn ratio
        if (pfp is None) or (pfn is None):
            pfp = np.min(n_muts)*0.5/len(dataset_dic)
            pfn = np.min(n_muts)*0.5/len(dataset_dic)
        prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error, mut_rates=n_muts)
        prog_mo = prog_mo.assign_f_values(dataset_dic, weights_dic)
        if error_estimation:
            prog_mo = prog_mo.assign_error_values(dataset_dic, weights_dic)
        return(prog_mo)

    def to_matrix(self):
        # matrix[i,i]=0
        # matrix[i,j]=1, iff, i and j are in the same node
        # matrix[i,j]=2, iff, the node including i is a descendant of the node including j
        # matrix[i,j]=3, iff, the node including i is an ancestor of the node including j
        matrix = np.zeros(shape=(self.n_genes, self.n_genes), dtype=int)
        for node in PostOrderIter(self.root):
            for i in node.genes:
                for j in node.genes:
                    matrix[i, j] = 1
                for anc_node in node.ancestors:
                    for j in anc_node.genes:
                        matrix[i, j] = 2
                        matrix[j, i] = 3
        for i in range(self.n_genes):
            matrix[i, i] = 0
        return(matrix)

    def edge_scan(self):
        # matrix[i,i]=False
        # matrix[i,j]=True, iff, the node including i is the parent of the node including j
        # Last entry is the root, as a special node!
        matrix = np.zeros(shape=(self.n_genes+1, self.n_genes+1), dtype=bool)
        for node in PostOrderIter(self.root):
            for i in node.genes:
                for j in node.parent.genes:
                    matrix[j, i] = True
        for node in self.root.children:
            for i in node.genes:
                matrix[-1,i] = True
        return(matrix)

    def weighted_compare(self, ref_mat, weights=None):
        # Compares self with a reference matrix or OncoTree object
        self_mat = self.to_matrix()
        if type(ref_mat) == OncoTree:
            ref_mat = ref_mat.to_matrix()
        if weights is None:
            weights = np.ones_like(ref_mat)
            for i in range(weights.shape[0]):
                weights[i,i]=0
            weights = weights/np.sum(weights)
        prog_TP = np.sum(weights*(ref_mat==2)*(self_mat==2))
        prog_FP = np.sum(weights*(ref_mat!=2)*(self_mat==2))
        prog_FN = np.sum(weights*(ref_mat==2)*(self_mat!=2))
        if prog_TP == 0:
            prog_precision = 0
            prog_recall = 0
            prog_f = 0
        else:
            prog_precision = prog_TP/(prog_TP+prog_FP)
            prog_recall = prog_TP/(prog_TP+prog_FN)
            prog_f=(2*prog_precision*prog_recall)/(prog_precision+prog_recall)
        mx_TP = np.sum(weights*(ref_mat==1)*(self_mat==1))
        mx_FP = np.sum(weights*(ref_mat!=1)*(self_mat==1))
        mx_FN = np.sum(weights*(ref_mat==1)*(self_mat!=1))
        if mx_TP == 0:
            mx_precision = 0
            mx_recall = 0
            mx_f = 0
        else:
            mx_precision = mx_TP/(mx_TP+mx_FP)
            mx_recall = mx_TP/(mx_TP+mx_FN)
            mx_f=(2*mx_precision*mx_recall)/(mx_precision+mx_recall)
        return(prog_precision, prog_recall, prog_f, mx_precision, mx_recall, mx_f)

    def compare(self, ref_mat):
        # Compares self with a reference matrix or OncoTree object
        self_mat = self.to_matrix()
        if type(ref_mat) == OncoTree:
            ref_mat = ref_mat.to_matrix()
        prog_TP = np.sum((ref_mat==2)*(self_mat==2))
        prog_FP = np.sum((ref_mat!=2)*(self_mat==2))
        prog_FN = np.sum((ref_mat==2)*(self_mat!=2))
        if prog_TP == 0:
            prog_precision = 0
            prog_recall = 0
            prog_f = 0
        else:
            prog_precision = prog_TP/(prog_TP+prog_FP)
            prog_recall = prog_TP/(prog_TP+prog_FN)
            prog_f=(2*prog_precision*prog_recall)/(prog_precision+prog_recall)
        mx_TP = int(0.5*np.sum((ref_mat==1)*(self_mat==1)))
        mx_FP = int(0.5*np.sum((ref_mat!=1)*(self_mat==1)))
        mx_FN = int(0.5*np.sum((ref_mat==1)*(self_mat!=1)))
        if mx_TP == 0:
            mx_precision = 0
            mx_recall = 0
            mx_f = 0
        else:
            mx_precision = mx_TP/(mx_TP+mx_FP)
            mx_recall = mx_TP/(mx_TP+mx_FN)
            mx_f=(2*mx_precision*mx_recall)/(mx_precision+mx_recall)
        return(prog_precision, prog_recall, prog_f, mx_precision, mx_recall, mx_f)

    def remove_subtree(self, node, mode="into_simple_nodes"):
        # modes: "into_simple_nodes", "spr_into_root", "spr_into_grandparent", "break_leaf", "remove_the_node_with_its_genes"
        # Note that "remove_the_node_with_its_genes" mode is only used for simulating the tumors
        if mode == "into_simple_nodes":
            for _node in PostOrderIter(node):
                for gene in _node.genes:
                    self.nodes.append(OncoNode(genes=[gene], f=0.5, parent=self.root))
                self.nodes.remove(_node)
            node.parent = None
            del(node)
        elif mode == "spr_into_root":
            node.parent = self.root
        elif mode == "spr_into_grandparent":
            if not(node.is_root):
                if not(node.parent.is_root):
                    node.parent = node.parent.parent
                else:
                    raise ValueError('Trying to SPR into grandparent failed, as the parent is root.')
            else:
                raise ValueError('Trying to SPR into grandparent failed, as the node is root.')
        elif mode == "break_leaf":
            if node.is_leaf:
                for gene in node.genes:
                    self.nodes.append(OncoNode(genes=[gene], f=0.5, parent=node.parent))
                self.nodes.remove(node)
                node.parent = None
                del(node)
        elif mode == "remove_the_node_with_its_genes": #used for synthetic data generation only
            for _node in PostOrderIter(node):
                self.nodes.remove(_node)
            node.parent = None
            del(node)
        else:
            raise ValueError('Pruning mode not supported!')
        return(self)

    def prune(self, dataset_dic, weights_dic, consider_mut_freqs=False, error_estimation=False):
        self = self.prune_by_p_values(dataset_dic, weights_dic, error_estimation=error_estimation)
        if consider_mut_freqs:
            self = self.prune_by_mut_freqs(dataset_dic, weights_dic, error_estimation=error_estimation)
        return(self)

    def prune_by_mut_freqs(self, dataset_dic, weights_dic, th_f=0.005, just_f=True, error_estimation=False):
        # th_p: threshold for prob. of driver mutation to keep the node
        #       set to pfp or pfp/100
        # th_f: threshold for f to keep the node
        change_occured = True
        pruned_tree = deepcopy(self)
        while change_occured:
            change_occured = False
            th_p = pruned_tree.pfp/100 ### could be set to pruned_tree.pfp
            nodes_to_remove = []
            for node in PostOrderIter(pruned_tree.root):
                if node.f <= th_f:
                    nodes_to_remove.append(node.genes)
                elif not(node.is_root):
                    if not just_f:
                        total_f = 1
                        for anc_node in node.ancestors:
                            total_f *= anc_node.f
                        total_f *= node.f
                        total_f *=  (self.mut_rates/np.sum(self.mut_rates)).max()
                        if total_f <= th_p:
                            nodes_to_remove.append(node.genes)
            for item in nodes_to_remove:
                g = item[0]
                for _i, _n in enumerate(pruned_tree.nodes):
                    if g in _n.genes:
                        the_node = _n
                if not(the_node.is_simple):
                    pruned_tree = pruned_tree.remove_subtree(the_node, mode="into_simple_nodes")
                    change_occured = True
            pruned_tree = pruned_tree.assign_f_values(dataset_dic, weights_dic, fine_tuning=True)
        if error_estimation:
            pruned_tree,_=pruned_tree.fit_error_params(dataset_dic, weights_dic)
        return(pruned_tree)

    def prune_by_p_values(self, dataset_dic, weights_dic, error_estimation=False):
        # Prunes based on p-values for progression
        # modes: "into_simple_nodes", "into_passengers"
        pruned_tree = deepcopy(self)
        ##### STEP 1: GET RID OF BAD ME SETS! #####
        nodes_to_remove = []
        for node in PostOrderIter(pruned_tree.root):
            if not(node.is_root):
                if len(node.genes)>1:
                    ME_score, ME_p = ME_test(node, dataset=np.concatenate([dataset_dic[key] for key in dataset_dic.keys()]))
                    if ME_score < 0:
                        nodes_to_remove.append(node.genes)
        for item in nodes_to_remove:
            g = item[0]
            for _i, _n in enumerate(pruned_tree.nodes):
                if g in _n.genes:
                    the_node = _n
            for child_node in the_node.children:
                pruned_tree = pruned_tree.remove_subtree(child_node, mode="spr_into_grandparent")
            pruned_tree = pruned_tree.remove_subtree(the_node, mode="break_leaf")
        ##### STEP 2: GET RID OF BAD PR EDGES! #####
        change_happend = True
        while change_happend:
            change_happend = False
            nodes_to_remove = []
            for node in PostOrderIter(pruned_tree.root):
                if not(node.is_root):
                    if not(node.parent.is_root):
                        PR_forward, _, _, _, BtoF = PR_test(node, dataset=np.concatenate([dataset_dic[key] for key in dataset_dic.keys()]))
                        if BtoF > 1 or PR_forward < 0:
                            nodes_to_remove.append(node.genes)
            if len(nodes_to_remove)>0:
                change_happend = True
            for item in nodes_to_remove:
                g = item[0]
                for _i, _n in enumerate(pruned_tree.nodes):
                    if g in _n.genes:
                        the_node = _n
                pruned_tree = pruned_tree.remove_subtree(the_node, mode="spr_into_grandparent")
        pruned_tree = pruned_tree.assign_f_values(dataset_dic, weights_dic, fine_tuning=True)
        if error_estimation:
            pruned_tree,_=pruned_tree.fit_error_params(dataset_dic, weights_dic)
        return(pruned_tree)

    def contraction(self, remaining_genes, dataset_dic, weights_dic, error_estimation=False):
        contracted_tree = deepcopy(self)
        for node in PostOrderIter(contracted_tree.root):
            if not(node.is_root):
                node.genes = [g for g in node.genes if g in remaining_genes]
        change_occured = True
        pruned_tree = deepcopy(contracted_tree)
        while change_occured:
            change_occured = False
            nodes_to_remove = [node for node in pruned_tree.nodes if (not(node.is_root) and len(node.genes)==0)]
            for the_node in nodes_to_remove:
                for child_node in the_node.children:
                    pruned_tree = pruned_tree.remove_subtree(child_node, mode="spr_into_grandparent")
                the_node.parent = None
                pruned_tree.nodes.remove(the_node)
                del(the_node)
                change_occured = True
        # renaming the genes from 0 to len(remaining_genes)-1, to fit with the dataset
        reversed_mapping = {
            g: i for i,g in enumerate(remaining_genes)
        }
        for node in PostOrderIter(pruned_tree.root):
            node.genes = [reversed_mapping[item] for item in node.genes]
        tmp = []
        for node in pruned_tree.nodes:
            tmp.extend(node.genes)
        pruned_tree.genes = tmp
        pruned_tree.genes.sort()
        pruned_tree = pruned_tree.assign_f_values(dataset_dic, weights_dic, fine_tuning=True)
        if error_estimation:
            pruned_tree,_=pruned_tree.fit_error_params(dataset_dic, weights_dic)
        return(pruned_tree)

    def sample_tumor(self, tnode=None, soft_nodes=None, p_soft=0.5):
        # Define soft nodes
        # use p_soft = 0 to force hard decisions everywhere leading to a single genotype
        if soft_nodes is None:
            leaf_nodes=[]
            for node in self.nodes:
                if node.is_leaf:
                    leaf_nodes.append(node)
            chosen_leaf=np.random.choice(leaf_nodes)
            soft_nodes=[]
            # iterate over self.nodes instead to allow soft-nodes everywhere
            for node in chosen_leaf.iter_path_reverse():
                if node.is_root:
                    continue
                else:
                    if bool(np.random.binomial(1, p_soft)):
                        soft_nodes.append(node)
        # Create tumor tree
        if tnode is None:
            tnode = self.root
        if len(tnode.genes)>0 :
            selected_gene = [np.random.choice(tnode.genes, p=self.mut_rates[tnode.genes]/np.sum(self.mut_rates[tnode.genes]))]
            tnode.genes=selected_gene
        for child in tnode.children:
            if bool(np.random.binomial(1, child.f)) or (child in soft_nodes):
                self.sample_tumor(tnode=child, soft_nodes=soft_nodes)
            else:
                self.remove_subtree(child,mode="remove_the_node_with_its_genes")
        return(self,soft_nodes)

    def sample_dataset(self, a):
        #a: the concentration parameter of the beta distribution

        #Initialisation
        dataset=np.array([[0]*self.n_genes])
        weights=np.array([[1]],dtype=float)
        #Create the dataset
        tumor=deepcopy(self)
        tumor, soft_nodes=tumor.sample_tumor()
        for node in PreOrderIter(tumor.root):
            if node in soft_nodes:
                if node.parent.is_root:
                    x=np.random.beta(a*node.f,a*(1-node.f))
                    for i, clone in enumerate(dataset):
                        new_clone=deepcopy(clone)
                        new_clone[node.genes]=1
                        new_weight=deepcopy(weights[i])
                        new_weight=new_weight*x
                        weights[i]=weights[i]*(1-x)
                        dataset=np.vstack([dataset, new_clone])
                        weights=np.vstack([weights, new_weight])
                else:
                    x=np.random.beta(a*node.f,a*(1-node.f))
                    for i, clone in enumerate(dataset):
                        if any(clone[node.parent.genes]==1):
                            new_clone=deepcopy(clone)
                            new_clone[node.genes]=1
                            new_weight=deepcopy(weights[i])
                            new_weight=new_weight*x
                            #Update
                            weights[i]=weights[i]*(1-x)
                            dataset=np.vstack([dataset, new_clone])
                            weights=np.vstack([weights, new_weight])
            else: #Hard nodes
                if node.is_root:
                    continue
                elif node.parent.is_root:
                    for clone in dataset:
                        clone[node.genes]=1
                else:
                    for clone in dataset:
                        if any(clone[node.parent.genes]==1):
                            clone[node.genes]=1
        dataset=dataset.astype(bool)
        return(dataset, weights)

    def draw_sample(self, n_tumors, a=100):
        #Generate from the tree -> B* and from B* -> B
        dataset_dic={}
        weights_dic={}
        for tumor_idx in range(n_tumors):
            clean_dataset, weights = self.sample_dataset(a=a)
            dataset = np.zeros(shape=(clean_dataset.shape[0], clean_dataset.shape[1]), dtype=bool)
            for i in range(clean_dataset.shape[0]):
                for j in range(clean_dataset.shape[1]):
                    if clean_dataset[i,j]:
                        dataset[i,j] = not(np.random.binomial(1, self.pfn))
                    else:
                        dataset[i,j] = bool(np.random.binomial(1, self.pfp))
            dataset_dic[tumor_idx]=dataset
            weights_dic[tumor_idx]=weights
        return(dataset_dic, weights_dic)

    def plot_single_clone(self, the_row, gene_names=None, dot_file='tmp.dot', fig_file='tmp.png'):
        driver_nodes = [self.root]
        driver_nodes.extend([node for node in self.root.descendants])
        if gene_names is None:
            gene_names= ['g%i'%tmp_i for tmp_i in range(self.n_genes)]
        txt = 'digraph tree {\n'
        for i, node in enumerate(driver_nodes):
            genes_list = ','.join(gene_names[tmp_i] for tmp_i in node.genes)
            if len(node.genes)==0:
                label = '< >'
                txt += '    Node%i [label=%s, peripheries=1, shape=circle, style=filled, fillcolor=grey34];\n'%(i, label)
            else:
                n_muts = np.sum([the_row[_tri] for _tri in node.genes])
                if n_muts==0:
                    fillcolor = 'grey95'
                elif n_muts==1:
                    fillcolor = 'limegreen'
                else:
                    fillcolor = 'lightcoral'
                bordercolor = 'black'
                peripheries = 1
                label = '<%s>'%(genes_list)
                txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
        for i, node in enumerate(driver_nodes):
            if node.is_root:
                for child in node.children:
                    j = driver_nodes.index(child)
                    txt += '    Node%i -> Node%i [label=< >];\n' %(i, j)
            else:
                for child in node.children:
                    j = driver_nodes.index(child)
                    #PR_score, PR_p = PR_test(child, dataset)
                    arrow_color = 'black'
                    label = '< >'
                    txt += '    Node%i -> Node%i [style=solid, color=%s, label=%s];\n' %(i, j, arrow_color, label)

        txt += '}'
        with open(dot_file, 'w') as f:
            f.write(txt)
        if fig_file.endswith('.pdf'):
            check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
        else:
            check_call(['dot','-Tpng',dot_file,'-o',fig_file])
        return(txt)

    def to_dot(self, dataset_dic=None, weights_dic=None, gene_names=None, dot_file='tmp.dot', fig_file='tmp.png', plot_reverse_edges=False):
        # All the util_ext functions needed take the concatenated dataset as input
        if dataset_dic is not None:
            dataset=np.concatenate([dataset_dic[key] for key in dataset_dic.keys()])
            include_validations = True
        else:
            include_validations = False
            plot_reverse_edges = False
        driver_nodes = [self.root]
        driver_nodes.extend([node for node in self.root.descendants])
        if gene_names is None:
            gene_names= ['g%i'%tmp_i for tmp_i in range(self.n_genes)]
        txt = 'digraph tree {\n'
        for i, node in enumerate(driver_nodes):
            genes_list = ','.join(gene_names[tmp_i] for tmp_i in node.genes)
            if len(genes_list)==0:
                label = '< >'
                txt += '    Node%i [label=%s, peripheries=1, shape=circle, style=filled, fillcolor=grey34];\n'%(i, label)
            elif len(node.genes)==1:
                label = '<%s>'%genes_list
                txt += '    Node%i [label=%s, peripheries=1, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
            else:
                
                # inserting line breaks for big passenger sets
                #count = 0
                #pos_list = []
                #for i, c in enumerate(genes_list):
                #    if c==',':
                #        count += 1
                #    if count == 4:
                #        pos_list.append(i+1)
                #        count = 0
                #for i, pos in enumerate(pos_list):
                #    genes_list = genes_list[:pos+i*5] + '<br/>' + genes_list[pos+i*5:]
                # done with the line breaks
                
                if include_validations:
                    ME_score, ME_p = ME_test(node, dataset)
                    if ME_p<0.01: # significant mutual exclusivity!
                        fillcolor = 'grey95'
                        bordercolor = 'black'
                    else:
                        fillcolor = 'grey95'
                        bordercolor = 'black'
                    peripheries = 1
                    label = '<%s<br/><font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(genes_list,ME_score,ME_p)
                    txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
                else:
                    fillcolor = 'grey95'
                    bordercolor = 'black'
                    peripheries = 1
                    label = '<%s>'%(genes_list)
                    txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
        for i, node in enumerate(driver_nodes):
            if not(include_validations) or node.is_root or np.sum(np.sum(dataset[:,node.genes], axis=1)>0)==dataset.shape[0]:
                for child in node.children:
                    j = driver_nodes.index(child)
                    txt += '    Node%i -> Node%i [label=< %.3f >];\n' %(i, j, child.f)
            else:
                for child in node.children:
                    j = driver_nodes.index(child)
                    #PR_score, PR_p = PR_test(child, dataset)
                    PR_forward, F_p, PR_backward, B_p, FtoB_p_ratio = PR_test(child, dataset)
                    if F_p < 0.01:
                        arrow_c = 'black'
                        #arrow_c = 'red'
                    else:
                        arrow_c = 'black'
                    #arrow_color = '\"%s:%s\"'%(arrow_c, arrow_c) #(used for double-line arrow in case of perfect PR)
                    arrow_color = arrow_c
                    label = '< %.3f <br/> <font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(child.f, PR_forward, F_p)
                    txt += '    Node%i -> Node%i [style=solid, color=%s, label=%s];\n' %(i, j, arrow_color, label)
                    if plot_reverse_edges and B_p < 0.1: # strong reverse relation as well!
                        txt += '    Node%i -> Node%i [style=dashed, color=black];\n' %(j, i)

        txt += '}'
        with open(dot_file, 'w') as f:
            f.write(txt)
        if fig_file.endswith('.pdf'):
            check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
        else:
            check_call(['dot','-Tpng',dot_file,'-o',fig_file])
        return(txt)

    def min_errors_pertumor(self, dataset, weights):
        n_errors = np.array([0, 0],dtype=float)
        n_ones=0.0
        n_zeros=0.0
        for tumor_idx in range(dataset.shape[0]):
            r = {}
            q_0 = {}
            q_1 = {}
            w_0 = {}
            w_1 = {}
            for node in PostOrderIter(self.root):
                if node.is_root:
                    q_0[node] = np.array([0, 0])
                    q_1[node] = np.array([0, 0])
                else:
                    r[node] = np.sum(dataset[tumor_idx, node.genes])
                    q_0[node] = np.array([r[node], 0])
                    if r[node] == 0:
                        q_1[node] = np.array([0, 1])
                    else:
                        q_1[node] = np.array([r[node]-1, 0])
                w_0[node] = np.array([0, 0])
                w_0[node] += q_0[node]
                for child in node.children:
                    w_0[node] += w_0[child]
                w_1[node] = np.array([0, 0])
                w_1[node] += q_1[node]
                for child in node.children:
                    choice=None
                    if w_1[child][0]+w_1[child][1] == w_0[child][0]+w_0[child][1]:
                        choice= np.random.choice([0,1])
                    if w_1[child][0]+w_1[child][1] < w_0[child][0]+w_0[child][1] or choice==1:
                        w_1[node] += w_1[child]
                    if w_1[child][0]+w_1[child][1] > w_0[child][0]+w_0[child][1] or choice==0:
                        w_1[node] += w_0[child]
            n_errors += weights[tumor_idx]*(w_1[self.root])
            temp_ones = np.sum(dataset[tumor_idx,:])
            n_ones += temp_ones*weights[tumor_idx][0]
            n_zeros += (dataset.shape[1]-temp_ones)*weights[tumor_idx][0]
        return(n_errors,n_ones,n_zeros)

    def min_errors(self, dataset_dic, weights_dic=None):
        constant_to_be_added = 0.00001
        n_errors = np.array([0, 0], dtype=float)
        n_ones=0.0
        n_zeros=0.0
        for key in dataset_dic.keys():
            dataset=dataset_dic[key]
            if weights_dic is not None:
                weights=weights_dic[key]
            else:
                weights=np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
            _temp_errors, _temp_ones , _temp_zeros = self.min_errors_pertumor(dataset,weights)
            n_errors += _temp_errors
            n_ones += _temp_ones
            n_zeros += _temp_zeros
        epsilon_hat = (n_errors[0]+constant_to_be_added)/(n_zeros-n_errors[1]+n_errors[0]+2*constant_to_be_added)
        delta_hat = (n_errors[1]+constant_to_be_added)/(n_ones-n_errors[0]+n_errors[1]+2*constant_to_be_added)
        e_hat = (n_errors[0]+n_errors[1]+constant_to_be_added)/(n_ones+n_zeros+2*constant_to_be_added)
        if epsilon_hat > 0.5 or delta_hat > 0.5:
            #print('n_fp and n_fn are:')
            #print(n_errors)
            epsilon_hat = 0.49
            delta_hat = 0.49
            e_hat = 0.49
        return(epsilon_hat, delta_hat, e_hat)

    def assign_error_values(self, dataset_dic,weights_dic):
        epsilon_hat, delta_hat, e_hat = self.min_errors(dataset_dic,weights_dic)
        if self.single_error:
            self.pfp = e_hat
            self.pfn = e_hat
        else:
            self.pfp = epsilon_hat
            self.pfn = delta_hat
        return(self)

    def assign_f_values(self, dataset_dic, weights_dic=None, fine_tuning=False):
        if fine_tuning:
            upperbound = 0.9999
            lowerbound = 0.0001
        else:
            upperbound = 0.99
            lowerbound = 0.01
        # Dataset-free version:
        if dataset_dic is None:
            for node in PostOrderIter(self.root):
                if not(node.is_root):
                    tmp = 1
                    for child in node.children:
                        tmp *= 1-child.f
                    node.f = 1/(1+tmp)
                    node.f = np.min([node.f, upperbound])
                    node.f = np.max([node.f, lowerbound])
        else:
            n_tumors=len(dataset_dic.keys())
            dataset=np.concatenate([dataset_dic[key] for key in dataset_dic.keys()])
            if weights_dic is not None:
                weights=np.concatenate([weights_dic[key] for key in weights_dic.keys()])
            else:
                weights=np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
            for node in PostOrderIter(self.root):
                if not(node.is_root):
                    if node.parent.is_root:
                        n_p = n_tumors
                        appearing_clones = np.sum(dataset[:,node.genes], axis=1)>0
                        n_u_p = np.sum(weights[appearing_clones])
                        #node.f = np.max((n_u_p-self.pfp*n_p)/(n_p*(1-self.pfp-self.pfn)), 0)
                        node.f = np.max((n_u_p)/(n_p), 0)
                    else:
                        appearing_clones_par=np.sum(dataset[:,node.parent.genes], axis=1)>0
                        n_p = np.sum(weights[appearing_clones_par])
                        if n_p == 0:
                            node.f = 0
                        else:
                            appearing_clones=(np.sum(dataset[:,node.parent.genes], axis=1)>0)*(np.sum(dataset[:,node.genes], axis=1)>0)
                            n_u_p = np.sum(weights[appearing_clones])
                            #node.f = np.max((n_u_p-self.pfp*n_p)/(n_p*(1-self.pfp-self.pfn)), 0)
                            node.f = np.max((n_u_p)/(n_p), 0)
                    node.f = np.min([node.f, upperbound])
                    node.f = np.max([node.f, lowerbound])
        return(self)

    def prior(self, power=0):
        # power = 0 : Uniform prior
        # power = 1 : Prior proportional to 1/(number of possible b*'s)
        p = 0
        if power != 0:
            v = {}
            for node in PostOrderIter(self.root):
                if not node.is_root:
                    v[node] = np.log(len(node.genes))
                else:
                    v[node] = 0
                for child in node.children:
                    v[node] += logsumexp([v[child], 0])
            p = -v[self.root]*power
        return(p)

    def fit_error_params(self, dataset_dic, weights_dic=None):
        step_size = 10**(-6)
        llh=0
        llh_del=0
        llh_eps=0
        for key in dataset_dic.keys():
            dataset=dataset_dic[key]
            if weights_dic is not None:
                weights=weights_dic[key]
            else:
                weights=np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
            _, tumor_llh, tumor_llh_eps, tumor_llh_del = self.error_derivatives(dataset, weights, self.pfp, self.pfn)
            llh += np.log(tumor_llh)
            llh_eps += tumor_llh_eps
            llh_del += tumor_llh_del
        llhs = [llh]
        progress = 1
        while progress>0.1:
            if self.single_error:
                llh_e = llh_eps + llh_del
                new_pfp = min(0.999, max(10**(-5), self.pfp+step_size*llh_e))
                new_pfn = min(0.999, max(10**(-5), self.pfp+step_size*llh_e))
            else:
                new_pfp = min(0.999, max(10**(-5), self.pfp+step_size*llh_eps))
                new_pfn = min(0.999, max(10**(-5), self.pfn+step_size*llh_del))
            llh=0
            llh_del=0
            llh_eps=0
            for key in dataset_dic.keys():
                dataset=dataset_dic[key]
                if weights_dic is not None:
                    weights=weights_dic[key]
                else:
                    weights=np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
                _, tumor_llh, tumor_llh_eps, tumor_llh_del = self.error_derivatives(dataset, weights, new_pfp, new_pfn)
                llh += np.log(tumor_llh)
                llh_eps += tumor_llh_eps
                llh_del += tumor_llh_del
            if llh>llhs[-1]:
                # Accepting new values:
                self.pfp = new_pfp
                self.pfn = new_pfn
                progress = llh-llhs[-1]
                llhs.append(llh)
            else:
                progress = -1
        return(self, llhs)

    def error_derivatives(self, dataset, weights, pfp, pfn):
        # computes derivatives of the log-likelihood w.r.t. the error parameters
        log_pfp = np.log(pfp)
        log_ptn = np.log(1-pfp)
        log_pfn = np.log(pfn)
        log_ptp = np.log(1-pfn)
        tumor_llh = 0
        tumor_llh_eps = 0
        tumor_llh_del = 0
        for tumor_idx in range(dataset.shape[0]): # tumor_idx corresponds to clones
            r = {}
            _gamma = {}
            _gamma_eps = {}
            _lambda = {}
            _lambda_eps = {}
            _lambda_del = {}
            _omega = {}
            _omega_eps = {}
            _psi = {}
            _psi_eps = {}
            _psi_del = {}
            for node in PostOrderIter(self.root):
                r[node] = np.sum(dataset[tumor_idx, node.genes])
                _gamma[node] = r[node]*log_pfp + (node.s-r[node])*log_ptn
                _gamma_eps[node] = r[node]/pfp - (node.s-r[node])/(1-pfp)
                if node.s == 0:
                    _lambda[node] = 0
                    _lambda_eps[node] = 0
                    _lambda_del[node] = 0
                elif r[node] == 0:
                    _lambda[node] = log_pfn+(node.s-r[node]-1)*log_ptn
                    _lambda_eps[node] = r[node]/pfp - (node.s-r[node]-1)/(1-pfp)
                    _lambda_del[node] = 1/pfn
                elif r[node] == node.s:
                    _lambda[node] = log_ptp+(r[node]-1)*log_pfp
                    _lambda_eps[node] = (r[node]-1)/pfp - (node.s-r[node])/(1-pfp)
                    _lambda_del[node] = -1/(1-pfn)
                else:
                    the_coeff = (np.sum(dataset[tumor_idx, node.genes]*self.mut_rates[node.genes]))/(np.sum(self.mut_rates[node.genes]))
                    if the_coeff==1: #When the genes being zero have mut_rate 0
                        _lambda[node] = log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn
                    else:
                        _lambda[node] = logsumexp([
                            np.log(the_coeff)+log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn,
                            np.log(1-the_coeff)+log_pfn+r[node]*log_pfp+(node.s-r[node]-1)*log_ptn
                        ])
                    _lambda_eps[node] = (
                            (the_coeff*(1-pfn))*((r[node]-1)*(pfp**(r[node]-2))*((1-pfp)**(node.s-r[node]))-(node.s-r[node])*((1-pfp)**(node.s-r[node]-1))*(pfp**(r[node]-1)))+
                            ((1-the_coeff)  *pfn)*((r[node])*(pfp**(r[node]-1))*((1-pfp)**(node.s-r[node]-1))-(node.s-r[node]-1)*((1-pfp)**(node.s-r[node]-2))*(pfp**(r[node])))
                        )/np.exp(_lambda[node])
                    _lambda_del[node] = ((node.s*pfp-r[node])*((pfp**(r[node]-1))*((1-pfp)**(node.s-r[node]-1))))/(node.s*np.exp(_lambda[node]))
                    _lambda_del[node] = (
                        (-the_coeff*(pfp**(r[node]-1))*((1-pfp)**(node.s-r[node])))+
                        ((1-the_coeff)*(pfp**(r[node]))*((1-pfp)**(node.s-r[node]-1)))
                        )/np.exp(_lambda[node])
                _omega[node] = _gamma[node]+np.sum([_omega[child] for child in node.children])
                _omega_eps[node] = _gamma_eps[node]+np.sum([_omega_eps[child] for child in node.children])
                tmp = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    if child.f == 0:
                        tmp[i] = _omega[child]
                    elif child.f == 1:
                        tmp[i] = _psi[child]
                    else:
                        tmp[i] = logsumexp([
                            np.log(child.f)+_psi[child],
                            np.log(1-child.f)+_omega[child]
                        ])
                _psi[node] = _lambda[node] + np.sum(tmp)
                tmp_2 = np.exp(-tmp)
                tmp_3 = np.zeros(len(node.children))
                tmp_4 = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    tmp_3[i] = child.f*np.exp(_psi[child])*_psi_eps[child]+(1-child.f)*np.exp(_omega[child])*_omega_eps[child]
                    tmp_4[i] = child.f*np.exp(_psi[child])*_psi_del[child]
                _psi_eps[node] = _lambda_eps[node] + np.sum(tmp_2*tmp_3)
                _psi_del[node] = _lambda_del[node] + np.sum(tmp_2*tmp_4)
            log_llh = _psi[self.root]
            llh_eps = _psi_eps[self.root]
            llh_del = _psi_del[self.root]
            tumor_llh += np.exp(log_llh)*weights[tumor_idx][0]
            tumor_llh_eps += np.exp(log_llh)*weights[tumor_idx][0]*llh_eps
            tumor_llh_del += np.exp(log_llh)*weights[tumor_idx][0]*llh_del
        if tumor_llh==0:
            tumor_llh_eps=0
            tumor_llh_del=0
        else:
            tumor_llh_eps=tumor_llh_eps*(1/tumor_llh)
            tumor_llh_del=tumor_llh_del*(1/tumor_llh)
        return(self, tumor_llh, tumor_llh_eps, tumor_llh_del)

    def likelihood_pertumor(self, dataset, weights, pfp=None, pfn=None):
        if (pfp is None) and (pfn is None):
            pfp = deepcopy(self.pfp)
            pfn = deepcopy(self.pfn)
        log_pfp = np.log(pfp)
        log_ptn = np.log(1-pfp)
        log_pfn = np.log(pfn)
        log_ptp = np.log(1-pfn)
        log_llh_vec = np.zeros(dataset.shape[0])
        for tumor_idx in range(dataset.shape[0]):
            r = {}
            _gamma = {}
            _lambda = {}
            _omega = {}
            _psi = {}
            for node in PostOrderIter(self.root):
                r[node] = np.sum(dataset[tumor_idx, node.genes])
                _gamma[node] = r[node]*log_pfp + (node.s-r[node])*log_ptn
                #Root node
                if node.s == 0:
                    _lambda[node] = 0
                elif r[node] == 0:
                    _lambda[node] = log_pfn+(node.s-r[node]-1)*log_ptn
                elif r[node] == node.s:
                    _lambda[node] = log_ptp+(r[node]-1)*log_pfp
                else:
                    the_coeff = (np.sum(dataset[tumor_idx, node.genes]*self.mut_rates[node.genes]))/(np.sum(self.mut_rates[node.genes]))
                    if the_coeff==1: #When the genes being zero have mut_rate 0
                        _lambda[node] = log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn
                    else:
                        _lambda[node] = logsumexp([
                            np.log(the_coeff)+log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn,
                            np.log(1-the_coeff)+log_pfn+r[node]*log_pfp+(node.s-r[node]-1)*log_ptn
                        ])
                _omega[node] = _gamma[node]+np.sum([_omega[child] for child in node.children])
                tmp = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    if child.f == 0:
                        tmp[i] = _omega[child]
                    elif child.f == 1:
                        tmp[i] = _psi[child]
                    else:
                        tmp[i] = logsumexp([
                            np.log(child.f)+_psi[child],
                            np.log(1-child.f)+_omega[child]
                        ])
                _psi[node] = _lambda[node] + np.sum(tmp)
            log_llh_vec[tumor_idx] = _psi[self.root]
        if len(log_llh_vec)==1: # only one clone, which has to have a weight equal to 1
            total_llh = log_llh_vec[0]
        else:
            total_llh = logsumexp(log_llh_vec+np.log(weights.flatten()))
        return(total_llh)

    def likelihood(self, dataset_dic, weights_dic=None, pfp=None, pfn=None):
        total_llh=0
        for key in dataset_dic.keys():
            dataset=dataset_dic[key]
            if weights_dic is not None:
                weights=weights_dic[key]
            else:
                weights=np.full((dataset.shape[0], 1), 1)/dataset.shape[0]
            total_llh+=self.likelihood_pertumor(dataset, weights, pfp, pfn)
        return(total_llh)

    def posterior(self, dataset_dic, weights_dic=None, pfp=None, pfn=None, pp=0):
        return(self.likelihood(dataset_dic, weights_dic, pfp, pfn) + self.prior(pp))

    def sample_structure(self, dataset_dic, weights_dic, p_moves, log_p_moves, current_posterior,pp, MI_log_pvalues=None, ME_log_pvalues=None, error_estimation=False):
        accepted_proposal = False
        new_posterior = current_posterior
        move_type = np.random.choice(list(p_moves.keys()), p=list(p_moves.values()))
        if move_type == 'hmerge':
            proposal, forward_prob, backward_prob, novel_proposal = self.hmerge(dataset_dic, weights_dic, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
            forward_prob += log_p_moves['hmerge']
            backward_prob += log_p_moves['hsplit']
        elif move_type == 'hsplit':
            proposal, forward_prob, backward_prob, novel_proposal = self.hsplit(dataset_dic, weights_dic, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
            forward_prob += log_p_moves['hsplit']
            backward_prob += log_p_moves['hmerge']
        elif move_type == 'vmerge':
            proposal, forward_prob, backward_prob, novel_proposal = self.vmerge(dataset_dic, weights_dic, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
            forward_prob += log_p_moves['vmerge']
            backward_prob += log_p_moves['vsplit']
        elif move_type == 'vsplit':
            proposal, forward_prob, backward_prob, novel_proposal = self.vsplit(dataset_dic, weights_dic, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
            forward_prob += log_p_moves['vsplit']
            backward_prob += log_p_moves['vmerge']
        elif move_type == 'swap':
            proposal, forward_prob, backward_prob, novel_proposal = self.swap(dataset_dic, weights_dic, MI_log_pvalues=MI_log_pvalues, error_estimation=error_estimation)
        elif move_type == 'spr':
            proposal, forward_prob, backward_prob, novel_proposal = self.spr(dataset_dic, weights_dic, MI_log_pvalues=MI_log_pvalues, error_estimation=error_estimation)
        elif move_type == 'gt':
            proposal, forward_prob, backward_prob, novel_proposal = self.gt(dataset_dic, weights_dic, MI_log_pvalues=MI_log_pvalues, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
        else:
            print('UNDEFINED MOVE!')

        #Metropolis Hastings ratio for proposal
        if novel_proposal:
            proposal_posterior = proposal.posterior(dataset_dic, weights_dic, pp=pp)
            ar = proposal_posterior-current_posterior-forward_prob+backward_prob
            if np.random.binomial(n=1, p=np.exp(np.min([0,ar]))):
                self=proposal
                new_posterior = proposal_posterior
                accepted_proposal = True

        return(self, new_posterior, move_type, novel_proposal, accepted_proposal)

    def hmerge(self, dataset_dic, weights_dic, ME_log_pvalues=None, error_estimation=False, debugging=False):
        # guided hmerge
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        leafset = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root))]
        candidates = []
        if len(leafset)>1:
            for i in range(len(leafset)):
                for j in range(i+1, len(leafset)):
                    if leafset[i].parent == leafset[j].parent:
                        candidates.append((i, j))
        if len(candidates)>0:
            if ME_log_pvalues is not None:
                log_weights = []
                for _tuple in candidates:
                    x = [ME_log_pvalues[_i, _j] for _i in leafset[_tuple[0]].genes for _j in leafset[_tuple[1]].genes]
                    log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(-np.array(log_weights)) # the lower, the better
            else:
                weights = np.ones(len(candidates))/len(candidates)
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_tuple = candidates[selected_idx]
            forward_prob = np.log(weights[selected_idx])

            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _tuple in enumerate(candidates):
                    print('%i) W: %.4f -> {%s} and {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in leafset[_tuple[0]].genes]),','.join([str(item) for item in leafset[_tuple[1]].genes])))
                print("Selected candidate index: %i"%selected_idx)
            #########################################################################

            # Making the changes
            leafset[selected_tuple[0]].genes.extend(leafset[selected_tuple[1]].genes)
            leafset[selected_tuple[1]].parent = None
            proposal.nodes.remove(leafset[selected_tuple[1]])
            del(leafset[selected_tuple[1]])
            # Calculating the backward prob
            bk_candidates = [node for node in proposal.nodes if (node.is_leaf and len(node.genes)>1 and not(node.is_ps) and not(node.is_root))]
            if ME_log_pvalues is not None:
                the_idx = bk_candidates.index(leafset[selected_tuple[0]])
                bk_log_weights = []
                for _bk_node in bk_candidates:
                    x = [ME_log_pvalues[_bk_node.genes[_i], _bk_node.genes[_j]] for _i in range(len(_bk_node.genes)) for _j in range(_i+1,len(_bk_node.genes))]
                    bk_log_weights.append(logsumexp(x)-np.log(len(x)))
                bk_weights = s_softmax(np.array(bk_log_weights)) # the higher, the better
                backward_prob_1 = np.log(bk_weights[the_idx]) # the first step, selecting a node to split
            else:
                bk_weights = np.ones(len(bk_candidates))/len(bk_candidates)
                backward_prob_1 = np.log(bk_weights[0]) # as we have equal weights
            _ng = len(leafset[selected_tuple[0]].genes)
            if _ng>10: # for numerical stability
                backward_prob_2 = -((_ng-1)*np.log(2))
            else:
                backward_prob_2 = -np.log((2**(_ng-1)-1))

            #########################################################################
            if debugging:
                print("Backward Candidates:")
                for _idx, _node in enumerate(bk_candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, bk_weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Backward Probability, part one: %.4f"%np.exp(backward_prob_1))
                print("Backward Probability, part two: %.4f"%np.exp(backward_prob_2))
            #########################################################################

            backward_prob = backward_prob_1 + backward_prob_2
            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def hsplit(self, dataset_dic, weights_dic, ME_log_pvalues=None, error_estimation=False, debugging=False):
        # guided hsplit
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        candidates = [node for node in proposal.nodes if (node.is_leaf and len(node.genes)>1 and not(node.is_ps) and not(node.is_root))]
        if len(candidates)>0:
            if ME_log_pvalues is not None:
                log_weights = []
                for _node in candidates:
                    x = [ME_log_pvalues[_node.genes[_i], _node.genes[_j]] for _i in range(len(_node.genes)) for _j in range(_i+1,len(_node.genes))]
                    log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(np.array(log_weights)) # the higher, the better
            else:
                weights = np.ones(len(candidates))/len(candidates)
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_node = candidates[selected_idx]
            forward_prob_1 = np.log(weights[selected_idx])
            selected_subset = []
            while len(selected_subset)==0 or len(selected_subset)==len(selected_node.genes):
                selected_subset = []
                for gene in selected_node.genes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(gene)

            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _node in enumerate(candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Selected candidate index: ", selected_idx)
                print("Selected subset: {%s}"%(','.join([str(item) for item in selected_subset])))
            #########################################################################

            if len(selected_node.genes)>10: # for numerical stability
                forward_prob_2 = -((len(selected_node.genes)-1)*np.log(2))
            else:
                forward_prob_2 = -np.log((2**(len(selected_node.genes))-2)/2)
            forward_prob = forward_prob_1 + forward_prob_2
            # Making the changes
            for gene in selected_subset:
                selected_node.genes.remove(gene)
            proposal.nodes.append(OncoNode(genes=selected_subset, f=0.5, parent=selected_node.parent))
            # Calculating the backward prob
            new_leafset = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root))]
            bk_candidates = []
            if len(new_leafset)>1:
                for i in range(len(new_leafset)):
                    for j in range(i+1, len(new_leafset)):
                        if new_leafset[i].parent == new_leafset[j].parent:
                            bk_candidates.append((i, j))
            if ME_log_pvalues is not None:
                bk_log_weights = []
                for _idx, _tuple in enumerate(bk_candidates):
                    if selected_node.genes[0] in new_leafset[_tuple[0]].genes and _tuple[1] == len(new_leafset)-1:
                        the_idx = _idx
                    x = [ME_log_pvalues[_i, _j] for _i in new_leafset[_tuple[0]].genes for _j in new_leafset[_tuple[1]].genes]
                    bk_log_weights.append(logsumexp(x)-np.log(len(x)))
                bk_weights = s_softmax(-np.array(bk_log_weights)) # the lower, the better
                backward_prob = np.log(bk_weights[the_idx])
            else:
                bk_weights = np.ones(len(bk_candidates))/len(bk_candidates)
                backward_prob = np.log(bk_weights[0]) # as we have equal weights
            
            #########################################################################
            if debugging:
                print("Backward Candidates:")
                for _idx, _tuple in enumerate(bk_candidates):
                    print('%i) W: %.4f -> {%s} and {%s}'%(_idx, bk_weights[_idx], ','.join([str(item) for item in new_leafset[_tuple[0]].genes]),','.join([str(item) for item in new_leafset[_tuple[1]].genes])))
                print("Backward Probability: %.4f"%np.exp(backward_prob))
            #########################################################################

            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def vmerge(self, dataset_dic, weights_dic, ME_log_pvalues=None, error_estimation=False, debugging=False):
        # guided vmerge
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root) and not(node.parent.is_root))]
        if len(candidates)>0:
            if ME_log_pvalues is not None:
                log_weights = []
                for _node in candidates:
                    x = [ME_log_pvalues[_i, _j] for _i in _node.genes for _j in _node.parent.genes]
                    log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(-np.array(log_weights)) # the lower, the better
            else:
                weights = np.ones(len(candidates))/len(candidates)
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_node = candidates[selected_idx]
            forward_prob = np.log(weights[selected_idx])
            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _node in enumerate(candidates):
                    print('%i) W: %.4f -> {%s} and {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in _node.genes]),','.join([str(item) for item in _node.parent.genes])))
                print("Selected candidate index: %i"%selected_idx)
            #########################################################################
            the_gene = selected_node.genes[0] # used to calculate the backward prob later
            # Making the changes
            selected_node.parent.genes.extend(selected_node.genes)
            _ng = len(selected_node.parent.genes) # used later for backward probability
            selected_node.parent = None
            proposal.nodes.remove(selected_node)
            del(selected_node)
            # Calculating the backward prob
            bk_candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
            if ME_log_pvalues is not None:
                bk_log_weights = []
                for _idx, _bk_node in enumerate(bk_candidates):
                    if the_gene in _bk_node.genes:
                        the_idx = _idx
                    x = [ME_log_pvalues[_bk_node.genes[_i], _bk_node.genes[_j]] for _i in range(len(_bk_node.genes)) for _j in range(_i+1,len(_bk_node.genes))]
                    bk_log_weights.append(logsumexp(x)-np.log(len(x)))
                bk_weights = s_softmax(np.array(bk_log_weights)) # the higher, the better
                backward_prob_1 = np.log(bk_weights[the_idx]) # the first step, selecting a node to split
            else:
                bk_weights = np.ones(len(bk_candidates))/len(bk_candidates)
                backward_prob_1 = np.log(bk_weights[0]) # as we have uniform weights
            if _ng>10: # for numerical stability
                backward_prob_2 = -(_ng*np.log(2))
            else:
                backward_prob_2 = -np.log(2**_ng-2)

            #########################################################################
            if debugging:
                print("Backward Candidates:")
                for _idx, _node in enumerate(bk_candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, bk_weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Backward Probability, part one: %.4f"%np.exp(backward_prob_1))
                print("Backward Probability, part two: %.4f"%np.exp(backward_prob_2))
            #########################################################################

            backward_prob = backward_prob_1 + backward_prob_2
            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def vsplit(self, dataset_dic, weights_dic, ME_log_pvalues=None, error_estimation=False, debugging=False):
        # guided vsplit
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
        if len(candidates)>0:
            if ME_log_pvalues is not None:
                log_weights = []
                for _node in candidates:
                    x = [ME_log_pvalues[_node.genes[_i], _node.genes[_j]] for _i in range(len(_node.genes)) for _j in range(_i+1,len(_node.genes))]
                    log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(np.array(log_weights)) # the higher, the better
            else:
                weights = np.ones(len(candidates))/len(candidates)
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_node = candidates[selected_idx]
            forward_prob_1 = np.log(weights[selected_idx])
            selected_subset = []
            while len(selected_subset)==0 or len(selected_subset)==len(selected_node.genes):
                selected_subset = []
                for gene in selected_node.genes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(gene)
            
            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _node in enumerate(candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Selected candidate index: ", selected_idx)
                print("Selected subset: {%s}"%(','.join([str(item) for item in selected_subset])))
            #########################################################################

            if len(selected_node.genes)>10: # for numerical stability
                forward_prob_2 = -((len(selected_node.genes))*np.log(2))
            else:
                forward_prob_2 = -np.log(2**(len(selected_node.genes))-2)
            forward_prob = forward_prob_1 + forward_prob_2
            # Making the changes
            for gene in selected_subset:
                selected_node.genes.remove(gene)
            proposal.nodes.append(OncoNode(genes=selected_subset, f=0.5, parent=selected_node))
            # Calculating the backward prob
            bk_candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root) and not(node.parent.is_root))]
            if ME_log_pvalues is not None:
                bk_log_weights = []
                for _idx, _node in enumerate(bk_candidates):
                    if selected_subset[0] in _node.genes:
                        the_idx = _idx
                    x = [ME_log_pvalues[_i, _j] for _i in _node.genes for _j in _node.parent.genes]
                    bk_log_weights.append(logsumexp(x)-np.log(len(x)))
                bk_weights = s_softmax(-np.array(bk_log_weights)) # the lower, the better
                backward_prob = np.log(bk_weights[the_idx])
            else:
                bk_weights = np.ones(len(bk_candidates))/len(bk_candidates)
                backward_prob = np.log(bk_weights[0]) # as we have uniform weights
            
            #########################################################################
            if debugging:
                print("Backward Candidates:")
                for _idx, _node in enumerate(bk_candidates):
                    print('%i) W: %.4f -> {%s} and {%s}'%(_idx, bk_weights[_idx], ','.join([str(item) for item in _node.genes]),','.join([str(item) for item in _node.parent.genes])))
                print("Backward Probability: %.4f"%np.exp(backward_prob))
            #########################################################################

            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def swap(self, dataset_dic, weights_dic, MI_log_pvalues=None, error_estimation=False, debugging=False):
        # Sample Structure - swap the genes of two connected nodes
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        candidates = [node for node in proposal.nodes if ((node.parent is not None) and not(node.parent.is_root))]
        if len(candidates)>0:
            if MI_log_pvalues is not None:
                log_weights = []
                for _node in candidates:
                    x = [MI_log_pvalues[_i, _j] for _i in _node.genes for _j in _node.parent.genes]
                    log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(-np.array(log_weights)) # the lower, the better
            else:
                weights= np.ones(len(candidates))/len(candidates)
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_node = candidates[selected_idx]
            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _node in enumerate(candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Selected node: {%s}"%(','.join([str(item) for item in selected_node.genes])))
            #########################################################################
            child_genes = deepcopy(selected_node.genes)
            selected_parent = selected_node.parent
            selected_node.genes = deepcopy(selected_parent.genes)
            selected_parent.genes = deepcopy(child_genes)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def spr(self, dataset_dic, weights_dic, MI_log_pvalues=None, error_estimation=False, debugging=False):
        # Sample Structure - Subtree Pruning and Regrafting
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        if MI_log_pvalues is not None:
            mmilp = np.mean(MI_log_pvalues) # used several times below
        candidates = [node for node in proposal.nodes if (node.parent is not None)]
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            forward_prob += -np.log(len(candidates))
            backward_prob += -np.log(len(candidates))
            subtree_nodes = list(selected_node.descendants)
            subtree_nodes.append(selected_node)
            parent_candidates = [node for node in proposal.nodes if (not(node in subtree_nodes) and not(node.is_ps))]
            if MI_log_pvalues is not None:
                log_weights = []
                for _node in parent_candidates:
                    if _node.is_root:
                        log_weights.append(mmilp)
                    else:
                        x = [MI_log_pvalues[_i, _j] for _i in selected_node.genes for _j in _node.genes]
                        log_weights.append(logsumexp(x)-np.log(len(x)))
                weights = s_softmax(-np.array(log_weights)) # the lower, the better
            else:
                weights= np.ones(len(parent_candidates))/len(parent_candidates)
            selected_idx = np.random.choice(len(parent_candidates), p=weights)
            selected_parent = parent_candidates[selected_idx]
            forward_prob += np.log(weights[selected_idx])
            backward_prob += np.log(weights[parent_candidates.index(selected_node.parent)])
            #########################################################################
            if debugging:
                print("Candidates:")
                for _idx, _node in enumerate(candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, 1/(len(candidates)), ','.join([str(item) for item in _node.genes])))
                print("Selected node: {%s}"%(','.join([str(item) for item in selected_node.genes])))
                print("Parent Candidates:")
                for _idx, _node in enumerate(parent_candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Selected parent: {%s}"%(','.join([str(item) for item in selected_parent.genes])))
            #########################################################################
            # Making the changes
            if selected_parent != selected_node.parent:
                selected_node.parent = selected_parent
                if error_estimation:
                    proposal = proposal.assign_error_values(dataset_dic, weights_dic)
                proposal = proposal.assign_f_values(dataset_dic, weights_dic)
                novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def gt(self, dataset_dic, weights_dic, MI_log_pvalues=None, ME_log_pvalues=None, error_estimation=False, debugging=False):
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        if MI_log_pvalues is not None:
            mmilp = np.mean(MI_log_pvalues) # used many times below
        the_dict = {}
        for _node in [_node for _node in proposal.nodes if len(_node.genes)>1 and not(_node.is_ps)]:
            for _gene in _node.genes:
                the_dict[_gene] = _node
        candidates = list(the_dict.keys())
        if len(candidates)>0:
            log_weights = []
            for _gene in candidates:
                # ME part
                if ME_log_pvalues is not None:
                    x = [ME_log_pvalues[_gene, ccg] for ccg in the_dict[_gene].genes if ccg != _gene]
                    me_part = logsumexp(x)-np.log(len(x))
                else:
                    me_part = 0
                # MI part
                if MI_log_pvalues is not None:
                    if the_dict[_gene].parent.is_root:
                        pmi_part = mmilp
                    else:
                        x = [MI_log_pvalues[_gene, pg] for pg in the_dict[_gene].parent.genes]
                        pmi_part = logsumexp(x)-np.log(len(x))
                    x = [MI_log_pvalues[_gene, cg] for _child_node in the_dict[_gene].children for cg in _child_node.genes]
                    if len(x)==0:
                        cmi_part = np.mean(MI_log_pvalues)
                    else:
                        cmi_part = logsumexp(x)-np.log(len(x))
                    log_weights.append(me_part+pmi_part+cmi_part)
                else:
                    log_weights.append(me_part)
            weights = s_softmax(log_weights) # the higher, the better
            selected_idx = np.random.choice(len(candidates), p=weights)
            selected_gene = candidates[selected_idx]
            forward_prob += np.log(weights[selected_idx])
            # Deciding on the new node for the selected gene
            dest_candidates = [_node for _node in proposal.nodes if not(_node.is_root) and not(_node.is_ps)]
            dest_log_weights = []
            for _node in dest_candidates:
                # ME part
                if ME_log_pvalues is not None:
                    x = [ME_log_pvalues[selected_gene, ccg] for ccg in _node.genes if ccg != selected_gene]
                    me_part = logsumexp(x)-np.log(len(x))
                else:
                    me_part = 0
                # MI part
                if MI_log_pvalues is not None:
                    if _node.parent.is_root:
                        pmi_part = mmilp
                    else:
                        x = [MI_log_pvalues[selected_gene, pg] for pg in _node.parent.genes if pg != selected_gene]
                        pmi_part = logsumexp(x)-np.log(len(x))
                    x = [MI_log_pvalues[selected_gene, cg] for _child_node in _node.children for cg in _child_node.genes if cg != selected_gene]
                    if len(x)==0:
                        cmi_part = np.mean(MI_log_pvalues)
                    else:
                        cmi_part = logsumexp(x)-np.log(len(x))
                    dest_log_weights.append(me_part+pmi_part+cmi_part)
                else:
                    dest_log_weights.append(me_part)
            dest_weights = s_softmax(-np.array(dest_log_weights)) # the lower, the better
            selected_dest_idx = np.random.choice(len(dest_candidates), p=dest_weights)
            selected_dest = dest_candidates[selected_dest_idx]
            forward_prob += np.log(dest_weights[selected_dest_idx])
            if debugging:
                print("Candidates:")
                for _idx, _gene in enumerate(candidates):
                    print('%i) W: %.4f -> %i'%(_idx, weights[_idx], _gene))
                print("Selected gene: {%i}"%(selected_gene))
                print("Dest Candidates:")
                for _idx, _node in enumerate(dest_candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, dest_weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Selected dest: {%s}"%(','.join([str(item) for item in selected_dest.genes])))
            # Making the changes
            the_dict[selected_gene].genes.remove(selected_gene)
            selected_dest.genes.append(selected_gene)
            # Calulating the backward prob
            bk_dict = {}
            for _node in [_node for _node in proposal.nodes if len(_node.genes)>1 and not(_node.is_ps)]:
                for _gene in _node.genes:
                    bk_dict[_gene] = _node
            bk_candidates = list(bk_dict.keys())
            bk_log_weights = []
            for _gene in bk_candidates:
                if ME_log_pvalues is not None:
                    x = [ME_log_pvalues[_gene, ccg] for ccg in bk_dict[_gene].genes if ccg != _gene]
                    me_part = logsumexp(x)-np.log(len(x))
                else:
                    me_part = 0
                if MI_log_pvalues is not None:
                    if bk_dict[_gene].parent.is_root:
                        pmi_part = mmilp
                    else:
                        x = [MI_log_pvalues[_gene, pg] for pg in bk_dict[_gene].parent.genes]
                        pmi_part = logsumexp(x)-np.log(len(x))
                    x = [MI_log_pvalues[_gene, cg] for _child_node in bk_dict[_gene].children for cg in _child_node.genes]
                    if len(x)==0:
                        cmi_part = np.mean(MI_log_pvalues)
                    else:
                        cmi_part = logsumexp(x)-np.log(len(x))
                    bk_log_weights.append(me_part+pmi_part+cmi_part)
                else:
                    bk_log_weights.append(me_part)
            bk_weights = s_softmax(bk_log_weights) # the higher, the better
            backward_prob += np.log(bk_weights[bk_candidates.index(selected_gene)])
            backward_prob += np.log(dest_weights[dest_candidates.index(the_dict[selected_gene])])
            if debugging:
                print("Backward Candidates:")
                for _idx, _gene in enumerate(bk_candidates):
                    print('%i) W: %.4f -> %i'%(_idx, bk_weights[_idx], _gene))
                print("Selected gene: {%i}"%(selected_gene))
                print("Backward dest Candidates:")
                for _idx, _node in enumerate(dest_candidates):
                    print('%i) W: %.4f -> {%s}'%(_idx, dest_weights[_idx], ','.join([str(item) for item in _node.genes])))
                print("Backward selected dest: {%s}"%(','.join([str(item) for item in the_dict[selected_gene].genes])))
                # Note that (original_node = the_dict[selected_gene])
            if error_estimation:
                proposal = proposal.assign_error_values(dataset_dic, weights_dic)
            proposal = proposal.assign_f_values(dataset_dic, weights_dic)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def fast_training_iteration(
        self,
        dataset_dic,
        weights_dic,
        n_iters,
        pp,
        seed=None,
        current_posterior=None,
        p_moves=None,
        collapse_interval=10000,
        guided_move_prob=0.5,
        thinning_interval=10,
        error_estimation=False
        ):
        # if n_iters is divisable by collapse_interval, the output will be already pruned
        if seed is not None:
            np.random.seed(seed)
        MI_scores, MI_log_pvalues, ME_log_pvalues = calc_pair_mat(np.concatenate([dataset_dic[key] for key in dataset_dic.keys()]))
        if p_moves is None:
            p_moves = {
                'hmerge': 1,
                'hsplit': 1,
                'vmerge': 1,
                'vsplit': 1,
                'swap': 1,
                'spr': 5,
                'gt': 5
            }
            # Normalization:
            factor=1.0/sum(p_moves.values())
            for k in p_moves:
                p_moves[k] = p_moves[k]*factor
        log_p_moves = {k: np.log(v) for k, v in p_moves.items() if v>0}
        if current_posterior is None:
            current_posterior = self.posterior(dataset_dic, weights_dic, pp=pp)
        # To record move-specific stats #
        n_proposed = {k: 0 for k, _ in p_moves.items()}
        n_novel = {k: 0 for k, _ in p_moves.items()}
        n_accepted = {k: 0 for k, _ in p_moves.items()}
        # To record the scans #
        scans_tensor = np.empty(shape=(self.n_genes+1, self.n_genes+1, 0), dtype=bool)
        # Main outputs #
        posteriors_list = []
        best_sample = deepcopy(self)
        best_posterior = current_posterior
        n_updates = 0
        for _iter in range(n_iters):
            if (_iter+1) % collapse_interval == 0: # special iteration!
                # collapsing the tree
                self = self.prune(dataset_dic, weights_dic, consider_mut_freqs=False, error_estimation=error_estimation)
                new_posterior = self.posterior(dataset_dic, weights_dic, pp=pp)
            else: # normal iteration
                if np.random.binomial(1, guided_move_prob):
                    self, new_posterior, move_type, novel_proposal, accepted_proposal = self.sample_structure(dataset_dic, weights_dic, p_moves, log_p_moves, current_posterior, pp, MI_log_pvalues=MI_log_pvalues, ME_log_pvalues=ME_log_pvalues, error_estimation=error_estimation)
                else:
                    self, new_posterior, move_type, novel_proposal, accepted_proposal = self.sample_structure(dataset_dic, weights_dic, p_moves, log_p_moves, current_posterior, pp, MI_log_pvalues=None, ME_log_pvalues=None, error_estimation=error_estimation)
            
            if (_iter+1)%thinning_interval==0:
                scans_tensor = np.concatenate([scans_tensor, self.edge_scan().reshape(self.n_genes+1, self.n_genes+1, 1)], axis=2)
            n_proposed[move_type] += 1
            if novel_proposal:
                n_novel[move_type] += 1
            if accepted_proposal:
                n_accepted[move_type] += 1
                n_updates+=1
            # in any iteration we have:
            posteriors_list.append(new_posterior)
            current_posterior = new_posterior
            if new_posterior>best_posterior:
                best_sample = deepcopy(self)
                best_posterior = deepcopy(new_posterior)
        details = {
            'n_proposed': n_proposed,
            'n_novel': n_novel,
            'n_accepted': n_accepted
        }
        return(self, current_posterior, best_sample, best_posterior, posteriors_list, n_updates, details, scans_tensor)


if __name__ == "__main__":
    pass