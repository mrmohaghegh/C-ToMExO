import pandas as pd
from anytree import RenderTree, Node, NodeMixin, PostOrderIter
from anytree.exporter import DotExporter 
import os
import numpy as np
from copy import deepcopy
from subprocess import check_call

class PhyloNode(NodeMixin):

    def __init__(self, name, genes=[], cp=[], parent=None, children=[]):
        super().__init__()
        self.name=name
        self.genes = genes
        self.cp = cp
        self.parent = parent
        self.children = children
        
    @property
    def is_root(self):
        # Is it the root?
        if self.parent is None:
            return(True)
        else:
            return(False)
    
    @property
    def is_leaf(self):
        # Is it a leaf node?
        if len(self.children)==0:
            return(True)
        else:
            return(False)


class PhyloTree():

    def __init__(self, nodes, llh):

        self.nodes=nodes
        temp_genes=[]
        for node in self.nodes:
            temp_genes.extend(node.genes)
        self.genes = temp_genes
        for node in self.nodes:
            if node.is_root:
                self.root = node
        #LogLikelihood of the tree
        self.llh=llh

    @classmethod
    def from_pairtree(cls, structure, node_to_gene, cp, llh):
        root_node = PhyloNode(name='S0',cp=cp.loc['S0',0])
        nodes=[root_node]
        clones=structure.loc[structure['parent']==0].index.to_list() #Have as parent the root node
        while(len(clones)>0):
            to_remove=clones.pop(0)
            par=structure.loc[to_remove,'parent']
            if par==0: #root node
                nodes.append(PhyloNode(name=to_remove, genes=node_to_gene[to_remove], cp=cp.loc[to_remove, 0], parent=root_node, children=[]))
            else:
                par_idx=[i for i,x in enumerate(nodes) if x.name=='S'+str(par)][0] #which entry is the parent
                nodes.append(PhyloNode(name=to_remove, genes=node_to_gene[to_remove], cp=cp.loc[to_remove, 0], parent=nodes[par_idx], children=[]))
            unlocked_par=int(to_remove[1:])
            clones.extend(structure.loc[structure['parent']==unlocked_par].index.to_list())
        assert len(nodes)==structure.shape[0]+1, "Haven't visited all clones"
        tree = cls(nodes, llh)
        return(tree)
    

    def print(self):
        for pre, _, node in RenderTree(self.root):
            print("%s%s:\n      %s\n        %s" % (pre, node.name,",".join(node.genes),str(node.cp)))
        return()

    def prune_by_IntOgen(self, IntOgenGenes):
        pruned_tree = deepcopy(self)
        for node in PostOrderIter(pruned_tree.root):
            if not(node.is_root):
                to_keep=[gene for gene in node.genes if gene in IntOgenGenes]
                if len(to_keep)>0:
                    node.genes=to_keep
                else:
                    if node.is_leaf:
                        node.parent=None
                        pruned_tree.nodes.remove(node)
                    else:
                        for child in node.children:
                            child.parent=node.parent
                        node.parent=None
                        pruned_tree.nodes.remove(node)
        return(pruned_tree)

    @property
    def clones(self):
        clones={}
        for node in self.root.descendants:
            clones[node.name]=node.genes
            for anc in node.ancestors:
                clones[node.name].extend(anc.genes)    
        return(clones)

    def weights(self):
        if not any(self.clones):
            return([])
        else:
            weight=[]
            node_names=[]
            for node in self.root.descendants:
                if node.is_leaf :
                    weight.append(node.cp)
                    node_names.append(node.name)
                else:
                    np.sum(
                    weight.append(node.cp - n[child.cp for child in node.children]))
                    node_names.append(node.name)
            weight=pd.DataFrame(weight, index=node_names)
            #Normalise
            weight=weight/weight.sum()
            return(weight) #list of dataframes
            
    def matrix(self,IntOgenGenes):
        if not any(self.clones): #empty, the tree has only the root
            return([])
        else:
            df=pd.DataFrame(0,index=self.clones.keys(),columns=IntOgenGenes)
            for key in self.clones.keys():
                for gene in self.clones[key]:
                    df.loc[key][gene]=1
            return(df)

def add_legend(TreeObject,filepath):

    with open(filepath, 'r+') as fp:

        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines
        fp.writelines(lines[:-1])
        
        #Add the legend
        fp.write("\n")
        fp.write("    node[shape=plaintext]\n")
        fp.write("    fontsize=\"10\"\n")
        fp.write("    struct1 [label=\n")
        fp.write("    <<TABLE BORDER=\"1\" CELLBORDER=\"1\" CELLSPACING=\"0\" >\n")
        
        # for key in TreeObject.genes.keys():
        #     fp.write("    <TR><TD ALIGN=\"LEFT\">" + 
        #     "Node" + str(key) + ":" + 
        #     ', '.join([elem for elem in TreeObject.genes[key]]) + 
        #     "</TD><TD ALIGN=\"LEFT\">" + 
        #     "cp" + str(key) + ":" + 
        #     ', '.join([str(round(elem, 3)) for elem in TreeObject.cp[key]]) + 
        #     "</TD></TR>\n")
        fp.write("    <TR><TD ALIGN=\"LEFT\">LogLikelihood:" + 
        "{:.2f}".format(TreeObject.llh) + "</TD></TR>\n")
        fp.write("    </TABLE>>];\n")
        fp.write("\n")
        #fp.write("    rankdir = \"LR\"\n")
        fp.write("}\n")
        
        fp.close()
        return
        

def to_dot(TreeObject,file_name, outdir, mainpath=os.getcwd()):

    #Function to add the cp to the vertices
    def nodenamefunc(node):
        if node.name == "S0":
            return("root")
        else:
            return '{}'.format(node.name)
    
    def nodeattrfunc(node):
        if node.name=="S0":
            return('label=<<font color=\'Blue\'> root </font><br/><font color=\'Blue\' POINT-SIZE=\'12\'> cp : 1 </font>>, shape=oval, color=Blue') 
        else:
            gene_names=", ".join(node.genes)
            node_cp=str(round(node.cp,4))
            return('label =<%s <br/> <font color=\'Red\' POINT-SIZE=\'12\'> cp : %s </font>>, shape=box' % (gene_names, node_cp))

    dot_folder= "dot_files"
    if not os.path.exists(os.path.join(mainpath,outdir,dot_folder)):
        os.makedirs(os.path.join(mainpath,outdir,dot_folder),exist_ok=True) 
    filepath=os.path.join(mainpath, outdir, dot_folder, file_name +".dot")

    #Export it to a dot file
    DotExporter(TreeObject.root, graph="digraph",
                            nodenamefunc=nodenamefunc,
                            nodeattrfunc=nodeattrfunc).to_dotfile(filepath)
                            #nodeattrfunc=lambda node: "shape=circle").to_dotfile(filepath)

    #Add legend
    add_legend(TreeObject,filepath)

    picture_folder="png_files"
    if not os.path.exists(os.path.join(mainpath,outdir,picture_folder)):
        os.makedirs(os.path.join(mainpath,outdir,picture_folder),exist_ok=True) 
    picture_file_path=os.path.join(mainpath,outdir,picture_folder,file_name +".png")
    check_call(['dot','-Tpng',filepath,'-o',picture_file_path])
    return
    




    








