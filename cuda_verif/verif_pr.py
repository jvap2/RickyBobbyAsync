import cugraph
import cudf
import os


folder=os.getcwd()
#Read in graph data
folder=os.getcwd()
df=cudf.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm.csv"),names=["from","to","cluster"])

#Create a Graph
G = cugraph.Graph()
G.from_cudf_edgelist(df, source='from', destination='to')
#Perform PageRank on the graph
df_pagerank = cugraph.pagerank(G)
print(df_pagerank)
#Print the top 5 connections
print(df_pagerank.nlargest(5, 'pagerank'))
#Export the results to a csv
df_pagerank.to_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm_pagerank_py.csv"))