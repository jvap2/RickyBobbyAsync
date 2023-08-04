import cugraph
import cudf

num_clusters=10
graph= cudf.read_csv("webGoogle.csv", delimiter='\t', dtype=['int32', 'int32'])
print(graph.head())
g=cugraph.Graph()
g.from_cudf_edgelist(graph, source='FromNodeId', destination='ToNodeId')
df_cluster=[]
score_cluster=[]
for i in range(1,num_clusters+1):
    df=cugraph.spectralBalancedCutClustering(g, num_clusters=num_clusters, num_eigen_vects=i)
    score=cugraph.analyzeClustering_edge_cut(g,num_clusters,df)
    df_cluster.append(df)
    score_cluster.append(score)
idx=score_cluster.index(max(score_cluster))
df_cluster[idx].to_csv("Cluster_Assignment.csv")

