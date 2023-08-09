import cugraph
import cudf

num_clusters=10
graph= cudf.read_csv("../Data/google/webGoogle.csv", dtype=['int32', 'int32'])
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
fin_cluster=df_cluster[idx].sort_values('cluster')
print(fin_cluster.describe())
print(fin_cluster.summary())
fin_cluster.to_csv("../Data/google/Cluster_Assignment.csv", index=False)

