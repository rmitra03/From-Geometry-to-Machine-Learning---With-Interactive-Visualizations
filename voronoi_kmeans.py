# voronoi_demo.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

st.title("Voronoi Diagrams & k-means Clustering")

st.write("""
Click on the plot to add points, then run the algorithms to see how they're geometrically equivalent!
""")

# Initialize session state for points
if 'points' not in st.session_state:
    st.session_state.points = []

# Sidebar controls
st.sidebar.header("Controls")
n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)

if st.sidebar.button("Clear Points"):
    st.session_state.points = []
    st.rerun()

if st.sidebar.button("Generate Random Points"):
    st.session_state.points = np.random.rand(20, 2).tolist()
    st.rerun()

# Main plotting area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Voronoi Diagram")
    fig_voronoi = go.Figure()
    
    if len(st.session_state.points) > 0:
        points = np.array(st.session_state.points)
        
        # Plot points
        fig_voronoi.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Data Points'
        ))
        
        # Compute and plot Voronoi if enough points
        if len(points) >= 4:
            vor = Voronoi(points)
            
            # Plot Voronoi edges
            for simplex in vor.ridge_vertices:
                if -1 not in simplex:  # Skip infinite edges for now
                    line = vor.vertices[simplex]
                    fig_voronoi.add_trace(go.Scatter(
                        x=line[:, 0],
                        y=line[:, 1],
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False
                    ))
    
    fig_voronoi.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_voronoi, use_container_width=True, key="voronoi_plot")

with col2:
    st.subheader("k-means Clustering")
    fig_kmeans = go.Figure()
    
    if len(st.session_state.points) >= n_clusters:
        points = np.array(st.session_state.points)
        
        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        centroids = kmeans.cluster_centers_
        
        # Plot points colored by cluster
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            fig_kmeans.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(size=8),
                name=f'Cluster {i}'
            ))
        
        # Plot centroids
        fig_kmeans.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(size=15, symbol='x', color='black', line=dict(width=2)),
            name='Centroids'
        ))
    
    fig_kmeans.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_kmeans, use_container_width=True, key="kmeans_plot")

# Instructions
st.info("Add points using 'Generate Random Points' or manually add coordinates below")

# Manual point input (temporary solution - it'll be clickable later)
with st.expander("Add points manually"):
    col_x, col_y = st.columns(2)
    with col_x:
        x_coord = st.number_input("X coordinate", 0.0, 1.0, 0.5, step=0.1)
    with col_y:
        y_coord = st.number_input("Y coordinate", 0.0, 1.0, 0.5, step=0.1)
    
    if st.button("Add Point"):
        st.session_state.points.append([x_coord, y_coord])
        st.rerun()

st.write(f"Current points: {len(st.session_state.points)}")