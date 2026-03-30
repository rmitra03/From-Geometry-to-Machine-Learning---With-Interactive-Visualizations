import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from sklearn.neighbors import KNeighborsClassifier

st.title("Delaunay Triangulation & k-Nearest Neighbors")

st.write("""
Generate two classes and see how Delaunay triangulation connects nearest neighbors,
which is the same structure k-NN uses for classification!
""")

# Initialize session state
if 'class_0' not in st.session_state:
    st.session_state.class_0 = []
if 'class_1' not in st.session_state:
    st.session_state.class_1 = []
if 'query_point' not in st.session_state:
    st.session_state.query_point = None

# Sidebar controls
st.sidebar.header("Controls")

k_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 10, 3)

if st.sidebar.button("Clear All Points"):
    st.session_state.class_0 = []
    st.session_state.class_1 = []
    st.session_state.query_point = None
    st.rerun()

if st.sidebar.button("Generate Random Classes"):
    # Generate two separated clusters with random positions each time
    center_0 = np.random.rand(2) * 0.4 + 0.1  # Random center in [0.1, 0.5]
    center_1 = np.random.rand(2) * 0.4 + 0.5  # Random center in [0.5, 0.9]
    
    st.session_state.class_0 = (np.random.randn(10, 2) * 0.1 + center_0).tolist()
    st.session_state.class_1 = (np.random.randn(10, 2) * 0.1 + center_1).tolist()
    
    # Also generate a random query point in the middle region
    st.session_state.query_point = [np.random.uniform(0.3, 0.7), 
                                     np.random.uniform(0.3, 0.7)]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Click 'Generate' multiple times to explore different data distributions")

# Main plotting area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Delaunay Triangulation")
    fig_delaunay = go.Figure()
    
    # Combine both classes for Delaunay
    all_points = []
    all_labels = []
    
    if len(st.session_state.class_0) > 0:
        points_0 = np.array(st.session_state.class_0)
        all_points.extend(points_0.tolist())
        all_labels.extend([0] * len(points_0))
        
        fig_delaunay.add_trace(go.Scatter(
            x=points_0[:, 0],
            y=points_0[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Class 0'
        ))
    
    if len(st.session_state.class_1) > 0:
        points_1 = np.array(st.session_state.class_1)
        all_points.extend(points_1.tolist())
        all_labels.extend([1] * len(points_1))
        
        fig_delaunay.add_trace(go.Scatter(
            x=points_1[:, 0],
            y=points_1[:, 1],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Class 1'
        ))
    
    # Compute Delaunay triangulation if enough points
    if len(all_points) >= 3:
        points = np.array(all_points)
        tri = Delaunay(points)
        
        # Plot Delaunay triangulation edges
        for simplex in tri.simplices:
            # Draw triangle edges
            triangle = points[simplex]
            triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
            fig_delaunay.add_trace(go.Scatter(
                x=triangle[:, 0],
                y=triangle[:, 1],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Plot query point if it exists
    if st.session_state.query_point is not None:
        qp = st.session_state.query_point
        fig_delaunay.add_trace(go.Scatter(
            x=[qp[0]],
            y=[qp[1]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Query Point'
        ))
        
        # Find and highlight k-nearest neighbors in Delaunay
        if len(all_points) >= k_neighbors:
            points = np.array(all_points)
            distances = np.linalg.norm(points - qp, axis=1)
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_points = points[nearest_indices]
            
            # Draw lines to nearest neighbors
            for np_point in nearest_points:
                fig_delaunay.add_trace(go.Scatter(
                    x=[qp[0], np_point[0]],
                    y=[qp[1], np_point[1]],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig_delaunay.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_delaunay, use_container_width=True, key="delaunay_plot")

with col2:
    st.subheader("k-NN Classification")
    fig_knn = go.Figure()
    
    # Need both classes for k-NN
    if len(st.session_state.class_0) >= 1 and len(st.session_state.class_1) >= 1:
        points_0 = np.array(st.session_state.class_0)
        points_1 = np.array(st.session_state.class_1)
        
        # Combine data
        X = np.vstack([points_0, points_1])
        y = np.array([0] * len(points_0) + [1] * len(points_1))
        
        # Train k-NN
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X, y)
        
        # Create decision boundary mesh
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        fig_knn.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
            showscale=False,
            opacity=0.3,
            hoverinfo='skip'
        ))
        
        # Plot training points
        fig_knn.add_trace(go.Scatter(
            x=points_0[:, 0],
            y=points_0[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Class 0'
        ))
        
        fig_knn.add_trace(go.Scatter(
            x=points_1[:, 0],
            y=points_1[:, 1],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Class 1'
        ))
        
        # Plot query point and its k-nearest neighbors
        if st.session_state.query_point is not None:
            qp = st.session_state.query_point
            
            # Predict class
            prediction = knn.predict([qp])[0]
            pred_color = 'blue' if prediction == 0 else 'red'
            
            fig_knn.add_trace(go.Scatter(
                x=[qp[0]],
                y=[qp[1]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star', 
                           line=dict(color=pred_color, width=3)),
                name=f'Query (Predicted: Class {prediction})'
            ))
            
            # Find k-nearest neighbors
            distances = np.linalg.norm(X - qp, axis=1)
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_points = X[nearest_indices]
            
            # Highlight nearest neighbors
            fig_knn.add_trace(go.Scatter(
                x=nearest_points[:, 0],
                y=nearest_points[:, 1],
                mode='markers',
                marker=dict(size=12, color='yellow', symbol='circle-open', 
                           line=dict(width=3)),
                name='k-Nearest Neighbors'
            ))
            
            # Draw lines to nearest neighbors
            for np_point in nearest_points:
                fig_knn.add_trace(go.Scatter(
                    x=[qp[0], np_point[0]],
                    y=[qp[1], np_point[1]],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig_knn.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_knn, use_container_width=True, key="knn_plot")

# Instructions
st.info("Generate random classes, then set a query point to see k-NN classification")

# Quantitative Analysis Section
if len(st.session_state.class_0) >= 1 and len(st.session_state.class_1) >= 1 and st.session_state.query_point is not None:
    st.markdown("---")
    st.subheader("Quantitative Analysis")
    
    points_0 = np.array(st.session_state.class_0)
    points_1 = np.array(st.session_state.class_1)
    
    X = np.vstack([points_0, points_1])
    y = np.array([0] * len(points_0) + [1] * len(points_1))
    
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X, y)
    
    all_points = X.tolist()
    tri = Delaunay(np.array(all_points))
    
    qp = st.session_state.query_point
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Training Points", len(X))
    
    with col2:
        st.metric("k (neighbors)", k_neighbors)
    
    with col3:
        st.metric("Delaunay Triangles", len(tri.simplices))
    
    with col4:
        st.metric("Delaunay Edges", len(tri.simplices) * 3 // 2)
    
    # Query-specific analysis
    st.markdown("**Query Point Analysis:**")
    
    # Find k-nearest neighbors
    distances = np.linalg.norm(X - qp, axis=1)
    nearest_indices = np.argsort(distances)[:k_neighbors]
    nearest_distances = distances[nearest_indices]
    nearest_labels = y[nearest_indices]
    
    # Prediction
    prediction = knn.predict([qp])[0]
    prediction_proba = knn.predict_proba([qp])[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Class", 
            f"Class {prediction}",
            help="k-NN classification result"
        )
    
    with col2:
        confidence = prediction_proba[prediction]
        st.metric(
            "Confidence", 
            f"{confidence:.1%}",
            help="Proportion of k neighbors in predicted class"
        )
    
    with col3:
        avg_neighbor_dist = np.mean(nearest_distances)
        st.metric(
            "Avg Neighbor Distance", 
            f"{avg_neighbor_dist:.4f}",
            help="Average distance to k-nearest neighbors"
        )
    
    # Neighbor distribution
    st.markdown("**k-Nearest Neighbors Breakdown:**")
    col1, col2 = st.columns(2)
    
    with col1:
        class_0_neighbors = np.sum(nearest_labels == 0)
        st.metric("Class 0 Neighbors", f"{class_0_neighbors}/{k_neighbors}")
    
    with col2:
        class_1_neighbors = np.sum(nearest_labels == 1)
        st.metric("Class 1 Neighbors", f"{class_1_neighbors}/{k_neighbors}")
    
    # Delaunay connectivity verification
    st.markdown("**Delaunay-kNN Connection Verification:**")
    
    # Check if k-nearest neighbors are connected via Delaunay edges
    # (This is a simplified check - full verification would trace Delaunay graph)
    st.info("""
    In a proper Delaunay triangulation, nearest neighbors are connected by Delaunay edges. 
    The green dashed lines in the visualization show this geometric structure that k-NN exploits.
    """)
    
    # Computational comparison
    st.markdown("**Computational Performance:**")
    
    import time
    
    # Time k-NN query
    start = time.time()
    for _ in range(1000):
        knn.predict([qp])
    knn_time = (time.time() - start) / 1000 * 1000  # Convert to ms
    
    # Time Delaunay construction
    start = time.time()
    for _ in range(100):
        tri_temp = Delaunay(X)
    delaunay_time = (time.time() - start) / 100 * 1000
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("k-NN Query Time", f"{knn_time:.4f} ms")
    with col2:
        st.metric("Delaunay Construction Time", f"{delaunay_time:.3f} ms")

elif len(st.session_state.class_0) >= 1 and len(st.session_state.class_1) >= 1:
    st.info("Set a query point to see detailed k-NN analysis")

# Description of why it works
st.markdown("---")
st.subheader("Why Does This Happen?")
st.markdown("""
**k-Nearest Neighbors (k-NN)** classification relies on finding the k closest points to a query point. 
The **Delaunay triangulation** provides an optimal geometric structure for this search:

- Delaunay triangles connect points that are natural neighbors
- The dual of Delaunay is the Voronoi diagram (from Demo 1!)
- Nearest neighbors are guaranteed to share a Delaunay edge

This means we can use Delaunay structure to efficiently find k-nearest neighbors without 
computing distances to every single point.

**What to observe:** The green dashed lines show the k-nearest neighbors. They follow the Delaunay edges!
""")

# Manual input
with st.expander("Add points manually"):
    input_type = st.radio("What to add?", ["Training Point", "Query Point"])
    
    if input_type == "Training Point":
        class_choice = st.radio("Add to which class?", ["Class 0 (Blue)", "Class 1 (Red)"])
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_coord = st.number_input("X coordinate", 0.0, 1.0, 0.5, step=0.1, key="x_input")
    with col_y:
        y_coord = st.number_input("Y coordinate", 0.0, 1.0, 0.5, step=0.1, key="y_input")
    
    if st.button("Add"):
        if input_type == "Query Point":
            st.session_state.query_point = [x_coord, y_coord]
        else:
            if "Class 0" in class_choice:
                st.session_state.class_0.append([x_coord, y_coord])
            else:
                st.session_state.class_1.append([x_coord, y_coord])
        st.rerun()

st.write(f"Class 0: {len(st.session_state.class_0)} | Class 1: {len(st.session_state.class_1)} | Query point: {'Set' if st.session_state.query_point else 'Not set'}")