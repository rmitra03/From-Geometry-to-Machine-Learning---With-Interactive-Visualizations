import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.svm import SVC

st.title("Convex Hulls & Support Vector Machines")

st.write("""
Generate two classes of points and see how SVM finds the decision boundary 
by working with the convex hulls of each class!
""")

# Initialize session state for two classes
if 'class_0' not in st.session_state:
    st.session_state.class_0 = []
if 'class_1' not in st.session_state:
    st.session_state.class_1 = []

# Sidebar controls
st.sidebar.header("Controls")

if st.sidebar.button("Clear All Points"):
    st.session_state.class_0 = []
    st.session_state.class_1 = []
    st.rerun()

if st.sidebar.button("Generate Random Classes"):
    # Generate two separated clusters with random positions each time
    center_0 = np.random.rand(2) * 0.4 + 0.1  # Random center in [0.1, 0.5]
    center_1 = np.random.rand(2) * 0.4 + 0.5  # Random center in [0.5, 0.9]
    
    st.session_state.class_0 = (np.random.randn(15, 2) * 0.12 + center_0).tolist()
    st.session_state.class_1 = (np.random.randn(15, 2) * 0.12 + center_1).tolist()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Click 'Generate' multiple times to see different configurations")

# Main plotting area - WITH OVERLAY
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Convex Hulls")
    fig_hull = go.Figure()
    
    # Plot class 0
    if len(st.session_state.class_0) > 0:
        points_0 = np.array(st.session_state.class_0)
        fig_hull.add_trace(go.Scatter(
            x=points_0[:, 0],
            y=points_0[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Class 0'
        ))
        
        # Compute convex hull for class 0
        if len(points_0) >= 3:
            hull_0 = ConvexHull(points_0)
            # Plot hull edges
            for simplex in hull_0.simplices:
                fig_hull.add_trace(go.Scatter(
                    x=points_0[simplex, 0],
                    y=points_0[simplex, 1],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))
    
    # Plot class 1
    if len(st.session_state.class_1) > 0:
        points_1 = np.array(st.session_state.class_1)
        fig_hull.add_trace(go.Scatter(
            x=points_1[:, 0],
            y=points_1[:, 1],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Class 1'
        ))
        
        # Compute convex hull for class 1
        if len(points_1) >= 3:
            hull_1 = ConvexHull(points_1)
            # Plot hull edges
            for simplex in hull_1.simplices:
                fig_hull.add_trace(go.Scatter(
                    x=points_1[simplex, 0],
                    y=points_1[simplex, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False
                ))
    
    fig_hull.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_hull, use_container_width=True, key="hull_plot")

with col2:
    st.subheader("SVM Decision Boundary")
    fig_svm = go.Figure()
    
    # Need both classes with enough points
    if len(st.session_state.class_0) >= 3 and len(st.session_state.class_1) >= 3:
        points_0 = np.array(st.session_state.class_0)
        points_1 = np.array(st.session_state.class_1)
        
        # Combine data for SVM
        X = np.vstack([points_0, points_1])
        y = np.array([0] * len(points_0) + [1] * len(points_1))
        
        # Train SVM
        svm = SVC(kernel='linear', C=1000)
        svm.fit(X, y)
        
        # Plot points
        fig_svm.add_trace(go.Scatter(
            x=points_0[:, 0],
            y=points_0[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Class 0'
        ))
        
        fig_svm.add_trace(go.Scatter(
            x=points_1[:, 0],
            y=points_1[:, 1],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Class 1'
        ))
        
        # Highlight support vectors
        support_vectors = svm.support_vectors_
        fig_svm.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='circle-open', line=dict(width=3)),
            name='Support Vectors'
        ))
        
        # Plot decision boundary and margins
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary (Z=0)
        fig_svm.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            contours=dict(start=0, end=0, size=1),
            line=dict(color='green', width=3),
            showscale=False,
            name='Decision Boundary'
        ))
        
        # Plot margins (Z=-1 and Z=1)
        fig_svm.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            contours=dict(start=-1, end=1, size=2),
            line=dict(color='gray', width=1, dash='dash'),
            showscale=False,
            name='Margins'
        ))
    
    fig_svm.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_svm, use_container_width=True, key="svm_plot")

with col3:
    st.subheader("Overlay View")
    fig_overlay = go.Figure()
    
    if len(st.session_state.class_0) >= 3 and len(st.session_state.class_1) >= 3:
        points_0 = np.array(st.session_state.class_0)
        points_1 = np.array(st.session_state.class_1)
        
        X = np.vstack([points_0, points_1])
        y = np.array([0] * len(points_0) + [1] * len(points_1))
        
        # Train SVM
        svm = SVC(kernel='linear', C=1000)
        svm.fit(X, y)
        
        # Compute convex hulls
        hull_0 = ConvexHull(points_0)
        hull_1 = ConvexHull(points_1)
        
        # Plot points
        fig_overlay.add_trace(go.Scatter(
            x=points_0[:, 0],
            y=points_0[:, 1],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Class 0'
        ))
        
        fig_overlay.add_trace(go.Scatter(
            x=points_1[:, 0],
            y=points_1[:, 1],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Class 1'
        ))
        
        # Plot both convex hulls in white
        for simplex in hull_0.simplices:
            fig_overlay.add_trace(go.Scatter(
                x=points_0[simplex, 0],
                y=points_0[simplex, 1],
                mode='lines',
                line=dict(color='white', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for simplex in hull_1.simplices:
            fig_overlay.add_trace(go.Scatter(
                x=points_1[simplex, 0],
                y=points_1[simplex, 1],
                mode='lines',
                line=dict(color='white', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Highlight support vectors
        support_vectors = svm.support_vectors_
        fig_overlay.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='circle-open', line=dict(width=3)),
            name='Support Vectors'
        ))
        
        # Plot SVM decision boundary
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig_overlay.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            contours=dict(start=0, end=0, size=1),
            line=dict(color='green', width=3),
            showscale=False,
            name='Decision Boundary'
        ))
    
    fig_overlay.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
        title_text="Support vectors on hull boundaries"
    )
    st.plotly_chart(fig_overlay, use_container_width=True, key="overlay_plot")

# Instructions
st.info("Click 'Generate Random Classes' to see two separated point clouds")

# Quantitative Analysis Section
if len(st.session_state.class_0) >= 3 and len(st.session_state.class_1) >= 3:
    st.markdown("---")
    st.subheader("Quantitative Analysis")
    
    points_0 = np.array(st.session_state.class_0)
    points_1 = np.array(st.session_state.class_1)
    
    X = np.vstack([points_0, points_1])
    y = np.array([0] * len(points_0) + [1] * len(points_1))
    
    svm = SVC(kernel='linear', C=1000)
    svm.fit(X, y)
    
    hull_0 = ConvexHull(points_0)
    hull_1 = ConvexHull(points_1)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Support Vectors", len(svm.support_vectors_))
    
    with col2:
        # Margin width: 2 / ||w||
        margin = 2.0 / np.linalg.norm(svm.coef_)
        st.metric("Margin Width", f"{margin:.4f}")
    
    with col3:
        st.metric("Class 0 Hull Vertices", len(hull_0.vertices))
    
    with col4:
        st.metric("Class 1 Hull Vertices", len(hull_1.vertices))
    
    # Geometric equivalence verification
    st.markdown("**Geometric Equivalence Verification:**")
    
    # Check what percentage of support vectors lie on convex hull boundaries
    hull_0_vertices = set(hull_0.vertices)
    hull_1_vertices = set(hull_1.vertices + len(points_0))  # Offset for second class
    
    sv_indices = set(svm.support_)
    
    # Count SVs that are on hull vertices
    sv_on_hull_0 = len(sv_indices & hull_0_vertices)
    sv_on_hull_1 = len(sv_indices & hull_1_vertices)
    sv_on_hull_total = sv_on_hull_0 + sv_on_hull_1
    
    sv_on_hull_pct = (sv_on_hull_total / len(svm.support_)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Support Vectors on Hull", 
            f"{sv_on_hull_pct:.1f}%",
            help="Percentage of support vectors that lie on convex hull boundaries"
        )
    
    with col2:
        # Distance between closest points on hulls
        min_dist = float('inf')
        for v0_idx in hull_0.vertices:
            for v1_idx in hull_1.vertices:
                dist = np.linalg.norm(points_0[v0_idx] - points_1[v1_idx])
                min_dist = min(min_dist, dist)
        
        st.metric(
            "Min Hull-to-Hull Distance", 
            f"{min_dist:.4f}",
            help="Shortest distance between the two convex hulls"
        )
    
    with col3:
        # Training accuracy
        accuracy = svm.score(X, y)
        st.metric("Training Accuracy", f"{accuracy:.1%}")
    
    # Computational comparison
    st.markdown("**Computational Performance:**")
    
    import time
    
    # Time SVM training
    start = time.time()
    for _ in range(50):
        svm_temp = SVC(kernel='linear', C=1000)
        svm_temp.fit(X, y)
    svm_time = (time.time() - start) / 50 * 1000
    
    # Time convex hull computation
    start = time.time()
    for _ in range(50):
        hull_0_temp = ConvexHull(points_0)
        hull_1_temp = ConvexHull(points_1)
    hull_time = (time.time() - start) / 50 * 1000
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("SVM Training Time", f"{svm_time:.3f} ms")
    with col2:
        st.metric("Convex Hull Time", f"{hull_time:.3f} ms")

# Description of why it works
st.markdown("---")
st.subheader("Why Does This Happen?")
st.markdown("""
**Support Vector Machines** find the maximum-margin hyperplane that separates two classes. 
Geometrically, this is equivalent to finding the shortest line segment connecting the convex hulls of each class, 
then drawing a perpendicular bisector.

**Key insight:** The support vectors (highlighted in yellow) are always points that lie on the boundary 
of their class's convex hull. These are the "critical points" that define the decision boundary.

**Watch:** The decision boundary is always equidistant from the closest points on each hull.
""")

# Manual point input
with st.expander("Add points manually"):
    class_choice = st.radio("Add to which class?", ["Class 0 (Blue)", "Class 1 (Red)"])
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_coord = st.number_input("X coordinate", 0.0, 1.0, 0.5, step=0.1, key="x_input")
    with col_y:
        y_coord = st.number_input("Y coordinate", 0.0, 1.0, 0.5, step=0.1, key="y_input")
    
    if st.button("Add Point"):
        if "Class 0" in class_choice:
            st.session_state.class_0.append([x_coord, y_coord])
        else:
            st.session_state.class_1.append([x_coord, y_coord])
        st.rerun()

st.write(f"Class 0 points: {len(st.session_state.class_0)} | Class 1 points: {len(st.session_state.class_1)}")