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
    # Generate two separated clusters
    np.random.seed(42)
    st.session_state.class_0 = (np.random.randn(15, 2) * 0.15 + [0.3, 0.3]).tolist()
    st.session_state.class_1 = (np.random.randn(15, 2) * 0.15 + [0.7, 0.7]).tolist()
    st.rerun()

# Main plotting area
col1, col2 = st.columns(2)

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
        height=500,
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
        svm = SVC(kernel='linear', C=1000)  # High C for hard margin
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
        # Create a mesh
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary (Z=0)
        fig_svm.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            contours=dict(
                start=0,
                end=0,
                size=1,
            ),
            line=dict(color='green', width=3),
            showscale=False,
            name='Decision Boundary'
        ))
        
        # Plot margins (Z=-1 and Z=1)
        fig_svm.add_trace(go.Contour(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            z=Z,
            contours=dict(
                start=-1,
                end=1,
                size=2,
            ),
            line=dict(color='gray', width=1, dash='dash'),
            showscale=False,
            name='Margins'
        ))
    
    fig_svm.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_svm, use_container_width=True, key="svm_plot")

# Instructions
st.info("Click 'Generate Random Classes' to see two separated point clouds")

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