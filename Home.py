import streamlit as st

st.set_page_config(
    page_title="Geometric Foundations of Machine Learning",
    layout="wide"
)

st.title("Geometric Foundations of Machine Learning")

st.markdown("""
Interactive visualizations demonstrating the connections between computational geometry and machine learning algorithms.

### Demonstrations:

Use the sidebar to navigate between the three algorithm pairings.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Voronoi & k-means")
    st.write("k-means clustering is geometrically equivalent to iteratively computing Voronoi diagrams.")

with col2:
    st.subheader("Convex Hulls & SVM")
    st.write("Support Vector Machines find maximum-margin separating hyperplanes between convex hulls of classes.")

with col3:
    st.subheader("Delaunay & k-NN")
    st.write("Delaunay triangulation provides an efficient structure for k-nearest neighbor queries.")

st.markdown("---")
st.caption("Built with Python · Streamlit · scipy.spatial · scikit-learn")