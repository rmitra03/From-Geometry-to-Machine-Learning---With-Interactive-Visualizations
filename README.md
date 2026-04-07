# Geometric Foundations of Machine Learning

Interactive visualizations demonstrating the connections between computational geometry algorithms and machine learning techniques.

## Overview

This project explores how fundamental machine learning algorithms (k-means, SVM, k-NN) are built on geometric principles (Voronoi diagrams, Convex Hulls, Delaunay triangulation). Through interactive demos and quantitative analysis, the visualizations make these abstract connections concrete and intuitive.

## Demonstrations

### 1. Voronoi Diagrams ↔ k-means Clustering

Demonstrates that k-means cluster assignments are equivalent to Voronoi cells around centroids. Each iteration of k-means creates a Voronoi partition of the space.

**Key Insight:** Every point is assigned to its nearest centroid, which is precisely the Voronoi property.

### 2. Convex Hulls ↔ Support Vector Machines

Shows how SVM decision boundaries relate to the convex hulls of each class. Support vectors consistently lie on or near the hull boundaries.

**Key Insight:** The SVM margin is determined by the distance between the closest points on each class's convex hull.

### 3. Delaunay Triangulation ↔ k-Nearest Neighbors

Illustrates how Delaunay triangulation provides an efficient spatial structure for k-NN queries by connecting nearest neighbors.

**Key Insight:** Delaunay edges connect points that are likely to be nearest neighbors, enabling efficient spatial searches.

## Features

- **Interactive Visualizations:** Real-time parameter adjustment and random data generation
- **Side-by-side Comparisons:** Geometry and ML algorithms displayed simultaneously
- **Overlay Views:** Combined visualizations showing geometric equivalence
- **Quantitative Metrics:** Statistical validation of geometric connections
- **Performance Analysis:** Systematic experiments across different dataset sizes

## Tech Stack

- **Python 3.8+**
- **Streamlit:** Web interface
- **scipy.spatial:** Geometric algorithms (Voronoi, ConvexHull, Delaunay)
- **scikit-learn:** Machine learning implementations
- **Plotly:** Interactive visualizations
- **NumPy:** Numerical computations

## Installation

Step 1: Clone the repository:

```bash
git clone https://github.com/rmitra03/From-Geometry-to-Machine-Learning---With-Interactive-Visualizations.git
cd From-Geometry-to-Machine-Learning---With-Interactive-Visualizations
```

Step 2: Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Step 3: Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run Home.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

``` text
├── Home.py                           # Landing page
├── pages/
│   ├── 1_Voronoi_and_kmeans.py       # Voronoi/k-means demo
│   ├── 2_ConvexHull_and_SVM.py       # ConvexHull/SVM demo
│   ├── 3_Delaunay_and_kNN.py         # Delaunay/k-NN demo
│   └── 4_Performance_Analysis.py     # Performance experiments
├── .streamlit/
│   └── config.toml                   # Theme configuration
├── .gitignore                        # Git ignore rules
├── LICENSE                           # License file
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── venv/                             # Virtual environment (not in git)
```

## Metrics & Validation

Each demonstration includes comprehensive quantitative metrics:

**Voronoi/k-means:**

- Silhouette scores
- Inertia (WCSS)
- Voronoi property verification (% of points assigned to nearest centroid)
- Convergence iteration counts
- Timing comparisons

**ConvexHull/SVM:**

- Support vector counts
- Percentage of support vectors on hull boundaries
- Margin widths
- Hull vertex counts

**Delaunay/k-NN:**

- Query times
- Classification accuracy
- Neighbor distribution analysis
- Delaunay edge counts

## Performance Analysis

The Performance Analysis page runs systematic experiments showing:

- Voronoi construction vs k-means iteration times
- Support vector distribution across different class separations
- k-NN query efficiency as dataset size scales

## Design

The application uses a custom pastel color scheme for a clean, modern aesthetic:

- Primary: Lavender (#B8A5D6)
- Background: Soft pink/white (#FFF5F7)
- Secondary: Light lavender (#E8DFF5)
- Serif fonts for readability

## Course Context

This project was developed for CS 6319 (Computational Geometry) at UT Dallas, Spring 2026. The goal was to bridge computational geometry theory with practical machine learning applications through interactive visualization.

## Author

Risheeka Mitra  
UT Dallas - MS Computer Science (Data Science)  
Spring 2026

## Acknowledgments

- CS 6319 course materials and lectures
- scipy.spatial and scikit-learn documentation
- Streamlit documentation and community examples
