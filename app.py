import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
import base64
import re

# Set page config
st.set_page_config(
    page_title="DEG Analysis Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pattern-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .criteria-box {
        background-color: #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 3px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def parse_sample_patterns(pattern_text, available_samples):
    """
    Parse sample patterns and return matching samples
    Supports:
    - Exact names: Sample1, Sample2
    - Patterns: Control*, *_treated, *Control*
    - Ranges: 1-5, 10,15,20 (0-based indexing)
    """
    if not pattern_text.strip():
        return []
    
    selected_samples = []
    patterns = [p.strip() for p in pattern_text.split(',')]
    
    for pattern in patterns:
        if not pattern:
            continue
            
        # Check if it's an index range (e.g., "1-5" or "10")
        if pattern.replace('-', '').replace(' ', '').isdigit() or '-' in pattern:
            try:
                if '-' in pattern:
                    start, end = map(int, pattern.split('-'))
                    indices = list(range(start-1, min(end, len(available_samples))))  # Convert to 0-based
                else:
                    indices = [int(pattern)-1]  # Convert to 0-based
                
                for idx in indices:
                    if 0 <= idx < len(available_samples):
                        selected_samples.append(available_samples[idx])
            except:
                continue
        
        # Check if it's a wildcard pattern
        elif '*' in pattern:
            pattern_regex = pattern.replace('*', '.*')
            for sample in available_samples:
                if re.match(pattern_regex, sample, re.IGNORECASE):
                    selected_samples.append(sample)
        
        # Exact match
        else:
            if pattern in available_samples:
                selected_samples.append(pattern)
    
    return list(set(selected_samples))  # Remove duplicates

def perform_pca_analysis(df, control_samples, treatment_samples, n_components=2):
    """
    Perform PCA analysis on the data
    """
    # Filter samples
    all_samples = control_samples + treatment_samples
    df_filtered = df[all_samples].copy()
    
    # Remove genes with zero variance
    df_filtered = df_filtered.loc[df_filtered.var(axis=1) > 0]
    
    # Log transform (add pseudocount to avoid log(0))
    df_log = np.log2(df_filtered + 1)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_log.T),
        index=df_log.columns,
        columns=df_log.index
    )
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, len(all_samples)-1, df_scaled.shape[1]))
    pca_result = pca.fit_transform(df_scaled)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(pca_result, index=all_samples)
    pca_df.columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
    
    # Add group information
    pca_df['Group'] = ['Control' if sample in control_samples else 'Treatment' 
                       for sample in all_samples]
    
    return pca_df, pca.explained_variance_ratio_

def create_pca_plot(pca_df, explained_variance):
    """
    Create PCA visualization
    """
    fig = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2',
        color='Group',
        hover_name=pca_df.index,
        title=f'PCA Plot (PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%})',
        color_discrete_map={'Control': '#4444FF', 'Treatment': '#FF4444'}
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        width=800,
        height=600,
        template="plotly_white"
    )
    
    return fig

def perform_deg_analysis(df, control_samples, treatment_samples, 
                         log2fc_threshold=2.0, pvalue_threshold=0.05):
    """
    Perform differential expression analysis with proper statistical criteria:
    - Adjusted p-value < 0.05 for statistical significance
    - |Log2FC| >= 2 for biological significance (4-fold change)
    Both conditions must be met for a gene to be considered differentially expressed
    """
    results = []
    
    for gene in df.index:
        control_values = df.loc[gene, control_samples].values
        treatment_values = df.loc[gene, treatment_samples].values
        
        # Remove NaN values and ensure values are non-negative
        control_clean = control_values[~np.isnan(control_values) & (control_values >= 0)]
        treatment_clean = treatment_values[~np.isnan(treatment_values) & (treatment_values >= 0)]
        
        # Add a small pseudocount (0.1) for log2FC calculation stability
        control_mean_for_log = np.mean(control_clean) if len(control_clean) > 0 else 0
        treatment_mean_for_log = np.mean(treatment_clean) if len(treatment_clean) > 0 else 0

        control_mean_for_log = control_mean_for_log + 0.1 if control_mean_for_log == 0 else control_mean_for_log
        treatment_mean_for_log = treatment_mean_for_log + 0.1 if treatment_mean_for_log == 0 else treatment_mean_for_log

        if len(control_clean) < 2 or len(treatment_clean) < 2:
            results.append({
                'Gene': gene,
                'Control_Mean': np.nan, 
                'Treatment_Mean': np.nan, 
                'Log2FC': np.nan,
                'P_value': np.nan
            })
            continue
            
        log2fc = np.log2(treatment_mean_for_log / control_mean_for_log)
            
        try:
            _, pvalue = ttest_ind(control_clean, treatment_clean, equal_var=False) # Welch's t-test
        except Exception as e:
            results.append({
                'Gene': gene,
                'Control_Mean': np.nan, 
                'Treatment_Mean': np.nan, 
                'Log2FC': log2fc,
                'P_value': np.nan
            })
            continue
            
        results.append({
            'Gene': gene,
            'Control_Mean': control_mean_for_log - 0.1,
            'Treatment_Mean': treatment_mean_for_log - 0.1,
            'Log2FC': log2fc,
            'P_value': pvalue
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        valid_p_values_df = results_df.dropna(subset=['P_value'])
        
        if not valid_p_values_df.empty:
            _, padj_values, _, _ = multipletests(valid_p_values_df['P_value'], method='fdr_bh')
            results_df['P_adj'] = np.nan
            results_df.loc[valid_p_values_df.index, 'P_adj'] = padj_values
        else:
            results_df['P_adj'] = np.nan
        
        regulation = []
        for _, row in results_df.iterrows():
            is_statistically_significant = (not pd.isna(row['P_adj'])) and (row['P_adj'] <= pvalue_threshold)
            is_biologically_significant = (not pd.isna(row['Log2FC'])) and (abs(row['Log2FC']) >= log2fc_threshold)
            
            if is_statistically_significant and is_biologically_significant:
                if row['Log2FC'] > 0:
                    regulation.append("Up-regulated")
                else:
                    regulation.append("Down-regulated")
            else:
                regulation.append("Not significant")
        
        results_df['Regulation'] = regulation
        
        classification = []
        for _, row in results_df.iterrows():
            is_statistically_significant = (not pd.isna(row['P_adj'])) and (row['P_adj'] <= pvalue_threshold)
            is_biologically_significant = (not pd.isna(row['Log2FC'])) and (abs(row['Log2FC']) >= log2fc_threshold)

            if is_statistically_significant and is_biologically_significant:
                classification.append("Significant DEG")
            elif is_statistically_significant:
                classification.append("Statistically significant only")
            elif is_biologically_significant:
                classification.append("Large fold change only")
            else:
                classification.append("Not significant")
        
        results_df['Classification'] = classification
    
    return results_df

def create_volcano_plot(deg_results, log2fc_threshold=2.0, pvalue_threshold=0.05):
    """
    Create volcano plot with proper thresholds for DEG identification
    """
    deg_results['neg_log10_padj'] = -np.log10(deg_results['P_adj'])
    
    # Color mapping with enhanced visualization
    colors = []
    sizes = []
    for _, row in deg_results.iterrows():
        if row['Regulation'] == 'Up-regulated':
            colors.append('#FF4444')  # Red for up-regulated
            sizes.append(8)
        elif row['Regulation'] == 'Down-regulated':
            colors.append('#4444FF')  # Blue for down-regulated
            sizes.append(8)
        elif row['Classification'] == 'Statistically significant only':
            colors.append('#FFA500')  # Orange for stat sig only
            sizes.append(6)
        elif row['Classification'] == 'Large fold change only':
            colors.append('#9370DB')  # Purple for large FC only
            sizes.append(6)
        else:
            colors.append('#CCCCCC')  # Gray for not significant
            sizes.append(4)
    
    fig = go.Figure()
    
    # Add scatter plot with enhanced hover information
    fig.add_trace(go.Scatter(
        x=deg_results['Log2FC'],
        y=deg_results['neg_log10_padj'],
        mode='markers',
        marker=dict(
            color=colors,
            size=sizes,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        text=deg_results['Gene'],
        customdata=deg_results[['P_adj', 'Classification']],
        hovertemplate='<b>%{text}</b><br>' +
                     'Log2FC: %{x:.3f}<br>' +
                     'Adjusted P-value: %{customdata[0]:.2e}<br>' +
                     '-log10(p-adj): %{y:.2f}<br>' +
                     'Status: %{customdata[1]}<br>' +
                     '<extra></extra>',
        showlegend=False
    ))
    
    # Add threshold lines with labels
    fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", 
                  line_color="red", opacity=0.8, line_width=2,
                  annotation_text=f"p-adj = {pvalue_threshold}", 
                  annotation_position="bottom right")
    
    fig.add_vline(x=log2fc_threshold, line_dash="dash", 
                  line_color="red", opacity=0.8, line_width=2,
                  annotation_text=f"Log2FC = {log2fc_threshold}", 
                  annotation_position="top")
    
    fig.add_vline(x=-log2fc_threshold, line_dash="dash", 
                  line_color="red", opacity=0.8, line_width=2,
                  annotation_text=f"Log2FC = -{log2fc_threshold}", 
                  annotation_position="top")
    
    # Add quadrant labels
    max_x = deg_results['Log2FC'].max()
    max_y = deg_results['neg_log10_padj'].max()
    
    fig.add_annotation(x=max_x*0.8, y=max_y*0.9, text="Up-regulated DEGs", 
                      showarrow=False, font=dict(color="red", size=14, family="Arial Black"))
    fig.add_annotation(x=-max_x*0.8, y=max_y*0.9, text="Down-regulated DEGs", 
                      showarrow=False, font=dict(color="blue", size=14, family="Arial Black"))
    
    fig.update_layout(
        title={
            'text': f"Volcano Plot - DEG Analysis<br><sub>Criteria: |Log2FC| ≥ {log2fc_threshold} AND p-adj < {pvalue_threshold}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Log2 Fold Change",
        yaxis_title="-Log10(Adjusted P-value)",
        width=900,
        height=700,
        template="plotly_white",
        font=dict(size=12)
    )
    
    return fig

def create_summary_plots(deg_results):
    """
    Create summary plots for DEG analysis results
    """
    class_counts = deg_results['Classification'].value_counts()
    
    classification_order = [
        "Significant DEG",
        "Statistically significant only",
        "Large fold change only",
        "Not significant"
    ]
    class_counts = class_counts.reindex(classification_order).dropna()

    classification_colors = {
        "Significant DEG": '#5a4ffc', 
        "Statistically significant only": '#FFA500', 
        "Large fold change only": '#9370DB', 
        "Not significant": '#CCCCCC'
    }
    pie_colors = [classification_colors[label] for label in class_counts.index]

    fig_pie = go.Figure(data=[go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.4,
        marker_colors=pie_colors
    )])
    
    fig_pie.update_layout(
        title="Gene Classification Distribution",
        width=500,
        height=400
    )
    
    reg_counts = deg_results['Regulation'].value_counts()
    
    regulation_order = ["Up-regulated", "Down-regulated", "Not significant"]
    reg_counts = reg_counts.reindex(regulation_order).dropna()

    regulation_bar_colors = {
        "Up-regulated": '#FF4444', 
        "Down-regulated": '#4444FF', 
        "Not significant": '#CCCCCC'
    }
    bar_colors = [regulation_bar_colors[label] for label in reg_counts.index]

    fig_bar = go.Figure(data=[go.Bar(
        x=reg_counts.index,
        y=reg_counts.values,
        marker_color=bar_colors
    )])
    
    fig_bar.update_layout(
        title="Differential Expression Status",
        xaxis_title="Regulation Status",
        yaxis_title="Number of Genes",
        width=500,
        height=400
    )
    
    return fig_pie, fig_bar

def main():
    st.markdown('<h1 class="main-header">🧬 Enhanced Differential Gene Expression Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="criteria-box">
    <h3>🎯 DEG Identification Criteria</h3>
    <p><strong>A gene is considered differentially expressed if BOTH conditions are met:</strong></p>
    <ul>
        <li><strong>Statistical Significance:</strong> Adjusted p-value (FDR) < 0.05</li>
        <li><strong>Biological Significance:</strong> |Log2 Fold Change| ≥ 2 </li>
    </ul>
    <p><em>This ensures only genes with both statistical reliability and biological relevance are identified as DEGs.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Upload your RNA-seq count matrix and define sample groups to identify differentially expressed genes
    using rigorous statistical criteria.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Analysis Parameters")
    st.sidebar.markdown("**DEG Identification Thresholds:**")
    
    log2fc_threshold = st.sidebar.slider(
        "Log2 Fold Change Threshold", 
        1.0, 4.0, 2.0, 0.1,
        help="Minimum |Log2FC| for biological significance (2 = 4-fold change)"
    )
    
    pvalue_threshold = st.sidebar.slider(
        "Adjusted P-value Threshold", 
        0.001, 0.1, 0.05, 0.001,
        help="Maximum adjusted p-value for statistical significance"
    )
    
    st.sidebar.markdown(f"""
    **Current Criteria:**
    - |Log2FC| ≥ {log2fc_threshold} ({2**log2fc_threshold:.1f}-fold change)
    - p-adj < {pvalue_threshold}
    """)
    
    # File upload
    st.markdown('<h2 class="sub-header">📁 Upload RNA-seq Count Data</h2>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing RNA-seq counts (genes as rows, samples as columns)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file, index_col=0)
            
            # Data validation
            if df.shape[0] == 0:
                st.error("The uploaded file appears to be empty.")
                return
            
            # Validate that all values are numeric and non-negative
            
            numeric_df = df.apply(pd.to_numeric, errors='coerce')

            if numeric_df.isnull().values.any():
                st.error("Detected non-numeric values in your count data that cannot be interpreted as numbers. Please ensure all count values are numerical (e.g., no text like 'NA', 'missing').")
                return

            # After ensuring all are numeric, check for non-negativity
            if (numeric_df < 0).any().any():
                st.error("All count values must be non-negative. Detected negative values in your data.")
                return
            
            # Use the coerced numeric df for further processing
            df = numeric_df

            if df.select_dtypes(include=[np.number]).shape[1] < 4:
                st.error("Need at least 4 numeric columns (2 control + 2 treatment samples minimum).")
                return
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Display data preview and statistics
            with st.expander("📊 Data Preview & Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.write("**Data Statistics:**")
                    st.write(f"- **Genes:** {df.shape[0]:,}")
                    st.write(f"- **Samples:** {df.shape[1]}")
                    numeric_cols = df.select_dtypes(include=[np.number])
                    st.write(f"- **Zero values:** {(numeric_cols == 0).sum().sum():,}")
                    st.write(f"- **Mean expression:** {df.mean().mean():.2f}")
                    st.write(f"- **Max expression:** {df.max().max():.2f}")
            
            # Sample group definition
            st.markdown('<h2 class="sub-header">🏷️ Define Sample Groups</h2>', 
                        unsafe_allow_html=True)
            
            # Show available samples
            st.write("**Available Samples:**")
            sample_list = df.columns.tolist()
            
            # Display samples in a more organized way
            cols_per_row = 6
            for i in range(0, len(sample_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(sample_list):
                        col.write(f"{i+j+1}. {sample_list[i+j]}")
            
            # Selection method
            selection_method = st.radio(
                "Choose selection method:",
                ["Pattern-based Selection", "Manual Selection", "Upload Sample Groups"]
            )
            
            control_samples = []
            treatment_samples = []
            
            if selection_method == "Pattern-based Selection":
                st.markdown('<div class="pattern-input">', unsafe_allow_html=True)
                st.write("**Pattern-based Selection Help:**")
                st.write("""
                - **Wildcards**: `Control*` (starts with Control), `*_ctrl` (ends with _ctrl), `*control*` (contains control)
                - **Exact names**: `Sample1, Sample2, Sample3`
                - **Index ranges**: `1-5` (samples 1 to 5), `10,15,20` (specific indices)
                - **Mixed**: `Control*, 15-20, Sample_X`
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Control Samples Pattern**")
                    control_pattern = st.text_area(
                        "Enter patterns for control samples:",
                        placeholder="e.g., Control*, 1-5, Sample_C1",
                        height=100,
                        key="control_pattern"
                    )
                    
                    if control_pattern:
                        control_samples = parse_sample_patterns(control_pattern, sample_list)
                        st.write(f"**Selected Control Samples ({len(control_samples)}):**")
                        if control_samples:
                            st.write(", ".join(control_samples))
                        else:
                            st.warning("No samples matched the pattern")
                
                with col2:
                    st.write("**Condition Samples Pattern**")
                    treatment_pattern = st.text_area(
                        "Enter patterns for treatment samples:",
                        placeholder="e.g., Treatment*, 6-10, Sample_T*",
                        height=100,
                        key="treatment_pattern"
                    )
                    
                    if treatment_pattern:
                        treatment_samples = parse_sample_patterns(treatment_pattern, sample_list)
                        st.write(f"**Selected Condition Samples ({len(treatment_samples)}):**")
                        if treatment_samples:
                            st.write(", ".join(treatment_samples))
                        else:
                            st.warning("No samples matched the pattern")
            
            elif selection_method == "Manual Selection":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Control Samples**")
                    control_samples = st.multiselect(
                        "Select control samples:",
                        options=sample_list,
                        key="control_manual"
                    )
                
                with col2:
                    st.write("**Treatment Samples**")
                    treatment_samples = st.multiselect(
                        "Select treatment samples:",
                        options=sample_list,
                        key="treatment_manual"
                    )
            
            elif selection_method == "Upload Sample Groups":
                st.write("**Upload a CSV file with sample group definitions**")
                group_file = st.file_uploader(
                    "Upload CSV with columns: 'Sample' and 'Group' (Control/Treatment)",
                    type=['csv'],
                    key="group_file"
                )
                
                if group_file is not None:
                    try:
                        group_df = pd.read_csv(group_file)
                        if 'Sample' in group_df.columns and 'Group' in group_df.columns:
                            control_samples = group_df[group_df['Group'].str.lower().str.contains('control', na=False)]['Sample'].tolist()
                            treatment_samples = group_df[group_df['Group'].str.lower().str.contains('treatment', na=False)]['Sample'].tolist()
                            
                            # Filter to only include samples that exist in the data
                            control_samples = [s for s in control_samples if s in sample_list]
                            treatment_samples = [s for s in treatment_samples if s in sample_list]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Control Samples ({len(control_samples)}):**")
                                if control_samples:
                                    st.write(", ".join(control_samples))
                            with col2:
                                st.write(f"**Treatment Samples ({len(treatment_samples)}):**")
                                if treatment_samples:
                                    st.write(", ".join(treatment_samples))
                        else:
                            st.error("CSV file must contain 'Sample' and 'Group' columns")
                    except Exception as e:
                        st.error(f"Error reading group file: {str(e)}")
                else:
                    st.info("Sample group file format: CSV with 'Sample' and 'Group' columns")
                    sample_group_example = pd.DataFrame({
                        'Sample': ['Sample1', 'Sample2', 'Sample3', 'Sample4'],
                        'Group': ['Control', 'Control', 'Treatment', 'Treatment']
                    })
                    st.dataframe(sample_group_example, use_container_width=True)
            
            # Validation and analysis
            if len(control_samples) >= 2 and len(treatment_samples) >= 2:
                # Check for overlap
                overlap = set(control_samples) & set(treatment_samples)
                if overlap:
                    st.error(f"❌ Samples cannot be in both groups: {', '.join(overlap)}")
                else:
                    # Show sample group summary
                    st.success(f"✅ Sample groups defined: {len(control_samples)} controls, {len(treatment_samples)} treatments")
                    
                    # PCA Analysis Section
                    st.markdown('<h2 class="sub-header">🔬 Principal Component Analysis (PCA)</h2>', 
                                unsafe_allow_html=True)
                    
                    try:
                        pca_df, explained_variance = perform_pca_analysis(df, control_samples, treatment_samples)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            pca_fig = create_pca_plot(pca_df, explained_variance)
                            st.plotly_chart(pca_fig, use_container_width=True)
                        
                        with col2:
                            st.write("**PCA Summary:**")
                            st.write(f"- PC1 explains {explained_variance[0]:.1%} of variance")
                            st.write(f"- PC2 explains {explained_variance[1]:.1%} of variance")
                            st.write(f"- Total variance explained: {sum(explained_variance):.1%}")
                            
                            st.write("**Sample Coordinates:**")
                            st.dataframe(pca_df[['PC1', 'PC2', 'Group']], use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error in PCA analysis: {str(e)}")
                    
                    # DEG Analysis Section
                    st.markdown('<h2 class="sub-header">🧬 Differential Expression Analysis</h2>', 
                                unsafe_allow_html=True)
                    
                    if st.button("🚀 Run DEG Analysis", type="primary"):
                        with st.spinner("Performing differential expression analysis..."):
                            try:
                                # Perform DEG analysis
                                deg_results = perform_deg_analysis(
                                    df, control_samples, treatment_samples,
                                    log2fc_threshold, pvalue_threshold
                                )
                                
                                if not deg_results.empty:
                                    # Store results in session state
                                    st.session_state['deg_results'] = deg_results
                                    st.session_state['analysis_params'] = {
                                        'log2fc_threshold': log2fc_threshold,
                                        'pvalue_threshold': pvalue_threshold,
                                        'control_samples': control_samples,
                                        'treatment_samples': treatment_samples
                                    }
                                    
                                    st.success("✅ DEG analysis completed successfully!")
                                else:
                                    st.error("❌ DEG analysis failed - no results generated")
                                    
                            except Exception as e:
                                st.error(f"❌ Error in DEG analysis: {str(e)}")
                    
                    # Display results if they exist
                    if 'deg_results' in st.session_state:
                        deg_results = st.session_state['deg_results']
                        params = st.session_state['analysis_params']
                        
                        # Results Summary
                        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', 
                                    unsafe_allow_html=True)
                        
                        # Summary metrics
                        total_genes = len(deg_results)
                        significant_degs = len(deg_results[deg_results['Regulation'] != 'Not significant'])
                        upregulated = len(deg_results[deg_results['Regulation'] == 'Up-regulated'])
                        downregulated = len(deg_results[deg_results['Regulation'] == 'Down-regulated'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f'''
                            <div class="container">
                                <h3 style="color: #ff7f0e; margin: 0;">{total_genes:,}</h3>
                                <p style="margin: 0;">Total Genes</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'''
                            <div class="container">
                                <h3 style="color: #17becf; margin: 0;">{significant_degs:,}</h3>
                                <p style="margin: 0;">Significant DEGs</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'''
                            <div class="container">
                                <h3 style="color: #2ca02c; margin: 0;">{upregulated:,}</h3>
                                <p style="margin: 0;">Up-regulated</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f'''
                            <div class="container">
                                <h3 style="color: #d62728; margin: 0;">{downregulated:,}</h3>
                                <p style="margin: 0;">Down-regulated</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Volcano Plot
                        st.markdown('<h3 class="sub-header">🌋 Volcano Plot</h3>', 
                                    unsafe_allow_html=True)
                        
                        volcano_fig = create_volcano_plot(
                            deg_results, 
                            params['log2fc_threshold'], 
                            params['pvalue_threshold']
                        )
                        st.plotly_chart(volcano_fig, use_container_width=True)
                        
                        # Summary plots
                        st.markdown('<h3 class="sub-header">📈 Summary Plots</h3>', 
                                    unsafe_allow_html=True)
                        
                        pie_fig, bar_fig = create_summary_plots(deg_results)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(pie_fig, use_container_width=True)
                        with col2:
                            st.plotly_chart(bar_fig, use_container_width=True)
                        
                        # Top DEGs tables
                        st.markdown('<h3 class="sub-header">🏆 Top Differentially Expressed Genes</h3>', 
                                    unsafe_allow_html=True)
                        
                        significant_deg_df = deg_results[deg_results['Regulation'] != 'Not significant'].copy()
                        
                        if not significant_deg_df.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Top 10 Up-regulated Genes**")
                                top_up = deg_results[deg_results['Regulation'] == 'Up-regulated'].nlargest(10, 'Log2FC')
                                if not top_up.empty:
                                    display_up = top_up[['Gene', 'Log2FC', 'P_adj']].copy()
                                    display_up['Log2FC'] = display_up['Log2FC'].round(3)
                                    display_up['P_adj'] = display_up['P_adj'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
                                    st.dataframe(display_up, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No significantly up-regulated genes found")
                            
                            with col2:
                                st.write("**Top 10 Down-regulated Genes**")
                                top_down = deg_results[deg_results['Regulation'] == 'Down-regulated'].nsmallest(10, 'Log2FC')
                                if not top_down.empty:
                                    display_down = top_down[['Gene', 'Log2FC', 'P_adj']].copy()
                                    display_down['Log2FC'] = display_down['Log2FC'].round(3)
                                    display_down['P_adj'] = display_down['P_adj'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
                                    st.dataframe(display_down, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No significantly down-regulated genes found")
                        else:
                            st.info("No significantly differentially expressed genes found with current thresholds")
                        
                        # Full results table
                        st.markdown('<h3 class="sub-header">📋 Complete Results Table</h3>', 
                                    unsafe_allow_html=True)
                        
                        # Filter options
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            regulation_filter = st.selectbox(
                                "Filter by Regulation:",
                                ["All", "Up-regulated", "Down-regulated", "Not significant"]
                            )
                        
                        with col2:
                            classification_filter = st.selectbox(
                                "Filter by Classification:",
                                ["All", "Significant DEG", "Statistically significant only", 
                                 "Large fold change only", "Not significant"]
                            )
                        
                        with col3:
                            show_top_n = st.selectbox(
                                "Show top N genes:",
                                [100, 500, 1000, "All"],
                                index=0
                            )
                        
                        # Apply filters
                        filtered_results = deg_results.copy()
                        
                        if regulation_filter != "All":
                            filtered_results = filtered_results[filtered_results['Regulation'] == regulation_filter]
                        
                        if classification_filter != "All":
                            filtered_results = filtered_results[filtered_results['Classification'] == classification_filter]
                        
                        # Sort by absolute Log2FC and limit results
                        if not filtered_results.empty:
                            filtered_results['abs_log2fc'] = abs(filtered_results['Log2FC'])
                            filtered_results = filtered_results.sort_values('abs_log2fc', ascending=False)
                            
                            if show_top_n != "All":
                                filtered_results = filtered_results.head(show_top_n)
                            
                            # Prepare display dataframe
                            display_df = filtered_results[['Gene', 'Control_Mean', 'Treatment_Mean', 
                                                         'Log2FC', 'P_value', 'P_adj', 
                                                         'Regulation', 'Classification']].copy()
                            
                            # Format numeric columns
                            for col in ['Control_Mean', 'Treatment_Mean', 'Log2FC']:
                                display_df[col] = display_df[col].round(3)
                            
                            for col in ['P_value', 'P_adj']:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            st.write(f"Showing {len(display_df)} of {len(deg_results)} total genes")
                        else:
                            st.info("No genes match the selected filters")
                        
                        # Download results
                        st.markdown('<h3 class="sub-header">💾 Download Results</h3>', 
                                    unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Complete results
                            csv_buffer = io.StringIO()
                            deg_results.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="📁 Download Complete Results",
                                data=csv_buffer.getvalue(),
                                file_name=f"deg_analysis_complete_results.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Significant DEGs only
                            if not significant_deg_df.empty:
                                csv_buffer_sig = io.StringIO()
                                significant_deg_df.drop('abs_log2fc', axis=1, errors='ignore').to_csv(csv_buffer_sig, index=False)
                                
                                st.download_button(
                                    label="🎯 Download Significant DEGs",
                                    data=csv_buffer_sig.getvalue(),
                                    file_name=f"significant_degs_only.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No significant DEGs to download")
                        
                        with col3:
                            # Analysis summary
                            summary_data = {
                                'Parameter': ['Total Genes', 'Significant DEGs', 'Up-regulated', 'Down-regulated',
                                            'Log2FC Threshold', 'P-value Threshold', 'Control Samples', 'Treatment Samples'],
                                'Value': [total_genes, significant_degs, upregulated, downregulated,
                                        params['log2fc_threshold'], params['pvalue_threshold'],
                                        len(params['control_samples']), len(params['treatment_samples'])]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            
                            csv_buffer_summary = io.StringIO()
                            summary_df.to_csv(csv_buffer_summary, index=False)
                            
                            st.download_button(
                                label="📊 Download Analysis Summary",
                                data=csv_buffer_summary.getvalue(),
                                file_name=f"analysis_summary.csv",
                                mime="text/csv"
                            )
                        
                        # Gene Set Analysis Preparation
                        st.markdown('<h3 class="sub-header">🧬 Gene Lists for Pathway Analysis</h3>', 
                                    unsafe_allow_html=True)
                        
                        if not significant_deg_df.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                up_genes = deg_results[deg_results['Regulation'] == 'Up-regulated']['Gene'].tolist()
                                if up_genes:
                                    st.write(f"**Up-regulated Genes ({len(up_genes)}):**")
                                    up_genes_text = '\n'.join(map(str, up_genes))
                                    st.text_area("Copy for pathway analysis:", up_genes_text, height=150, key="up_genes")
                                    
                                    st.download_button(
                                        label="📄 Download Up-regulated Gene List",
                                        data=up_genes_text,
                                        file_name="upregulated_genes.txt",
                                        mime="text/plain"
                                    )
                            
                            with col2:
                                down_genes = deg_results[deg_results['Regulation'] == 'Down-regulated']['Gene'].tolist()
                                if down_genes:
                                    st.write(f"**Down-regulated Genes ({len(down_genes)}):**")
                                    down_genes_text = '\n'.join(map(str, down_genes))
                                    st.text_area("Copy for pathway analysis:", down_genes_text, height=150, key="down_genes")
                                    
                                    st.download_button(
                                        label="📄 Download Down-regulated Gene List",
                                        data=down_genes_text,
                                        file_name="downregulated_genes.txt",
                                        mime="text/plain"
                                    )
                        
                        # Analysis Notes
                        st.markdown('<h3 class="sub-header">📝 Analysis Notes</h3>', 
                                    unsafe_allow_html=True)
                        
                        with st.expander("ℹ️ Analysis Details & Interpretation"):
                            st.write(f"""
                            **Analysis Parameters Used:**
                            - Log2 Fold Change Threshold: ≥ {params['log2fc_threshold']} ({2**params['log2fc_threshold']:.1f}-fold change)
                            - Adjusted P-value Threshold: < {params['pvalue_threshold']}
                            - Statistical Test: Welch's t-test (unequal variances)
                            - Multiple Testing Correction: Benjamini-Hochberg (FDR)
                            - Control Samples: {len(params['control_samples'])} ({', '.join(params['control_samples'])})
                            - Treatment Samples: {len(params['treatment_samples'])} ({', '.join(params['treatment_samples'])})
                            
                            **Gene Classification:**
                            - **Significant DEG**: Meets both statistical (p-adj < {params['pvalue_threshold']}) and biological (|Log2FC| ≥ {params['log2fc_threshold']}) significance criteria
                            - **Statistically significant only**: p-adj < {params['pvalue_threshold']} but |Log2FC| < {params['log2fc_threshold']}
                            - **Large fold change only**: |Log2FC| ≥ {params['log2fc_threshold']} but p-adj ≥ {params['pvalue_threshold']}
                            - **Not significant**: Neither criterion met
                            
                            **Interpretation Guidelines:**
                            - Focus on "Significant DEG" category for downstream analysis
                            - Up-regulated genes have positive Log2FC values
                            - Down-regulated genes have negative Log2FC values
                            - Use gene lists for pathway enrichment analysis (GSEA, DAVID, etc.)
                            """)
            
            else:
                if len(control_samples) < 2:
                    st.warning("⚠️ Please select at least 2 control samples")
                if len(treatment_samples) < 2:
                    st.warning("⚠️ Please select at least 2 treatment samples")
        except Exception as e:
            raise e 
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show example data format
        with st.expander("📋 Expected Data Format"):
            st.write("Your CSV file should have:")
            st.write("- **Rows**: Gene names/IDs")
            st.write("- **Columns**: Sample names")  
            st.write("- **Values**: Raw or normalized count data (non-negative numbers)")
            
            example_data = pd.DataFrame({
                'Control_1': [100, 250, 50, 0],
                'Control_2': [120, 300, 45, 2],
                'Control_3': [90, 280, 60, 1],
                'Treatment_1': [500, 150, 40, 15],
                'Treatment_2': [480, 180, 35, 12],
                'Treatment_3': [520, 160, 42, 18]
            }, index=['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D'])
            
            st.write("**Example format:**")
            st.dataframe(example_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🧬 Enhanced DEG Analysis Tool | Built with Streamlit | 
        Statistical significance + Biological relevance = Reliable DEGs
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()