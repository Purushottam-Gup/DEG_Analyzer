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
import io
import base64
import re
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote

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

@st.cache_data
def get_gene_info(gene_id):
    """
    Get gene symbol and description from NCBI Gene databases
    """
    try:
        # Try NCBI Gene search
        search_url = f"https://www.ncbi.nlm.nih.gov/gene/?term={quote(gene_id)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for gene symbol and description
            symbol = gene_id  # Default to original ID
            description = "No description available"
            
            # Try to find gene symbol
            symbol_element = soup.find('span', class_='gene-symbol')
            if symbol_element:
                symbol = symbol_element.text.strip()
            
            # Try to find description
            desc_element = soup.find('dd', class_='gene-summary')
            if desc_element:
                description = desc_element.text.strip()
            
            return symbol, description
        
    except Exception as e:
        pass
    
    # Fallback: try to extract symbol from gene ID patterns
    symbol = gene_id
    description = "No description available"
    
    return symbol, description

def get_gene_symbols_batch(gene_ids, progress_bar=None):
    """
    Get gene symbols and descriptions for a batch of gene IDs
    """
    results = {}
    total = len(gene_ids)
    
    for i, gene_id in enumerate(gene_ids):
        if progress_bar:
            progress_bar.progress((i + 1) / total)
        
        symbol, description = get_gene_info(gene_id)
        results[gene_id] = {
            'Symbol': symbol,
            'Description': description
        }
        
        # Add small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    return results

def parse_sample_patterns(pattern_text, available_samples):
    """
    Parse sample patterns and return matching samples
    Supports:
    - Exact names: Sample1, Sample2
    - Patterns: Control*, *_treated, *Control*
    - Ranges: Sample1-Sample10
    - Indices: 1-5, 10,15,20
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
        
        # Remove zeros and NaN values, then add small pseudocount for log calculations
        control_clean = control_values[~np.isnan(control_values) & (control_values >= 0)]
        treatment_clean = treatment_values[~np.isnan(treatment_values) & (treatment_values >= 0)]
        
        # Add pseudocount (0.1) to avoid log(0) issues
        control_clean = control_clean + 0.1
        treatment_clean = treatment_clean + 0.1
        
        if len(control_clean) < 2 or len(treatment_clean) < 2:
            continue
            
        # Calculate means
        control_mean = np.mean(control_clean)
        treatment_mean = np.mean(treatment_clean)
        
        # Calculate log2 fold change
        log2fc = np.log2(treatment_mean / control_mean)
            
        # Perform t-test
        try:
            _, pvalue = ttest_ind(control_clean, treatment_clean)
        except:
            continue
            
        results.append({
            'Gene': gene,
            'Control_Mean': control_mean - 0.1,  # Remove pseudocount for display
            'Treatment_Mean': treatment_mean - 0.1,  # Remove pseudocount for display
            'Log2FC': log2fc,
            'P_value': pvalue
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Calculate adjusted p-values using Benjamini-Hochberg method (FDR correction)
        _, padj_values, _, _ = multipletests(results_df['P_value'], method='fdr_bh')
        results_df['P_adj'] = padj_values
        
        # Apply STRICT differential expression criteria:
        # 1. Adjusted p-value < 0.05 (statistical significance)
        # 2. |Log2FC| >= 2 (biological significance - 4-fold change)
        regulation = []
        for _, row in results_df.iterrows():
            # Both conditions must be satisfied
            is_statistically_significant = row['P_adj'] <= pvalue_threshold
            is_biologically_significant = abs(row['Log2FC']) >= log2fc_threshold
            
            if is_statistically_significant and is_biologically_significant:
                if row['Log2FC'] > 0:
                    regulation.append("Up-regulated")
                else:
                    regulation.append("Down-regulated")
            else:
                regulation.append("Not significant")
        
        results_df['Regulation'] = regulation
        
        # Add additional classification for interpretation
        classification = []
        for _, row in results_df.iterrows():
            if row['P_adj'] <= pvalue_threshold and abs(row['Log2FC']) >= log2fc_threshold:
                classification.append("Significant DEG")
            elif row['P_adj'] <= pvalue_threshold:
                classification.append("Statistically significant only")
            elif abs(row['Log2FC']) >= log2fc_threshold:
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
    # Classification counts
    class_counts = deg_results['Classification'].value_counts()
    
    # Create pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.4,
        marker_colors=['#FF4444', '#4444FF', '#FFA500', '#9370DB', '#CCCCCC']
    )])
    
    fig_pie.update_layout(
        title="Gene Classification Distribution",
        width=500,
        height=400
    )
    
    # Create bar plot for regulation status
    reg_counts = deg_results['Regulation'].value_counts()
    
    fig_bar = go.Figure(data=[go.Bar(
        x=reg_counts.index,
        y=reg_counts.values,
        marker_color=['#FF4444', '#4444FF', '#CCCCCC']
    )])
    
    fig_bar.update_layout(
        title="Differential Expression Status",
        xaxis_title="Regulation Status",
        yaxis_title="Number of Genes",
        width=500,
        height=400
    )
    
    return fig_pie, fig_bar

def annotate_significant_genes(deg_results):
    """
    Annotate significant genes with symbols and descriptions
    """
    # Get significant genes
    sig_genes = deg_results[deg_results['Regulation'] != 'Not significant']['Gene'].tolist()
    
    if len(sig_genes) > 0 and len(sig_genes) <= 1000:  # Reasonable threshold for annotation
        st.info(f"Fetching gene symbols and descriptions for {len(sig_genes)} significant DEGs...")
        
        progress_bar = st.progress(0)
        gene_annotations = get_gene_symbols_batch(sig_genes, progress_bar)
        
        # Map gene IDs to symbols and descriptions
        deg_results['Symbol'] = deg_results['Gene'].map(
            lambda x: gene_annotations.get(x, {}).get('Symbol', x))
        deg_results['Description'] = deg_results['Gene'].map(
            lambda x: gene_annotations.get(x, {}).get('Description', 'No description available'))
        
        progress_bar.empty()
        st.success("Gene annotation completed!")
        
    elif len(sig_genes) > 1000:
        st.warning(f"Too many significant genes ({len(sig_genes)}) for automatic annotation. Skipping gene symbol lookup.")
        # Provide default values when skipping annotation
        deg_results['Symbol'] = deg_results['Gene']  # Use gene ID as symbol
        deg_results['Description'] = 'No description available'
        
    else:
        # No significant genes found
        deg_results['Symbol'] = deg_results['Gene']
        deg_results['Description'] = 'No description available'
    
    return deg_results

def main():
    st.markdown('<h1 class="main-header">🧬 Enhanced Differential Gene Expression Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Display analysis criteria prominently
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
                    st.write(f"- **Zero values:** {(df == 0).sum().sum():,}")
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
                    st.write("**Treatment Samples Pattern**")
                    treatment_pattern = st.text_area(
                        "Enter patterns for treatment samples:",
                        placeholder="e.g., Treatment*, 6-10, Sample_T*",
                        height=100,
                        key="treatment_pattern"
                    )
                    
                    if treatment_pattern:
                        treatment_samples = parse_sample_patterns(treatment_pattern, sample_list)
                        st.write(f"**Selected Treatment Samples ({len(treatment_samples)}):**")
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
                    
                    if st.button("🚀 Run Enhanced DEG Analysis", type="primary"):
                        with st.spinner("Performing differential expression analysis with rigorous statistical criteria..."):
                            # Perform DEG analysis
                            deg_results = perform_deg_analysis(
                                df, control_samples, treatment_samples,
                                log2fc_threshold, pvalue_threshold
                            )
                            
                            if len(deg_results) > 0:
                                # Annotate significant genes
                                deg_results = annotate_significant_genes(deg_results)
                                
                                # Display summary statistics
                                st.markdown('<h2 class="sub-header">📊 Analysis Results Summary</h2>', 
                                           unsafe_allow_html=True)
                                
                                # Enhanced metrics
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                total_genes = len(deg_results)
                                up_regulated = len(deg_results[deg_results['Regulation'] == 'Up-regulated'])
                                down_regulated = len(deg_results[deg_results['Regulation'] == 'Down-regulated'])
                                total_degs = up_regulated + down_regulated
                                stat_sig_only = len(deg_results[deg_results['Classification'] == 'Statistically significant only'])
                                
                                with col1:
                                    st.metric("Total Genes", f"{total_genes:,}", help="Total genes analyzed")
                                with col2:
                                    st.metric("Significant DEGs", f"{total_degs:,}", 
                                             delta=f"{total_degs/total_genes*100:.1f}%", 
                                             help="Genes meeting both statistical and biological significance")
                                with col3:
                                    st.metric("Up-regulated", f"{up_regulated:,}", 
                                             delta=f"{up_regulated/total_genes*100:.1f}%",
                                             help="Significantly up-regulated genes")
                                with col4:
                                    st.metric("Down-regulated", f"{down_regulated:,}", 
                                             delta=f"{down_regulated/total_genes*100:.1f}%",
                                             help="Significantly down-regulated genes")
                                with col5:
                                    st.metric("Stat. Sig. Only", f"{stat_sig_only:,}", 
                                             help="Statistically significant but small fold change")
                                
                                # Create and display plots
                                col1, col2 = st.columns(2)
                                
                                fig_pie, fig_bar = create_summary_plots(deg_results)
                                
                                with col1:
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                with col2:
                                    st.plotly_chart(fig_bar, use_container_width=True)
                                
                                # Enhanced Volcano Plot
                                st.markdown('<h2 class="sub-header">🌋 Enhanced Volcano Plot</h2>', 
                                           unsafe_allow_html=True)
                                volcano_fig = create_volcano_plot(deg_results, log2fc_threshold, pvalue_threshold)
                                st.plotly_chart(volcano_fig, use_container_width=True)
                                
                                st.info("""
                                **Plot Legend:**
                                - 🔴 **Red**: Up-regulated DEGs (meet both |Log2FC| ≥ thresholds AND p-adj < threshold)
                                - 🔵 **Blue**: Down-regulated DEGs (meet both |Log2FC| ≥ thresholds AND p-adj < threshold)
                                - 🟠 **Orange**: Statistically significant only (p-adj < threshold, but |Log2FC| < threshold)
                                - 🟣 **Purple**: Large fold change only (|Log2FC| ≥ threshold, but p-adj > threshold)
                                - ⚫ **Gray**: Not significant (neither condition met)
                                """)
                                
                                # Display results table
                                st.markdown('<h2 class="sub-header">📋 Differential Expression Results Table</h2>', 
                                           unsafe_allow_html=True)
                                
                                st.dataframe(deg_results.sort_values(by='P_adj').head(20), use_container_width=True)
                                
                                with st.expander("View Full Results Table"):
                                    st.dataframe(deg_results.sort_values(by='P_adj'), use_container_width=True)
                                
                                # Download results
                                st.markdown('<h2 class="sub-header">⬇️ Download Results</h2>', 
                                           unsafe_allow_html=True)
                                
                                @st.cache_data
                                def convert_df_to_csv(df):
                                    # Cache the conversion to prevent computation on rerun
                                    return df.to_csv(index=False).encode('utf-8')
                                
                                csv_file = convert_df_to_csv(deg_results)
                                
                                st.download_button(
                                    label="Download DEG Results as CSV",
                                    data=csv_file,
                                    file_name='deg_analysis_results.csv',
                                    mime='text/csv',
                                    help="Download the complete table of differential expression results."
                                )
                                
                                # Optional: Download only significant DEGs
                                significant_degs_df = deg_results[deg_results['Regulation'] != 'Not significant'].copy()
                                if not significant_degs_df.empty:
                                    csv_significant = convert_df_to_csv(significant_degs_df)
                                    st.download_button(
                                        label="Download Significant DEGs Only as CSV",
                                        data=csv_significant,
                                        file_name='significant_degs.csv',
                                        mime='text/csv',
                                        help="Download only the genes classified as Up-regulated or Down-regulated."
                                    )
                                else:
                                    st.info("No significant DEGs found to download separately.")
                                
                            else:
                                st.warning("No differential expression results found with the current criteria. Try adjusting thresholds or checking your data/sample groups.")
                                
            else:
                st.warning("Please select at least 2 samples for both control and treatment groups to proceed with the analysis.")
                if len(control_samples) < 2:
                    st.info(f"Currently selected control samples: {len(control_samples)} (need at least 2)")
                if len(treatment_samples) < 2:
                    st.info(f"Currently selected treatment samples: {len(treatment_samples)} (need at least 2)")

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a file with data.")
        except pd.errors.ParserError:
            st.error("Could not parse the CSV file. Please ensure it is a valid CSV format.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}. Please check your file format and try again.")

if __name__ == '__main__':
    main()