#!/usr/bin/env python3
"""
Beyond-H-Index Survey Candidate Selection Tool
Interactive Streamlit Application

This app allows researchers to interactively adjust selection criteria 
for survey candidate pairs and visualize the results in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from utils.composite_scoring import calculate_composite_score

# Configure Streamlit page
st.set_page_config(
    page_title="Beyond-H-Index Survey Selection Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .metric-number {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9em;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the main researcher dataset"""
    try:
        df = pd.read_csv('../nlp_researcher_metrics.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Main dataset '../nlp_researcher_metrics.csv' not found. Please ensure the file exists in the current directory.")
        return None



def filter_data(df, criteria):
    """Filter data based on selection criteria"""
    df_filtered = df.copy()
    
    # Apply basic quality filters
    if criteria['min_h_index'] > 0:
        df_filtered = df_filtered[df_filtered['h_index'] >= criteria['min_h_index']]
    
    if criteria['min_papers'] > 0:
        papers_col = 'total_papers' if 'total_papers' in df_filtered.columns else 'total_paper_count'
        if papers_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[papers_col] >= criteria['min_papers']]
    
    if criteria['min_career_span'] > 0 and 'career_span' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['career_span'] >= criteria['min_career_span']]
    
    # Remove researchers with insufficient data
    required_cols = ['h_index']
    citation_col = 'total_citations' if 'total_citations' in df_filtered.columns else 'total_citation_count'
    papers_col = 'total_papers' if 'total_papers' in df_filtered.columns else 'total_paper_count'
    
    if citation_col in df_filtered.columns:
        required_cols.append(citation_col)
    if papers_col in df_filtered.columns:
        required_cols.append(papers_col)
    
    df_filtered = df_filtered.dropna(subset=required_cols)
    
    # Calculate composite scores if not present
    if 'composite_score' not in df_filtered.columns:
        max_citations = df_filtered[citation_col].max()
        max_papers = df_filtered[papers_col].max()
        max_h_index = df_filtered['h_index'].max()
        
        df_filtered['composite_score'] = df_filtered.apply(
            lambda row: calculate_composite_score(row), 
            axis=1
        )
    
    # Create rankings
    df_filtered['h_index_rank'] = (
        pd.to_numeric(df_filtered['h_index'], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .rank(ascending=False, method='min', na_option='bottom')
        .astype('Int64')
    )

    df_filtered['composite_rank'] = (
        pd.to_numeric(df_filtered['composite_score'], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .rank(ascending=False, method='min', na_option='bottom')
        .astype('Int64')
    )
    
    return df_filtered

def create_pair_dict(better, worse, better_idx, worse_idx, dramatic_diff, pair_type):
    """Helper function to create standardized pair dictionary"""
    return {
        'pair_type': pair_type,
        'better_researcher': {
            'name': better.get('name', f"Researcher_{better_idx}"),
            'h_index': better['h_index'],
            'h_index_rank': better['h_index_rank'],
            'composite_score': better['composite_score'],
            'composite_rank': better['composite_rank'],
            'total_citations': better.get('total_citations', better.get('total_citation_count', 'N/A')),
            'total_papers': better.get('total_papers', better.get('total_paper_count', 'N/A')),
            'career_span': better.get('career_span', 'N/A')
        },
        'worse_researcher': {
            'name': worse.get('name', f"Researcher_{worse_idx}"),
            'h_index': worse['h_index'],
            'h_index_rank': worse['h_index_rank'],
            'composite_score': worse['composite_score'],
            'composite_rank': worse['composite_rank'],
            'total_citations': worse.get('total_citations', worse.get('total_citation_count', 'N/A')),
            'total_papers': worse.get('total_papers', worse.get('total_paper_count', 'N/A')),
            'career_span': worse.get('career_span', 'N/A')
        },
        'differences': {
            'h_index_rank_diff': abs(better['h_index_rank'] - worse['h_index_rank']),
            'composite_rank_diff': abs(better['composite_rank'] - worse['composite_rank']),
            'h_index_diff': abs(better['h_index'] - worse['h_index']),
            'composite_score_diff': abs(better['composite_score'] - worse['composite_score']),
            'dramatic_diff': dramatic_diff
        }
    }

def select_dramatic_pairs(df, rank_tolerance=10, min_dramatic_diff=100, target_type1=10, target_type2=10):
    """
    Select two types of pairs:
    Type 1: Similar H-index ranks, dramatically different composite ranks
    Type 2: Similar composite ranks, dramatically different H-index ranks
    """
    all_pairs = []
    used_indices = set()
    
    # TYPE 1: Similar H-index ranks, dramatically different composite ranks
    df_sorted_h = df.sort_values('h_index_rank').copy().reset_index(drop=True)
    type1_pairs = []
    
    for i, researcher1 in df_sorted_h.iterrows():
        if len(type1_pairs) >= target_type1:
            break
        if i in used_indices:
            continue
            
        # Look for researchers with similar H-index rank
        h_rank_min = researcher1['h_index_rank'] - rank_tolerance
        h_rank_max = researcher1['h_index_rank'] + rank_tolerance
        
        candidates = df_sorted_h[
            (df_sorted_h['h_index_rank'] >= h_rank_min) & 
            (df_sorted_h['h_index_rank'] <= h_rank_max) &
            (df_sorted_h.index != i) &
            (~df_sorted_h.index.isin(used_indices))
        ]
        
        best_candidate = None
        best_diff = 0
        
        for j, researcher2 in candidates.iterrows():
            composite_diff = abs(researcher1['composite_rank'] - researcher2['composite_rank'])
            
            if composite_diff >= min_dramatic_diff and composite_diff > best_diff:
                best_candidate = (j, researcher2, composite_diff)
                best_diff = composite_diff
        
        if best_candidate is not None:
            j, researcher2, composite_diff = best_candidate
            
            # Create pair with better composite rank first
            if researcher1['composite_rank'] < researcher2['composite_rank']:
                better, worse = researcher1, researcher2
                better_idx, worse_idx = i, j
            else:
                better, worse = researcher2, researcher1
                better_idx, worse_idx = j, i
            
            pair = create_pair_dict(better, worse, better_idx, worse_idx, composite_diff, 'type1_similar_h_dramatic_composite')
            type1_pairs.append(pair)
            used_indices.update([i, j])
    
    # TYPE 2: Similar composite ranks, dramatically different H-index ranks
    df_sorted_c = df.sort_values('composite_rank').copy().reset_index(drop=True)
    type2_pairs = []
    
    for i, researcher1 in df_sorted_c.iterrows():
        if len(type2_pairs) >= target_type2:
            break
        if i in used_indices:
            continue
            
        # Look for researchers with similar composite rank
        c_rank_min = researcher1['composite_rank'] - rank_tolerance
        c_rank_max = researcher1['composite_rank'] + rank_tolerance
        
        candidates = df_sorted_c[
            (df_sorted_c['composite_rank'] >= c_rank_min) & 
            (df_sorted_c['composite_rank'] <= c_rank_max) &
            (df_sorted_c.index != i) &
            (~df_sorted_c.index.isin(used_indices))
        ]
        
        best_candidate = None
        best_diff = 0
        
        for j, researcher2 in candidates.iterrows():
            h_diff = abs(researcher1['h_index_rank'] - researcher2['h_index_rank'])
            
            if h_diff >= min_dramatic_diff and h_diff > best_diff:
                best_candidate = (j, researcher2, h_diff)
                best_diff = h_diff
        
        if best_candidate is not None:
            j, researcher2, h_diff = best_candidate
            
            # Create pair with better H-index rank first
            if researcher1['h_index_rank'] < researcher2['h_index_rank']:
                better, worse = researcher1, researcher2
                better_idx, worse_idx = i, j
            else:
                better, worse = researcher2, researcher1
                better_idx, worse_idx = j, i
            
            pair = create_pair_dict(better, worse, better_idx, worse_idx, h_diff, 'type2_similar_composite_dramatic_h')
            type2_pairs.append(pair)
            used_indices.update([i, j])
    
    # Combine all pairs and assign final IDs
    all_pairs = type1_pairs + type2_pairs
    for i, pair in enumerate(all_pairs, 1):
        pair['pair_id'] = i
    
    return all_pairs, len(type1_pairs), len(type2_pairs)

def create_survey_dataframe(selected_pairs):
    """Convert selected pairs to survey-ready DataFrame"""
    survey_data = []
    for pair in selected_pairs:
        better = pair['better_researcher']
        worse = pair['worse_researcher']
        diff = pair['differences']
        
        # Determine the comparison focus based on pair type
        if pair['pair_type'] == 'type1_similar_h_dramatic_composite':
            comparison_focus = "Similar H-index ranking, dramatically different composite ranking"
            question_context = f"Both researchers have similar citation impact (H-index ranks #{better['h_index_rank']} vs #{worse['h_index_rank']}). Who would you rank higher for overall research excellence?"
        else:  # type2
            comparison_focus = "Similar composite ranking, dramatically different H-index ranking"
            question_context = f"Both researchers have similar multi-dimensional scores (composite ranks #{better['composite_rank']} vs #{worse['composite_rank']}). Who has greater citation impact and influence?"
        
        survey_data.append({
            'pair_id': pair['pair_id'],
            'pair_type': pair['pair_type'],
            'comparison_focus': comparison_focus,
            'researcher_a_name': better['name'],
            'researcher_a_h_index': better['h_index'],
            'researcher_a_h_rank': better['h_index_rank'],
            'researcher_a_composite_rank': better['composite_rank'],
            'researcher_a_composite_score': better['composite_score'],
            'researcher_b_name': worse['name'],
            'researcher_b_h_index': worse['h_index'],
            'researcher_b_h_rank': worse['h_index_rank'],
            'researcher_b_composite_rank': worse['composite_rank'],
            'researcher_b_composite_score': worse['composite_score'],
            'h_rank_difference': diff['h_index_rank_diff'],
            'composite_rank_difference': diff['composite_rank_diff'],
            'dramatic_difference': diff['dramatic_diff'],
            'question_context': question_context
        })
    
    return pd.DataFrame(survey_data)

def create_visualizations(survey_df):
    """Create comprehensive visualizations for survey data"""
    
    # 1. Dramatic Differences Distribution
    fig1 = px.histogram(
        survey_df, 
        x='dramatic_difference',
        nbins=15,
        title='Distribution of Dramatic Rank Differences',
        labels={'dramatic_difference': 'Rank Difference (positions)', 'count': 'Number of Pairs'},
        color_discrete_sequence=['#FF6B6B']
    )
    fig1.add_vline(
        x=survey_df['dramatic_difference'].mean(), 
        line_dash="dash", 
        annotation_text=f"Mean: {survey_df['dramatic_difference'].mean():.0f}"
    )
    
    # 2. Pair Types Distribution
    type_counts = survey_df['pair_type'].value_counts()
    fig2 = px.pie(
        values=type_counts.values,
        names=['Type 1: Similar H-index', 'Type 2: Similar Composite'],
        title='Pair Types Distribution',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    # 3. Interactive Scatter Plot
    fig3 = go.Figure()
    
    type1_mask = survey_df['pair_type'] == 'type1_similar_h_dramatic_composite'
    
    # Type 1 pairs
    fig3.add_trace(go.Scatter(
        x=survey_df[type1_mask]['researcher_a_h_rank'],
        y=survey_df[type1_mask]['researcher_a_composite_rank'],
        mode='markers',
        name='Type 1 - Better Composite',
        marker=dict(color='blue', size=12, symbol='circle'),
        text=survey_df[type1_mask]['researcher_a_name'],
        hovertemplate='<b>%{text}</b><br>H-index Rank: %{x}<br>Composite Rank: %{y}<extra></extra>'
    ))
    
    fig3.add_trace(go.Scatter(
        x=survey_df[type1_mask]['researcher_b_h_rank'],
        y=survey_df[type1_mask]['researcher_b_composite_rank'],
        mode='markers',
        name='Type 1 - Worse Composite',
        marker=dict(color='lightblue', size=12, symbol='square'),
        text=survey_df[type1_mask]['researcher_b_name'],
        hovertemplate='<b>%{text}</b><br>H-index Rank: %{x}<br>Composite Rank: %{y}<extra></extra>'
    ))
    
    # Type 2 pairs
    type2_mask = ~type1_mask
    fig3.add_trace(go.Scatter(
        x=survey_df[type2_mask]['researcher_a_h_rank'],
        y=survey_df[type2_mask]['researcher_a_composite_rank'],
        mode='markers',
        name='Type 2 - Better H-index',
        marker=dict(color='red', size=12, symbol='circle'),
        text=survey_df[type2_mask]['researcher_a_name'],
        hovertemplate='<b>%{text}</b><br>H-index Rank: %{x}<br>Composite Rank: %{y}<extra></extra>'
    ))
    
    fig3.add_trace(go.Scatter(
        x=survey_df[type2_mask]['researcher_b_h_rank'],
        y=survey_df[type2_mask]['researcher_b_composite_rank'],
        mode='markers',
        name='Type 2 - Worse H-index',
        marker=dict(color='lightcoral', size=12, symbol='square'),
        text=survey_df[type2_mask]['researcher_b_name'],
        hovertemplate='<b>%{text}</b><br>H-index Rank: %{x}<br>Composite Rank: %{y}<extra></extra>'
    ))
    
    # Add diagonal line
    max_rank = max(survey_df[['researcher_a_h_rank', 'researcher_b_h_rank', 
                            'researcher_a_composite_rank', 'researcher_b_composite_rank']].max())
    fig3.add_trace(go.Scatter(
        x=[0, max_rank],
        y=[0, max_rank],
        mode='lines',
        name='Perfect Agreement',
        line=dict(color='black', dash='dash'),
        hoverinfo='skip'
    ))
    
    fig3.update_layout(
        title='H-index Rank vs Composite Rank',
        xaxis_title='H-index Rank',
        yaxis_title='Composite Rank',
        hovermode='closest'
    )
    
    # 4. Top Pairs Bar Chart
    survey_df_sorted = survey_df.sort_values('dramatic_difference', ascending=True)
    
    fig4 = go.Figure(go.Bar(
        x=survey_df_sorted['dramatic_difference'],
        y=[f"Pair {row['pair_id']}: {row['researcher_a_name'][:20]}... vs {row['researcher_b_name'][:20]}..." 
           for _, row in survey_df_sorted.iterrows()],
        orientation='h',
        marker_color=['#FF6B6B' if row['pair_type'] == 'type1_similar_h_dramatic_composite' else '#4ECDC4' 
                      for _, row in survey_df_sorted.iterrows()],
        text=survey_df_sorted['dramatic_difference'],
        textposition='inside',
        hovertemplate='<b>Pair %{customdata[0]}</b><br>' +
                      'Researcher A: %{customdata[1]}<br>' +
                      'Researcher B: %{customdata[2]}<br>' +
                      'Dramatic Difference: %{x} positions<br>' +
                      'H-index Diff: %{customdata[3]}<br>' +
                      'Composite Diff: %{customdata[4]}<extra></extra>',
        customdata=[[row['pair_id'], row['researcher_a_name'], row['researcher_b_name'], 
                    row['h_rank_difference'], row['composite_rank_difference']] 
                   for _, row in survey_df_sorted.iterrows()]
    ))
    
    fig4.update_layout(
        title='Pairs by Dramatic Rank Difference',
        xaxis_title='Dramatic Rank Difference (positions)',
        yaxis_title='Pair Details',
        height=600
    )
    
    return fig1, fig2, fig3, fig4

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Beyond-H-Index Survey Candidate Selection Tool</h1>
        <p>Interactive tool for selecting researcher pairs with dramatic ranking disagreements</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Selection Criteria")
    st.sidebar.markdown("Adjust the parameters below to modify pair selection:")
    
    # Selection criteria inputs
    rank_tolerance = st.sidebar.slider(
        "Rank Tolerance",
        min_value=1, max_value=50, value=10, step=1,
        help="Max difference for 'similar' ranks"
    )
    
    min_dramatic_diff = st.sidebar.slider(
        "Min Dramatic Difference",
        min_value=25, max_value=500, value=100, step=25,
        help="Minimum difference for dramatic disagreements"
    )
    
    max_total_pairs = st.sidebar.slider(
        "Max Total Pairs",
        min_value=5, max_value=50, value=20, step=5,
        help="Maximum total pairs to select"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quality Filters")
    
    min_h_index = st.sidebar.slider(
        "Min H-index",
        min_value=0, max_value=50, value=0, step=5,
        help="Minimum H-index for inclusion"
    )
    
    min_papers = st.sidebar.slider(
        "Min Papers",
        min_value=0, max_value=100, value=0, step=10,
        help="Minimum number of papers"
    )
    
    min_career_span = st.sidebar.slider(
        "Min Career Span",
        min_value=0, max_value=20, value=0, step=1,
        help="Minimum years of publication history"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Target Pairs")
    
    target_type1_pairs = st.sidebar.slider(
        "Target Type 1 Pairs",
        min_value=1, max_value=25, value=10, step=1,
        help="Similar H-index, different composite"
    )
    
    target_type2_pairs = st.sidebar.slider(
        "Target Type 2 Pairs",
        min_value=1, max_value=25, value=10, step=1,
        help="Similar composite, different H-index"
    )
    
    # Process button
    if st.sidebar.button("🔄 Process Selection", type="primary"):
        
        # Create criteria dictionary
        criteria = {
            'rank_tolerance': rank_tolerance,
            'min_dramatic_diff': min_dramatic_diff,
            'max_total_pairs': max_total_pairs,
            'min_h_index': min_h_index,
            'min_papers': min_papers,
            'min_career_span': min_career_span,
            'target_type1_pairs': target_type1_pairs,
            'target_type2_pairs': target_type2_pairs,
        }
        
        # Store criteria in session state
        st.session_state.criteria = criteria
        
        with st.spinner("🔄 Processing selection with new criteria..."):
            
            # Filter data
            df_filtered = filter_data(df, criteria)
            
            # Select pairs
            selected_pairs, type1_count, type2_count = select_dramatic_pairs(
                df_filtered,
                rank_tolerance=criteria['rank_tolerance'],
                min_dramatic_diff=criteria['min_dramatic_diff'],
                target_type1=criteria['target_type1_pairs'],
                target_type2=criteria['target_type2_pairs']
            )
            
            # Create survey DataFrame
            if selected_pairs:
                survey_df = create_survey_dataframe(selected_pairs)
                
                # Store results in session state
                st.session_state.survey_df = survey_df
                st.session_state.selected_pairs = selected_pairs
                st.session_state.df_filtered = df_filtered
                st.session_state.type1_count = type1_count
                st.session_state.type2_count = type2_count
                
                # Save CSV
                survey_df.to_csv('survey_candidate_pairs_final.csv', index=False)
                
                st.success(f"✅ Selection completed! {len(selected_pairs)} pairs selected.")
            else:
                st.error("❌ No pairs found with current criteria. Try adjusting parameters.")
    
    # Display results if available
    if hasattr(st.session_state, 'survey_df') and not st.session_state.survey_df.empty:
        
        survey_df = st.session_state.survey_df
        type1_count = st.session_state.type1_count
        type2_count = st.session_state.type2_count
        
        # Display summary metrics
        st.subheader("📊 Selection Results")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(survey_df)}</div>
                <div class="metric-label">Total Pairs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{type1_count}</div>
                <div class="metric-label">Type 1 Pairs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{type2_count}</div>
                <div class="metric-label">Type 2 Pairs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{survey_df['dramatic_difference'].mean():.0f}</div>
                <div class="metric-label">Avg Dramatic Diff</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{survey_df['dramatic_difference'].max():.0f}</div>
                <div class="metric-label">Max Dramatic Diff</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            unique_authors = len(set(list(survey_df['researcher_a_name']) + list(survey_df['researcher_b_name'])))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{unique_authors}</div>
                <div class="metric-label">Unique Authors</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create visualizations
        st.subheader("📈 Interactive Visualizations")
        
        fig1, fig2, fig3, fig4 = create_visualizations(survey_df)
        
        # Display charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🥧 Pair Types", "🎯 Rank Comparison", "🏆 Top Pairs"])
        
        with tab1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            st.plotly_chart(fig4, use_container_width=True)
        
        # Display detailed pair information
        st.subheader("📋 Selected Pairs Details")
        
        # Filter options
        pair_type_filter = st.selectbox(
            "Filter by Pair Type:",
            ["All", "Type 1 (Similar H-index)", "Type 2 (Similar Composite)"]
        )
        
        if pair_type_filter == "Type 1 (Similar H-index)":
            filtered_df = survey_df[survey_df['pair_type'] == 'type1_similar_h_dramatic_composite']
        elif pair_type_filter == "Type 2 (Similar Composite)":
            filtered_df = survey_df[survey_df['pair_type'] == 'type2_similar_composite_dramatic_h']
        else:
            filtered_df = survey_df
        
        # Display the dataframe
        st.dataframe(
            filtered_df[['pair_id', 'pair_type', 'researcher_a_name', 'researcher_b_name', 
                        'h_rank_difference', 'composite_rank_difference', 'dramatic_difference']],
            use_container_width=True
        )
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = survey_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"survey_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create JSON export
            survey_questions = []
            for _, row in survey_df.iterrows():
                if row['pair_type'] == 'type1_similar_h_dramatic_composite':
                    question = {
                        'question_id': f"Q{int(row['pair_id'])}",
                        'question_type': 'Type1_SimilarH_DramaticComposite',
                        'question_text': row['question_context'],
                        'researcher_a': {
                            'name': row['researcher_a_name'],
                            'h_index': int(row['researcher_a_h_index']),
                            'h_rank': f"#{int(row['researcher_a_h_rank'])}",
                            'composite_rank': f"#{int(row['researcher_a_composite_rank'])}"
                        },
                        'researcher_b': {
                            'name': row['researcher_b_name'], 
                            'h_index': int(row['researcher_b_h_index']),
                            'h_rank': f"#{int(row['researcher_b_h_rank'])}",
                            'composite_rank': f"#{int(row['researcher_b_composite_rank'])}"
                        },
                        'answer_options': ['Researcher A', 'Researcher B', 'Unable to determine']
                    }
                else:
                    question = {
                        'question_id': f"Q{int(row['pair_id'])}",
                        'question_type': 'Type2_SimilarComposite_DramaticH',
                        'question_text': row['question_context'],
                        'researcher_a': {
                            'name': row['researcher_a_name'],
                            'composite_rank': f"#{int(row['researcher_a_composite_rank'])}",
                            'h_index': int(row['researcher_a_h_index']),
                            'h_rank': f"#{int(row['researcher_a_h_rank'])}"
                        },
                        'researcher_b': {
                            'name': row['researcher_b_name'], 
                            'composite_rank': f"#{int(row['researcher_b_composite_rank'])}",
                            'h_index': int(row['researcher_b_h_index']),
                            'h_rank': f"#{int(row['researcher_b_h_rank'])}"
                        },
                        'answer_options': ['Researcher A', 'Researcher B', 'Unable to determine']
                    }
                survey_questions.append(question)
            
            json_data = json.dumps(survey_questions, indent=2)
            st.download_button(
                label="📋 Download Survey JSON",
                data=json_data,
                file_name=f"survey_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Adjust the criteria in the sidebar and click 'Process Selection' to generate survey pairs.")
        
        # Show dataset overview
        st.subheader("📊 Dataset Overview")
        st.write(f"**Total researchers:** {len(df):,}")
        st.write(f"**H-index range:** {df['h_index'].min():.0f} - {df['h_index'].max():.0f}")
        
        if 'total_citations' in df.columns:
            st.write(f"**Citations range:** {df['total_citations'].min():,} - {df['total_citations'].max():,}")
        
        # Sample data
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(), use_container_width=True)

if __name__ == "__main__":
    main()
