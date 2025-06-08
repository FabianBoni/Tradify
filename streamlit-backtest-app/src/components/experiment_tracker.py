"""
Experiment Tracking und Dokumentationssystem
===========================================

Dieses Modul stellt umfassende Experiment-Tracking Funktionalitäten zur Verfügung:
- Automatische Logging von Experimenten
- Performance Monitoring
- Model Versioning
- Experiment Vergleiche
- Automatische Dokumentation

Author: Trading Strategy Team
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import hashlib

class ExperimentTracker:
    """
    Zentrale Klasse für Experiment Tracking und Dokumentation
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialisiert die SQLite-Datenbank für Experiment Tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                target_type TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                cv_score REAL,
                test_score REAL,
                feature_count INTEGER,
                training_time REAL,
                parameters TEXT,
                feature_importance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Model Performance Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT, -- train, validation, test
                epoch INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # Feature Importance Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                feature_name TEXT NOT NULL,
                importance_value REAL NOT NULL,
                rank_position INTEGER,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_experiment(self, experiment_data: Dict[str, Any]) -> int:
        """
        Loggt ein neues Experiment
        
        Args:
            experiment_data: Dictionary mit Experiment-Daten
            
        Returns:
            ID des erstellten Experiments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO experiments 
                (experiment_name, symbol, target_type, horizon, model_type, 
                 cv_score, test_score, feature_count, training_time, parameters, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_data['experiment_name'],
                experiment_data['symbol'],
                experiment_data['target_type'],
                experiment_data['horizon'],
                experiment_data['model_type'],
                experiment_data.get('cv_score'),
                experiment_data.get('test_score'),
                experiment_data.get('feature_count'),
                experiment_data.get('training_time'),
                json.dumps(experiment_data.get('parameters', {})),
                experiment_data.get('notes', '')
            ))
            
            experiment_id = cursor.lastrowid
            
            # Log feature importance if available
            if 'feature_importance' in experiment_data:
                for rank, (feature, importance) in enumerate(experiment_data['feature_importance'].items()):
                    cursor.execute('''
                        INSERT INTO feature_importance 
                        (experiment_id, feature_name, importance_value, rank_position)
                        VALUES (?, ?, ?, ?)
                    ''', (experiment_id, feature, importance, rank + 1))
            
            conn.commit()
            return experiment_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_experiments(self, limit: int = 100) -> pd.DataFrame:
        """Lädt alle Experimente aus der Datenbank"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM experiments 
            ORDER BY created_at DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_experiment_details(self, experiment_id: int) -> Dict[str, Any]:
        """Lädt Details eines spezifischen Experiments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiment details
        cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
        experiment = cursor.fetchone()
        
        if not experiment:
            return None
        
        # Feature importance
        cursor.execute('''
            SELECT feature_name, importance_value, rank_position 
            FROM feature_importance 
            WHERE experiment_id = ? 
            ORDER BY rank_position
        ''', (experiment_id,))
        
        features = cursor.fetchall()
        
        conn.close()
        
        return {
            'experiment': experiment,
            'feature_importance': features
        }
    
    def render_tracking_dashboard(self):
        """Rendert das Experiment Tracking Dashboard"""
        
        st.header("📊 Experiment Tracking Dashboard")
        
        # Load experiments
        experiments_df = self.get_experiments()
        
        if experiments_df.empty:
            st.info("Noch keine Experimente vorhanden. Starte dein erstes ML-Experiment!")
            return
        
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🧪 Total Experimente", len(experiments_df))
        
        with col2:
            best_score = experiments_df['test_score'].max()
            st.metric("🏆 Beste Performance", f"{best_score:.4f}")
        
        with col3:
            avg_training_time = experiments_df['training_time'].mean()
            st.metric("⏱️ Ø Training Zeit", f"{avg_training_time:.1f}s")
        
        with col4:
            unique_models = experiments_df['model_type'].nunique()
            st.metric("🤖 Getestete Modelle", unique_models)
        
        # Performance Trend Chart
        st.subheader("📈 Performance Trend")
        
        fig = go.Figure()
        
        # Add performance trend
        fig.add_trace(go.Scatter(
            x=experiments_df['created_at'],
            y=experiments_df['test_score'],
            mode='lines+markers',
            name='Test Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        # Add best score line
        fig.add_hline(
            y=best_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Best: {best_score:.4f}"
        )
        
        fig.update_layout(
            title="Performance über Zeit",
            xaxis_title="Datum",
            yaxis_title="Test Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Modell-Vergleich")
            
            model_performance = experiments_df.groupby('model_type').agg({
                'test_score': ['mean', 'max', 'count'],
                'training_time': 'mean'
            }).round(4)
            
            model_performance.columns = ['Avg Score', 'Best Score', 'Experimente', 'Avg Zeit']
            st.dataframe(model_performance)
        
        with col2:
            st.subheader("🎯 Target Type Verteilung")
            
            target_dist = experiments_df['target_type'].value_counts()
            fig = px.pie(values=target_dist.values, names=target_dist.index,
                        title="Classification vs Regression")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Experiments Table
        st.subheader("📋 Experiment Details")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_filter = st.selectbox("Model Filter", 
                                      ['Alle'] + list(experiments_df['model_type'].unique()))
        
        with col2:
            target_filter = st.selectbox("Target Filter",
                                       ['Alle'] + list(experiments_df['target_type'].unique()))
        
        with col3:
            min_score = st.slider("Min Score", 
                                0.0, 1.0, 0.0, 0.01)
        
        # Apply filters
        filtered_df = experiments_df.copy()
        
        if model_filter != 'Alle':
            filtered_df = filtered_df[filtered_df['model_type'] == model_filter]
        
        if target_filter != 'Alle':
            filtered_df = filtered_df[filtered_df['target_type'] == target_filter]
        
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['test_score'] >= min_score]
        
        # Display table
        display_columns = ['experiment_name', 'symbol', 'model_type', 'target_type', 
                         'cv_score', 'test_score', 'feature_count', 'created_at']
        st.dataframe(filtered_df[display_columns], use_container_width=True)
        
        # Experiment Comparison
        if len(filtered_df) >= 2:
            st.subheader("⚖️ Experiment Vergleich")
            
            selected_experiments = st.multiselect(
                "Experimente auswählen",
                options=filtered_df['experiment_name'].tolist(),
                default=filtered_df.nlargest(2, 'test_score')['experiment_name'].tolist()
            )
            
            if len(selected_experiments) >= 2:
                self.render_experiment_comparison(selected_experiments)
    
    def render_experiment_comparison(self, experiment_names: List[str]):
        """Rendert einen detaillierten Vergleich von Experimenten"""
        
        comparison_data = []
        
        for exp_name in experiment_names:
            # Get experiment from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM experiments WHERE experiment_name = ?', (exp_name,))
            exp_data = cursor.fetchone()
            
            if exp_data:
                comparison_data.append({
                    'Name': exp_name,
                    'Model': exp_data[5],  # model_type
                    'CV Score': exp_data[6],  # cv_score
                    'Test Score': exp_data[7],  # test_score
                    'Features': exp_data[8],  # feature_count
                    'Training Time': f"{exp_data[9]:.1f}s" if exp_data[9] else "N/A"
                })
            
            conn.close()
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📊 **Performance Vergleich**")
                
                fig = go.Figure()
                
                for metric in ['CV Score', 'Test Score']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df['Name'],
                        y=comparison_df[metric],
                        text=comparison_df[metric],
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Score Vergleich",
                    barmode='group',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("⚡ **Effizienz Vergleich**")
                
                # Convert training time to float for plotting
                training_times = []
                for time_str in comparison_df['Training Time']:
                    if time_str != "N/A":
                        training_times.append(float(time_str.replace('s', '')))
                    else:
                        training_times.append(0)
                
                fig = px.scatter(
                    x=comparison_df['Features'],
                    y=training_times,
                    size=comparison_df['Test Score'],
                    color=comparison_df['Name'],
                    title="Features vs Training Zeit",
                    labels={'x': 'Anzahl Features', 'y': 'Training Zeit (s)'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.write("📋 **Detaillierter Vergleich**")
            st.dataframe(comparison_df, use_container_width=True)
    
    def export_experiment_report(self, experiment_id: int) -> str:
        """Exportiert einen detaillierten Experiment-Report"""
        
        details = self.get_experiment_details(experiment_id)
        
        if not details:
            return "Experiment nicht gefunden."
        
        exp = details['experiment']
        features = details['feature_importance']
        
        report = f"""
# Experiment Report: {exp[1]}

## Experiment Details
- **Symbol:** {exp[2]}
- **Target Type:** {exp[3]}
- **Horizon:** {exp[4]}
- **Model Type:** {exp[5]}
- **Created:** {exp[12]}

## Performance Metrics
- **Cross-Validation Score:** {exp[6]:.4f}
- **Test Score:** {exp[7]:.4f}
- **Feature Count:** {exp[8]}
- **Training Time:** {exp[9]:.2f} seconds

## Model Parameters
```json
{exp[10]}
```

## Feature Importance (Top 20)
| Rank | Feature | Importance |
|------|---------|------------|
"""
        
        for feature in features[:20]:
            report += f"| {feature[2]} | {feature[0]} | {feature[1]:.4f} |\n"
        
        report += f"""

## Notes
{exp[13] or 'Keine Notizen verfügbar.'}

---
*Report generiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        return report

def render_experiment_tracker():
    """Main function to render the experiment tracking interface"""
    tracker = ExperimentTracker()
    tracker.render_tracking_dashboard()
