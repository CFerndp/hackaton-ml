import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class MIDUSPreprocessor:
    """
    Specialized preprocessor for MIDUS stress, inflammation, and cognition data.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to data file (relative to project root)
            df (pd.DataFrame): DataFrame with data already loaded
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            # Convert to Path object and make it relative to project root
            self.data_path = Path(data_path)
            if not self.data_path.is_absolute():
                # If path is relative, make it relative to project root
                project_root = Path(__file__).parent.parent  # Go up from src/ to project root
                self.data_path = project_root / data_path
            
            self.df = pd.read_csv(self.data_path, sep='\t')
        else:
            raise ValueError("Must provide either data_path or df")
        
        self.original_df = self.df.copy()
        self.outlier_info = {}
        self.scaling_info = {}
        
        # Define variable groups
        self._define_variable_groups()
        
    def _define_variable_groups(self):
        """Define variable groups according to their nature."""
        
        # Identification variables
        self.id_vars = ['M2ID', 'M2FAMNUM', 'SAMPLMAJ']
        
        # Demographic variables
        self.demo_vars = ['B1PAGE_M', 'B1PGENDE']
        
        # Already log-transformed variables
        self.log_vars = ['M2_LOG_C', 'V7_A', 'M2_LOG_I', 'LOG_M2_N']
        
        # Biomarker variables in original scale
        self.biomarker_vars = ['M2_DOPAM', 'M2_EPINE', 'M2_FIBRI', 'M2_SICAM']
        
        # Cognitive variables (already as z-scores)
        self.cognitive_vars = ['M2_EPISO', 'M2_EXECU', 'M3_EPISO', 'M3_EXECU']
        
        # All numeric variables (excluding IDs)
        self.numeric_vars = (self.demo_vars + self.log_vars + 
                           self.biomarker_vars + self.cognitive_vars)
    
    def explore_data(self, save_plots=True):
        """Initial data exploration with visualizations."""
        
        print("=== MIDUS DATA EXPLORATION ===\n")
        
        # Basic information
        print(f"Dimensions: {self.df.shape}")
        print(f"Numeric variables: {len(self.numeric_vars)}")
        print(f"Missing values per variable:")
        missing = self.df[self.numeric_vars].isnull().sum()
        print(missing[missing > 0])
        print()
        
        # Descriptive statistics by group
        for group_name, vars_list in [
            ("Demographics", self.demo_vars),
            ("Log-transformed", self.log_vars),
            ("Original biomarkers", self.biomarker_vars),
            ("Cognitive (z-scores)", self.cognitive_vars)
        ]:
            print(f"\n=== {group_name.upper()} ===")
            available_vars = [var for var in vars_list if var in self.df.columns]
            if available_vars:
                desc = self.df[available_vars].describe()
                print(desc.round(3))
        
        # Create visualizations
        if save_plots:
            self._plot_data_exploration()
    
    def _plot_data_exploration(self):
        """Create comprehensive data exploration plots."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 1. Distribution plots by variable group
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MIDUS Data Distributions by Variable Type', fontsize=16, fontweight='bold')
        
        groups_to_plot = [
            ("Biomarkers (Original)", self.biomarker_vars),
            ("Log-transformed", self.log_vars),
            ("Cognitive (z-scores)", self.cognitive_vars),
            ("Demographics", self.demo_vars)
        ]
        
        for idx, (title, vars_list) in enumerate(groups_to_plot):
            ax = axes[idx // 2, idx % 2]
            available_vars = [var for var in vars_list if var in self.df.columns]
            
            if available_vars:
                # Create box plots for each variable group
                data_to_plot = []
                labels = []
                
                for var in available_vars[:4]:  # Limit to 4 variables per plot
                    data_to_plot.append(self.df[var].dropna())
                    labels.append(var.replace('M2_', '').replace('_', ' '))
                
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(title, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'midus_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        self._plot_correlation_matrix()
        
        # 3. Age distribution and cognitive performance
        self._plot_age_cognition_relationship()
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix of key variables."""
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Select key variables for correlation
        key_vars = []
        
        # Add available variables from each group
        for vars_list in [self.biomarker_vars, self.log_vars, self.cognitive_vars]:
            available_vars = [var for var in vars_list if var in self.df.columns]
            key_vars.extend(available_vars[:3])  # Limit to avoid overcrowding
        
        if len(key_vars) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[key_vars].corr()
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.2f')
            
            plt.title('MIDUS Variables Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(results_dir / 'midus_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_age_cognition_relationship(self):
        """Plot relationship between age and cognitive performance."""
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        if all(var in self.df.columns for var in ['B1PAGE_M', 'M2_EPISO', 'M2_EXECU']):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Episodic memory vs age
            axes[0].scatter(self.df['B1PAGE_M'], self.df['M2_EPISO'], 
                           alpha=0.6, color='steelblue', s=50)
            axes[0].set_xlabel('Age (years)')
            axes[0].set_ylabel('Episodic Memory (z-score)')
            axes[0].set_title('Age vs Episodic Memory')
            axes[0].grid(True, alpha=0.3)
            
            # Executive function vs age
            axes[1].scatter(self.df['B1PAGE_M'], self.df['M2_EXECU'], 
                           alpha=0.6, color='darkgreen', s=50)
            axes[1].set_xlabel('Age (years)')
            axes[1].set_ylabel('Executive Function (z-score)')
            axes[1].set_title('Age vs Executive Function')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(results_dir / 'midus_age_cognition.png', dpi=300, bbox_inches='tight')
            plt.show()

    def detect_outliers(self, method='multiple', contamination=0.05):
        """
        Detect outliers using multiple methods and create visualizations.
        
        Args:
            method (str): 'iqr', 'zscore', 'isolation', 'lof', 'multiple'
            contamination (float): Expected proportion of outliers
        """
        
        print("=== OUTLIER DETECTION ===\n")
        
        outlier_masks = {}
        
        # Apply different methods according to variable type
        for group_name, vars_list in [
            ("biomarker_vars", self.biomarker_vars),
            ("log_vars", self.log_vars),
            ("cognitive_vars", self.cognitive_vars)
        ]:
            
            available_vars = [var for var in vars_list if var in self.df.columns]
            if not available_vars:
                continue
                
            print(f"Analyzing {group_name}...")
            
            # Data for this group
            group_data = self.df[available_vars].dropna()
            
            if len(group_data) == 0:
                continue
            
            if method in ['multiple', 'iqr']:
                # IQR method
                outliers_iqr = self._detect_outliers_iqr(group_data)
                outlier_masks[f'{group_name}_iqr'] = outliers_iqr
                print(f"  IQR: {outliers_iqr.sum()} outliers detected")
            
            if method in ['multiple', 'isolation']:
                # Isolation Forest
                if len(group_data) > 10:  # Needs sufficient data
                    outliers_iso = self._detect_outliers_isolation(group_data, contamination)
                    outlier_masks[f'{group_name}_isolation'] = outliers_iso
                    print(f"  Isolation Forest: {outliers_iso.sum()} outliers detected")
        
        # Method consensus
        if method == 'multiple' and outlier_masks:
            consensus_outliers = self._get_consensus_outliers(outlier_masks)
            print(f"\nConsensus outliers (≥2 methods): {consensus_outliers.sum()}")
            self.outlier_info['consensus'] = consensus_outliers
        
        self.outlier_info['masks'] = outlier_masks
        
        # Create outlier visualizations
        self._plot_outliers()
    
    def _plot_outliers(self):
        """Create comprehensive outlier visualizations."""
        
        if not self.outlier_info or 'consensus' not in self.outlier_info:
            print("No outlier information available for plotting.")
            return
        
        # 1. Outlier overview plot
        self._plot_outlier_overview()
        
        # 2. Biomarker outliers
        self._plot_biomarker_outliers()
        
        # 3. Cognitive outliers
        self._plot_cognitive_outliers()
    
    def _plot_outlier_overview(self):
        """Plot overview of outliers across all methods."""
        
        if 'masks' not in self.outlier_info:
            return
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Count outliers by method
        method_counts = {}
        for method_name, mask in self.outlier_info['masks'].items():
            method_counts[method_name] = mask.sum()
        
        if 'consensus' in self.outlier_info:
            method_counts['Consensus'] = self.outlier_info['consensus'].sum()
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        
        bars = plt.bar(methods, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'])
        
        plt.title('Outliers Detected by Different Methods', fontsize=14, fontweight='bold')
        plt.xlabel('Detection Method')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(results_dir / 'midus_outlier_methods.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_biomarker_outliers(self):
        """Plot biomarker distributions highlighting outliers."""
        
        available_biomarkers = [var for var in self.biomarker_vars if var in self.df.columns]
        
        if not available_biomarkers or 'consensus' not in self.outlier_info:
            return
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        n_vars = len(available_biomarkers)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        fig.suptitle('Biomarker Distributions with Outliers Highlighted', fontsize=16, fontweight='bold')
        
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        outlier_mask = self.outlier_info['consensus']
        
        for idx, var in enumerate(available_biomarkers):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                break
                
            # Normal points
            normal_data = self.df.loc[~outlier_mask, var].dropna()
            outlier_data = self.df.loc[outlier_mask, var].dropna()
            
            # Create box plot
            ax.boxplot([normal_data], positions=[1], widths=0.6, 
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
            
            # Highlight outliers
            if len(outlier_data) > 0:
                ax.scatter([1] * len(outlier_data), outlier_data, 
                          color='red', s=50, alpha=0.7, label=f'Outliers (n={len(outlier_data)})')
                ax.legend()
            
            ax.set_title(f'{var.replace("M2_", "").replace("_", " ")}')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(available_biomarkers), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'midus_biomarker_outliers.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_cognitive_outliers(self):
        """Plot cognitive performance highlighting outliers."""
        
        if not all(var in self.df.columns for var in ['M2_EPISO', 'M2_EXECU']) or 'consensus' not in self.outlier_info:
            return
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        outlier_mask = self.outlier_info['consensus']
        
        plt.figure(figsize=(10, 8))
        
        # Normal participants
        normal_mask = ~outlier_mask
        plt.scatter(self.df.loc[normal_mask, 'M2_EPISO'], 
                   self.df.loc[normal_mask, 'M2_EXECU'],
                   alpha=0.6, color='steelblue', s=50, label='Normal')
        
        # Outliers
        if outlier_mask.sum() > 0:
            plt.scatter(self.df.loc[outlier_mask, 'M2_EPISO'], 
                       self.df.loc[outlier_mask, 'M2_EXECU'],
                       alpha=0.8, color='red', s=80, 
                       label=f'Outliers (n={outlier_mask.sum()})', 
                       edgecolors='black', linewidth=1)
        
        # Add reference lines at z-score thresholds
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='High performance')
        plt.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.7, label='Low performance')
        plt.axvline(x=0.5, color='green', linestyle=':', alpha=0.7)
        plt.axvline(x=-0.5, color='orange', linestyle=':', alpha=0.7)
        
        plt.xlabel('Episodic Memory (z-score)')
        plt.ylabel('Executive Function (z-score)')
        plt.title('Cognitive Performance: Episodic Memory vs Executive Function', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'midus_cognitive_outliers.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _detect_outliers_iqr(self, data, factor=1.5):
        """Detect outliers using IQR method."""
        outliers = pd.Series(False, index=data.index)
        
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers = outliers | col_outliers
        
        return outliers
    
    def _detect_outliers_isolation(self, data, contamination=0.05):
        """Detect outliers using Isolation Forest."""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data)
        return pd.Series(outlier_labels == -1, index=data.index)
    
    def _get_consensus_outliers(self, outlier_masks, min_methods=2):
        """Get outliers detected by at least min_methods methods."""
        if not outlier_masks:
            return pd.Series(False, index=self.df.index)
        
        # Create DataFrame with all masks, aligning indices
        all_masks = pd.DataFrame(index=self.df.index)
        
        for name, mask in outlier_masks.items():
            all_masks[name] = False
            all_masks.loc[mask.index[mask], name] = True
        
        consensus = all_masks.sum(axis=1) >= min_methods
        return consensus
    
    def handle_outliers(self, strategy='cap', percentile=99):
        """Handle detected outliers."""
        
        print(f"=== OUTLIER HANDLING: {strategy.upper()} ===\n")
        
        if strategy == 'cap':
            for var in self.biomarker_vars:  # Only for original biomarkers
                if var in self.df.columns:
                    lower_cap = self.df[var].quantile((100-percentile)/100)
                    upper_cap = self.df[var].quantile(percentile/100)
                    
                    original_outliers = ((self.df[var] < lower_cap) | 
                                       (self.df[var] > upper_cap)).sum()
                    
                    self.df[var] = np.clip(self.df[var], lower_cap, upper_cap)
                    
                    print(f"{var}: {original_outliers} values capped "
                          f"between {lower_cap:.2f} and {upper_cap:.2f}")
        
        elif strategy == 'keep':
            print("Outliers kept in dataset")
    
    def scale_features(self, method='robust', exclude_groups=None):
        """Scale numeric features."""
        
        if exclude_groups is None:
            exclude_groups = ['cognitive_vars']  # Already z-scores
        
        print(f"=== FEATURE SCALING: {method.upper()} ===\n")
        
        # Variables to scale
        vars_to_scale = []
        
        if 'demo_vars' not in exclude_groups:
            vars_to_scale.extend(self.demo_vars)
        if 'log_vars' not in exclude_groups:
            vars_to_scale.extend(self.log_vars)
        if 'biomarker_vars' not in exclude_groups:
            vars_to_scale.extend(self.biomarker_vars)
        
        # Filter existing variables
        vars_to_scale = [var for var in vars_to_scale if var in self.df.columns]
        
        if method == 'none' or not vars_to_scale:
            print("No scaling applied")
            return
        
        # Select scaler
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Apply scaling
        scaled_data = scaler.fit_transform(self.df[vars_to_scale])
        
        # Create new scaled columns
        for i, var in enumerate(vars_to_scale):
            self.df[f'{var}_scaled'] = scaled_data[:, i]
        
        # Store scaling info
        self.scaling_info = {
            'method': method,
            'scaler': scaler,
            'variables': vars_to_scale
        }
        
        print(f"Scaled {len(vars_to_scale)} variables using {method}")
    
    def create_derived_features(self):
        """Create relevant derived features."""
        
        print("=== CREATING DERIVED FEATURES ===\n")
        
        # 1. Longitudinal cognitive change
        if all(var in self.df.columns for var in ['M3_EPISO', 'M2_EPISO']):
            self.df['episodic_change'] = self.df['M3_EPISO'] - self.df['M2_EPISO']
            print("✓ Episodic memory change calculated")
        
        if all(var in self.df.columns for var in ['M3_EXECU', 'M2_EXECU']):
            self.df['executive_change'] = self.df['M3_EXECU'] - self.df['M2_EXECU']
            print("✓ Executive function change calculated")
        
        # 2. Age categories
        if 'B1PAGE_M' in self.df.columns:
            self.df['age_group'] = pd.cut(self.df['B1PAGE_M'], 
                         bins=[0, 45, 55, 65, 100], 
                         labels=[0, 1, 2, 3])
            print("✓ Age groups created")
        
        # 3. Cognitive profile
        if all(var in self.df.columns for var in ['M2_EPISO', 'M2_EXECU']):
            self.df['cognitive_profile'] = 0
            
            # High performance
            high_cog = (self.df['M2_EPISO'] > 0.5) & (self.df['M2_EXECU'] > 0.5)
            self.df.loc[high_cog, 'cognitive_profile'] = 1
            
            # Low performance
            low_cog = (self.df['M2_EPISO'] < -0.5) & (self.df['M2_EXECU'] < -0.5)
            self.df.loc[low_cog, 'cognitive_profile'] = 2
            
            # Dissociated
            dissoc = ((self.df['M2_EPISO'] > 0.5) & (self.df['M2_EXECU'] < -0.5)) | \
                    ((self.df['M2_EPISO'] < -0.5) & (self.df['M2_EXECU'] > 0.5))
            self.df.loc[dissoc, 'cognitive_profile'] = 3
            
            print("✓ Cognitive profiles created")
            print(self.df['cognitive_profile'].value_counts())
        
        # 4. Inflammation index (composite)
        inflammation_vars = ['M2_LOG_I', 'M2_FIBRI', 'M2_SICAM']
        available_inflam = [var for var in inflammation_vars if var in self.df.columns]
        
        if len(available_inflam) >= 2:
            # Normalize before creating index
            inflam_data = self.df[available_inflam].copy()
            for var in available_inflam:
                inflam_data[var] = (inflam_data[var] - inflam_data[var].mean()) / inflam_data[var].std()
            
            inflam_mean = inflam_data.mean(axis=1)
            self.df['is_inflammated'] = ((inflam_mean > 1) | (inflam_mean < -1)).astype(int)
            print("✓ Inflammation index created")
        
        # 5. Stress index (composite)
        stress_vars = ['M2_LOG_C', 'LOG_M2_N', 'V7_A']
        available_stress = [var for var in stress_vars if var in self.df.columns]
        
        if len(available_stress) >= 2:
            stress_data = self.df[available_stress].copy()
            for var in available_stress:
                stress_data[var] = (stress_data[var] - stress_data[var].mean()) / stress_data[var].std()

            stress_mean = stress_data.mean(axis=1)
            self.df['is_stressed'] = ((stress_mean > 1) | (stress_mean < -1)).astype(int)
            print("✓ Stress index created")
        
        # Plot derived features
        self._plot_derived_features()
    
    def _plot_derived_features(self):
        """Plot the newly created derived features."""
        
        derived_vars = ['episodic_change', 'executive_change', 'inflammation_index', 'stress_index']
        available_derived = [var for var in derived_vars if var in self.df.columns]
        
        if not available_derived:
            return
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        n_vars = len(available_derived)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        fig.suptitle('Derived Feature Distributions', fontsize=16, fontweight='bold')
        
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, var in enumerate(available_derived):
            if idx < len(axes):
                ax = axes[idx]
                data = self.df[var].dropna()
                
                # Histogram
                ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', 
                          label=f'Mean: {data.mean():.3f}')
                ax.axvline(data.median(), color='green', linestyle='--', 
                          label=f'Median: {data.median():.3f}')
                
                ax.set_title(var.replace('_', ' ').title())
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(available_derived), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'midus_derived_features.png', dpi=300, bbox_inches='tight')
        plt.show()

    def get_processed_data(self, include_original=False):
        """Get processed data."""
        return self.df.copy()
    
    def generate_report(self):
        """Generate preprocessing report."""
        
        print("\n" + "="*60)
        print("           MIDUS PREPROCESSING REPORT")
        print("="*60)
        
        print(f"\n📊 ORIGINAL DATA:")
        print(f"   • Dimensions: {self.original_df.shape}")
        
        print(f"\n📊 PROCESSED DATA:")                     
        print(f"   • Dimensions: {self.df.shape}")
        print(f"   • New features: {self.df.shape[1] - self.original_df.shape[1]}")
        
        if self.outlier_info:
            print(f"\n🔍 OUTLIERS:")
            if 'consensus' in self.outlier_info:
                print(f"   • Consensus detected: {self.outlier_info['consensus'].sum()}")
            print(f"   • Methods applied: {len(self.outlier_info.get('masks', {}))}")
        
        if self.scaling_info:
            print(f"\n📏 SCALING:")
            print(f"   • Method: {self.scaling_info['method']}")
            print(f"   • Scaled variables: {len(self.scaling_info['variables'])}")
        
        # Derived features
        derived_features = ['inflammation_index', 'stress_index', 'episodic_change', 
                          'executive_change', 'age_group', 'cognitive_profile']
        available_derived = [f for f in derived_features if f in self.df.columns]
        
        if available_derived:
            print(f"\n🆕 DERIVED FEATURES:")
            for feature in available_derived:
                print(f"   • {feature}")
        
        print(f"\n📊 VISUALIZATIONS CREATED:")
        print(f"   • Data exploration plots (results/)")
        print(f"   • Correlation matrix (results/)")
        print(f"   • Outlier analysis plots (results/)")
        print(f"   • Derived feature distributions (results/)")
        
        print(f"\n✅ Data ready for ML analysis")
        print("="*60)


def preprocess_midus_data(data_path, outlier_strategy='cap', scaling_method='robust'):
    """
    Convenience function to preprocess MIDUS data.
    
    Args:
        data_path (str): Path to data (relative to project root)
        outlier_strategy (str): Strategy for outliers
        scaling_method (str): Scaling method
        
    Returns:
        tuple: (preprocessor, processed_data)
    """
    
    # Create preprocessor
    preprocessor = MIDUSPreprocessor(data_path=data_path)
    
    # Preprocessing pipeline
    preprocessor.explore_data(save_plots=True)
    preprocessor.detect_outliers(method='multiple')
    preprocessor.handle_outliers(strategy=outlier_strategy)
    preprocessor.scale_features(method=scaling_method)
    preprocessor.create_derived_features()
    
    # Generate report
    preprocessor.generate_report()
    
    # Get processed data
    processed_data = preprocessor.get_processed_data()
    
    return preprocessor, processed_data


if __name__ == "__main__":
    # Example usage with relative path
    data_path = "data/25. 0504 Harvard dataverse. Stree, Inflammation, Cognition. sav.tab"
    
    print("Starting MIDUS data preprocessing...")
    preprocessor, processed_data = preprocess_midus_data(
        data_path=data_path,
        outlier_strategy='cap',
        scaling_method='robust'
    )
    
    print(f"\nProcessed data saved. Final shape: {processed_data.shape}")
    
    # Save processed data with relative path
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "midus_processed_data.csv"
    processed_data.to_csv(output_path, index=False)
    print(f"Data saved to '{output_path}'")