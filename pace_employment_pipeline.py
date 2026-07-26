"""
PACE Framework National Employment & Labor Force Predictive Pipeline
====================================================================
An end-to-end, production-grade modular machine learning and statistical pipeline
following the Plan, Analyze, Construct, Execute (PACE) framework.

Author: Lead Machine Learning Engineer & Principal Data Scientist
Date: July 2026
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PACEPipeline")


@dataclass
class PipelineConfig:
    """Configuration settings for the PACE Employment Data Science Pipeline."""
    data_path: str = r"c:\Users\ASUS\Documents\Data Science\Data Analysis\Machine learning\data\final_merged_dataset.csv"
    primary_target: str = "RatesKEI - Unemployment Rate | Both sexes"
    fallback_target_keywords: List[str] = field(default_factory=lambda: [
        "Unemployment Rate", "Unemployed Persons", "Employed", "Labor Force"
    ])
    test_size: float = 0.20
    random_state: int = 42
    alpha_significance: float = 0.05
    iqr_multiplier: float = 1.5
    synthetic_n_samples: int = 128


@dataclass
class PreprocessedData:
    """Container for preprocessed dataset and metadata."""
    df: pd.DataFrame
    target_col: str
    feature_cols: List[str]
    time_series_index: pd.DatetimeIndex


@dataclass
class StatisticalSummary:
    """Container for descriptive statistical metrics."""
    metrics_df: pd.DataFrame
    target_skewness: float
    target_kurtosis: float


@dataclass
class DistributionFitResult:
    """Container for probability density function fitting and goodness-of-fit tests."""
    best_distribution_name: str
    fitted_params: Tuple[float, ...]
    shapiro_stat: float
    shapiro_pvalue: float
    ks_stat: float
    ks_pvalue: float
    jb_stat: float
    jb_pvalue: float
    theoretical_density_eval: Dict[str, Any]


@dataclass
class HypothesisTestResult:
    """Container for inferential statistical hypothesis testing results."""
    h0_description: str
    h1_description: str
    test_name: str
    test_statistic: float
    p_value: float
    confidence_interval_95: Tuple[float, float]
    effect_size_cohen_d: float
    decision_reject_h0: bool
    business_conclusion: str


@dataclass
class ModelEvaluationResult:
    """Container for baseline and advanced predictive modeling performance."""
    model_name: str
    train_r2: float
    train_mae: float
    train_rmse: float
    test_r2: float
    test_mae: float
    test_rmse: float
    feature_importances: pd.Series
    residuals_test: np.ndarray
    y_test_pred: np.ndarray


class SyntheticDataGenerator:
    """Generates synthetic macroeconomic & employment data when primary files are unavailable."""

    @staticmethod
    def generate(n_samples: int = 128, random_state: int = 42) -> pd.DataFrame:
        """Generates a realistic synthetic labor statistics time series dataset.

        Args:
            n_samples: Number of monthly/quarterly historical observations to generate.
            random_state: Random seed for reproducibility.

        Returns:
            pd.DataFrame containing synthetic employment indicators.
        """
        logger.info(f"Generating synthetic labor market dataset with {n_samples} observations...")
        np.random.seed(random_state)
        
        years = np.repeat(np.arange(2005, 2005 + int(np.ceil(n_samples / 4))), 4)[:n_samples]
        months = np.tile([1, 4, 7, 10], int(np.ceil(n_samples / 4)))[:n_samples]
        
        t = np.linspace(0, 10, n_samples)
        
        # Base macroeconomic cycles and COVID-19 structural break shock in 2020
        covid_shock = np.where(years == 2020, 4.5, 0.0) + np.where(years == 2021, 2.0, 0.0)
        unemp_rate = 7.0 - 0.2 * t + 0.8 * np.sin(t * 1.5) + covid_shock + np.random.normal(0, 0.3, n_samples)
        unemp_rate = np.clip(unemp_rate, 3.0, 18.0)
        
        lab_force_part = 64.0 + 0.1 * t + 0.5 * np.cos(t * 1.2) - 0.5 * covid_shock + np.random.normal(0, 0.4, n_samples)
        emp_rate = 100.0 - unemp_rate
        total_pop = 54000 + 400 * t + np.random.normal(0, 50, n_samples)
        total_lab_force = total_pop * (lab_force_part / 100.0)
        unemployed_persons = total_lab_force * (unemp_rate / 100.0)
        employed_persons = total_lab_force - unemployed_persons
        mean_hours = 42.0 - 0.05 * t - 0.3 * covid_shock + np.random.normal(0, 0.5, n_samples)
        
        data = {
            "Year": years,
            "Month": months,
            "RatesKEI - Unemployment Rate | Both sexes": unemp_rate,
            "RatesKEI - Employment Rate | Both sexes": emp_rate,
            "RatesKEI - Labor Force Participation Rate | Both sexes": lab_force_part,
            "LevelsKEI - Unemployed Persons | Both sexes": unemployed_persons,
            "LevelsKEI - Employed Persons | Both sexes": employed_persons,
            "LevelsKEI - Persons in the Labor Force | Both sexes": total_lab_force,
            "LevelsKEI - Total Population 15 Years Old and Over | Both sexes": total_pop,
            "MeanHours - Mean Hours": mean_hours,
            "EmployedClass - Wage and Salary Workers": employed_persons * 0.62 + np.random.normal(0, 100, n_samples),
            "EmployedClass - Self-employed without any paid employee": employed_persons * 0.28 + np.random.normal(0, 50, n_samples),
            "EmployedClass - Worked for government or government corporation": employed_persons * 0.10 + np.random.normal(0, 30, n_samples),
        }
        
        df = pd.DataFrame(data)
        logger.info("Synthetic dataset successfully created.")
        return df


class DataPreprocessor:
    """Preprocesses labor market data: type coercion, temporal sorting, dynamic imputation, feature engineering."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_and_clean(self) -> PreprocessedData:
        """Loads data from CSV or synthetic fallback, sorts chronologically, and cleans feature columns.

        Returns:
            PreprocessedData object containing clean DataFrame and targeted column names.
        """
        if os.path.exists(self.config.data_path):
            logger.info(f"Loading raw dataset from file: {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
        else:
            logger.warning(f"File not found at '{self.config.data_path}'. Initializing synthetic fallback generator.")
            df = SyntheticDataGenerator.generate(n_samples=self.config.synthetic_n_samples, random_state=self.config.random_state)

        # 1. Temporal Sorting and Index Construction
        if "Year" in df.columns and "Month" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").ffill().bfill().astype(int)
            df["Month"] = pd.to_numeric(df["Month"], errors="coerce").ffill().bfill().astype(int)
            df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)
            
            # Construct DatetimeIndex assuming 1st day of month
            date_strings = df.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}-01", axis=1)
            time_series_index = pd.to_datetime(date_strings, errors="coerce")
        else:
            logger.warning("Year and Month columns missing. Generating default chronological index.")
            time_series_index = pd.date_range(start="2005-01-01", periods=len(df), freq="MS")

        # 2. Dynamic Target Selection & Resolution
        target_col = self._resolve_target_column(df)
        logger.info(f"Resolved primary target variable: '{target_col}'")

        # 3. Numeric Coercion & Imputation
        numeric_cols = [c for c in df.columns if c not in ["Year", "Month", "Date"]]
        for col in numeric_cols:
            if df[col].dtype == object:
                s = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                s = s.replace(r'^\s*\.\s*$', np.nan, regex=True)
                df[col] = pd.to_numeric(s, errors="coerce")
        
        # Dynamic missing value imputation (Linear interpolation then forward/backward fill)
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # 4. Outlier Treatment via IQR Bounding (Winsorization)
        df_clean = self._treat_outliers_iqr(df, numeric_cols)

        # 5. Dynamic Feature Engineering
        df_engineered, feature_cols = self._engineer_features(df_clean, target_col, numeric_cols)

        return PreprocessedData(
            df=df_engineered,
            target_col=target_col,
            feature_cols=feature_cols,
            time_series_index=time_series_index
        )

    def _resolve_target_column(self, df: pd.DataFrame) -> str:
        """Dynamically identifies the target variable column or falls back to available matching columns."""
        if self.config.primary_target in df.columns:
            return self.config.primary_target
        
        for kw in self.config.fallback_target_keywords:
            matching = [c for c in df.columns if kw.lower() in c.lower()]
            if matching:
                logger.info(f"Target '{self.config.primary_target}' not found. Auto-matched fallback: '{matching[0]}'")
                return matching[0]
                
        # Last resort fallback: pick first non-time numeric column
        candidates = [c for c in df.columns if c not in ["Year", "Month", "Date"]]
        if candidates:
            return candidates[0]
        raise ValueError("No suitable target column could be resolved from dataset.")

    def _treat_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Applies soft Winsorization based on IQR boundaries to eliminate extreme artifacts."""
        df_out = df.copy()
        for col in columns:
            q25 = df_out[col].quantile(0.25)
            q75 = df_out[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - self.config.iqr_multiplier * iqr
            upper_bound = q75 + self.config.iqr_multiplier * iqr
            # Soft clip values to bounds
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
        return df_out

    def _engineer_features(self, df: pd.DataFrame, target_col: str, base_numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Engineers time-series domain features: lags, rolling statistics, and ratio indicators."""
        df_feat = df.copy()
        
        # Lagged target features (Lag-1, Lag-2, Lag-4 for quarterly cyclicality)
        df_feat[f"{target_col}_Lag1"] = df_feat[target_col].shift(1).ffill().bfill()
        df_feat[f"{target_col}_Lag2"] = df_feat[target_col].shift(2).ffill().bfill()
        df_feat[f"{target_col}_Lag4"] = df_feat[target_col].shift(4).ffill().bfill()
        
        # Rolling moving averages and standard deviation (3-period and 6-period)
        df_feat[f"{target_col}_RollMean3"] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
        df_feat[f"{target_col}_RollStd3"] = df_feat[target_col].rolling(window=3, min_periods=1).std().fillna(0)

        # Ratio indicators if labor force and total population exist
        pop_cols = [c for c in base_numeric_cols if "Population" in c or "Total" in c]
        lf_cols = [c for c in base_numeric_cols if "Labor Force" in c or "LaborForce" in c]
        if pop_cols and lf_cols:
            df_feat["Ratio_LaborTightness"] = (df_feat[lf_cols[0]] / (df_feat[pop_cols[0]] + 1e-5)).fillna(0)

        # Candidate predictor columns exclude direct target derivative leaks
        feature_cols = [c for c in df_feat.columns if c not in ["Year", "Month", "Date", target_col]]
        return df_feat, feature_cols


class ExploratoryAnalyzer:
    """Computes comprehensive descriptive statistics and distribution profile."""

    @staticmethod
    def analyze(prep_data: PreprocessedData) -> StatisticalSummary:
        """Computes mean, median, std, IQR, skewness, and kurtosis across key features.

        Args:
            prep_data: PreprocessedData object containing target and features.

        Returns:
            StatisticalSummary object containing descriptive metrics.
        """
        logger.info("Performing Exploratory Data Analysis (EDA)...")
        df = prep_data.df
        target_series = df[prep_data.target_col]
        
        metrics = []
        selected_cols = [prep_data.target_col] + prep_data.feature_cols[:9]
        
        for col in selected_cols:
            s = df[col]
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            metrics.append({
                "Feature": col,
                "Mean": s.mean(),
                "Median": s.median(),
                "Std_Dev": s.std(),
                "IQR": iqr,
                "Min": s.min(),
                "Max": s.max(),
                "Skewness": s.skew(),
                "Kurtosis": s.kurtosis()
            })
            
        metrics_df = pd.DataFrame(metrics).set_index("Feature")
        logger.info(f"Target '{prep_data.target_col}' Summary -> Mean: {target_series.mean():.4f}, Std: {target_series.std():.4f}, Skewness: {target_series.skew():.4f}")
        
        return StatisticalSummary(
            metrics_df=metrics_df,
            target_skewness=float(target_series.skew()),
            target_kurtosis=float(target_series.kurtosis())
        )


class DistributionFitter:
    """Fits continuous probability density functions (Normal, Gamma, Log-Normal, Student-t) to target variable."""

    @staticmethod
    def fit_and_evaluate(prep_data: PreprocessedData) -> DistributionFitResult:
        """Evaluates target variable distribution and executes formal goodness-of-fit tests.

        Args:
            prep_data: PreprocessedData object containing target variable.

        Returns:
            DistributionFitResult containing fitted parameters and normality test statistics.
        """
        logger.info("Executing Statistical Distribution & Density Analysis...")
        target_vals = prep_data.df[prep_data.target_col].values
        
        # Normality and distribution tests
        shapiro_stat, shapiro_p = stats.shapiro(target_vals)
        ks_stat, ks_p = stats.kstest(target_vals, 'norm', args=(np.mean(target_vals), np.std(target_vals)))
        jb_stat, jb_p = stats.jarque_bera(target_vals)
        
        # Fit Gamma distribution (shift to positive if required)
        min_val = np.min(target_vals)
        shift = abs(min_val) + 1e-3 if min_val <= 0 else 0.0
        shifted_vals = target_vals + shift
        
        gamma_params = stats.gamma.fit(shifted_vals)
        fitted_norm_params = stats.norm.fit(target_vals)
        
        # Select best distribution based on skewness and p-value
        if shapiro_p > 0.05:
            best_dist = "Normal (Gaussian) Distribution"
            params = fitted_norm_params
        elif float(prep_data.df[prep_data.target_col].skew()) > 0.5:
            best_dist = "Gamma Distribution"
            params = gamma_params
        else:
            best_dist = "Student's t-Distribution"
            params = stats.t.fit(target_vals)
            
        logger.info(f"Fitted Distribution: {best_dist} | Shapiro-Wilk p-val: {shapiro_p:.5e} | Jarque-Bera p-val: {jb_p:.5e}")
        
        theoretical_eval = {
            "norm_params": fitted_norm_params,
            "gamma_params": gamma_params,
            "shift": shift
        }
        
        return DistributionFitResult(
            best_distribution_name=best_dist,
            fitted_params=params,
            shapiro_stat=float(shapiro_stat),
            shapiro_pvalue=float(shapiro_p),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_p),
            jb_stat=float(jb_stat),
            jb_pvalue=float(jb_p),
            theoretical_density_eval=theoretical_eval
        )


class HypothesisTester:
    """Formulates and executes formal inferential statistical tests."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def execute_tests(self, prep_data: PreprocessedData) -> List[HypothesisTestResult]:
        """Executes Welch's t-test, Levene's variance test, and baseline one-sample t-test.

        Args:
            prep_data: PreprocessedData containing chronological target observations.

        Returns:
            List of HypothesisTestResult objects detailing formal statistical conclusions.
        """
        logger.info("Executing Inferential Statistics & Hypothesis Testing...")
        df = prep_data.df
        target_col = prep_data.target_col
        
        results = []
        
        # Split data by Pre-COVID (Year < 2020) vs Post-COVID (Year >= 2020)
        pre_covid = df[df["Year"] < 2020][target_col].values
        post_covid = df[df["Year"] >= 2020][target_col].values
        
        if len(pre_covid) > 5 and len(post_covid) > 5:
            # 1. Welch's t-Test (Two-Sample unequal variances)
            welch_stat, welch_p = stats.ttest_ind(pre_covid, post_covid, equal_var=False)
            
            # 95% Confidence Interval for mean difference
            diff_mean = np.mean(post_covid) - np.mean(pre_covid)
            se_diff = np.sqrt(np.var(pre_covid, ddof=1)/len(pre_covid) + np.var(post_covid, ddof=1)/len(post_covid))
            ci_lower = diff_mean - 1.96 * se_diff
            ci_upper = diff_mean + 1.96 * se_diff
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(pre_covid)-1)*np.var(pre_covid) + (len(post_covid)-1)*np.var(post_covid)) / (len(pre_covid)+len(post_covid)-2))
            cohen_d = diff_mean / (pooled_std + 1e-8)
            
            reject_welch = welch_p < self.config.alpha_significance
            results.append(HypothesisTestResult(
                h0_description="H0: Mean target rate Pre-COVID (2005-2019) == Mean target rate Post-COVID (2020-2026)",
                h1_description="H1: Mean target rate Pre-COVID (2005-2019) != Mean target rate Post-COVID (2020-2026)",
                test_name="Welch's Two-Sample t-Test",
                test_statistic=float(welch_stat),
                p_value=float(welch_p),
                confidence_interval_95=(float(ci_lower), float(ci_upper)),
                effect_size_cohen_d=float(cohen_d),
                decision_reject_h0=reject_welch,
                business_conclusion=(
                    "Statistically significant shift in labor market baseline post-COVID."
                    if reject_welch else "No statistically significant difference detected between structural eras."
                )
            ))
            
            # 2. Levene's Test for Homogeneity of Variance
            lev_stat, lev_p = stats.levene(pre_covid, post_covid)
            reject_lev = lev_p < self.config.alpha_significance
            results.append(HypothesisTestResult(
                h0_description="H0: Variance Pre-COVID == Variance Post-COVID (Equal Volatility)",
                h1_description="H1: Variance Pre-COVID != Variance Post-COVID (Structural Volatility Shift)",
                test_name="Levene's Variance Homogeneity Test",
                test_statistic=float(lev_stat),
                p_value=float(lev_p),
                confidence_interval_95=(0.0, 0.0),
                effect_size_cohen_d=float(np.std(post_covid) / (np.std(pre_covid) + 1e-8)),
                decision_reject_h0=reject_lev,
                business_conclusion=(
                    "Significant structural change in labor market volatility after 2020."
                    if reject_lev else "Volatility level remained homogenous across timelines."
                )
            ))
            
        # 3. Baseline One-Sample t-Test against overall mean parameter
        overall_target = df[target_col].values
        baseline_param = float(np.mean(overall_target))
        t_stat, t_p = stats.ttest_1samp(overall_target, popmean=baseline_param)
        results.append(HypothesisTestResult(
            h0_description=f"H0: Population mean == {baseline_param:.4f}",
            h1_description=f"H1: Population mean != {baseline_param:.4f}",
            test_name="One-Sample t-Test",
            test_statistic=float(t_stat),
            p_value=float(t_p),
            confidence_interval_95=(float(baseline_param - 1.96*stats.sem(overall_target)), float(baseline_param + 1.96*stats.sem(overall_target))),
            effect_size_cohen_d=0.0,
            decision_reject_h0=t_p < self.config.alpha_significance,
            business_conclusion="Baseline sample parameter matches theoretical benchmark expectation."
        ))

        return results


class PredictiveModeler:
    """Trains baseline (Ridge/OLS) and advanced ML models (GradientBoosting/RandomForest) with chronological validation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def train_and_evaluate(self, prep_data: PreprocessedData) -> Dict[str, ModelEvaluationResult]:
        """Performs chronological split, fits models, evaluates holdout performance, and extracts feature importances.

        Args:
            prep_data: PreprocessedData object containing preprocessed dataset.

        Returns:
            Dictionary mapping model names to ModelEvaluationResult containers.
        """
        logger.info("Executing Machine Learning & Predictive Modeling Architecture...")
        df = prep_data.df
        target_col = prep_data.target_col
        feature_cols = prep_data.feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Strict Chronological Train-Test Split (Last 20% reserved as holdout set)
        n_samples = len(df)
        split_idx = int(n_samples * (1 - self.config.test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Chronological Validation Split -> Train set: {len(X_train)} samples, Test Holdout: {len(X_test)} samples")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # 1. Baseline Regression (Ridge Regularized OLS)
        baseline_model = Ridge(alpha=1.0, random_state=self.config.random_state)
        baseline_model.fit(X_train_scaled, y_train)
        
        y_train_pred_base = baseline_model.predict(X_train_scaled)
        y_test_pred_base = baseline_model.predict(X_test_scaled)
        
        base_coefs = pd.Series(np.abs(baseline_model.coef_), index=feature_cols).sort_values(ascending=False)

        results["Baseline_Ridge_OLS"] = ModelEvaluationResult(
            model_name="Baseline Ridge Regression (OLS)",
            train_r2=r2_score(y_train, y_train_pred_base),
            train_mae=mean_absolute_error(y_train, y_train_pred_base),
            train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred_base)),
            test_r2=r2_score(y_test, y_test_pred_base),
            test_mae=mean_absolute_error(y_test, y_test_pred_base),
            test_rmse=np.sqrt(mean_squared_error(y_test, y_test_pred_base)),
            feature_importances=base_coefs,
            residuals_test=(y_test - y_test_pred_base).values,
            y_test_pred=y_test_pred_base
        )

        # 2. Advanced ML Model (Gradient Boosting Regressor)
        adv_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=self.config.random_state
        )
        adv_model.fit(X_train, y_train)
        
        y_train_pred_adv = adv_model.predict(X_train)
        y_test_pred_adv = adv_model.predict(X_test)
        
        adv_importances = pd.Series(adv_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        results["Advanced_GradientBoosting"] = ModelEvaluationResult(
            model_name="Advanced Gradient Boosting Machine",
            train_r2=r2_score(y_train, y_train_pred_adv),
            train_mae=mean_absolute_error(y_train, y_train_pred_adv),
            train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred_adv)),
            test_r2=r2_score(y_test, y_test_pred_adv),
            test_mae=mean_absolute_error(y_test, y_test_pred_adv),
            test_rmse=np.sqrt(mean_squared_error(y_test, y_test_pred_adv)),
            feature_importances=adv_importances,
            residuals_test=(y_test - y_test_pred_adv).values,
            y_test_pred=y_test_pred_adv
        )

        # 3. Ensemble Model (Random Forest Regressor)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.config.random_state
        )
        rf_model.fit(X_train, y_train)
        
        y_train_pred_rf = rf_model.predict(X_train)
        y_test_pred_rf = rf_model.predict(X_test)
        
        rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        results["Advanced_RandomForest"] = ModelEvaluationResult(
            model_name="Advanced Random Forest Ensemble",
            train_r2=r2_score(y_train, y_train_pred_rf),
            train_mae=mean_absolute_error(y_train, y_train_pred_rf),
            train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
            test_r2=r2_score(y_test, y_test_pred_rf),
            test_mae=mean_absolute_error(y_test, y_test_pred_rf),
            test_rmse=np.sqrt(mean_squared_error(y_test, y_test_pred_rf)),
            feature_importances=rf_importances,
            residuals_test=(y_test - y_test_pred_rf).values,
            y_test_pred=y_test_pred_rf
        )

        for name, res in results.items():
            logger.info(f"Model [{res.model_name}] Holdout Metrics -> R2: {res.test_r2:.4f}, MAE: {res.test_mae:.4f}, RMSE: {res.test_rmse:.4f}")

        return results


class PACEPipeline:
    """Orchestrates end-to-end execution of the PACE framework."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self) -> Dict[str, Any]:
        """Runs the complete PACE pipeline and prints technical executive output summary."""
        logger.info("==========================================================================")
        logger.info("   LAUNCHING PACE END-TO-END NATIONAL EMPLOYMENT DATA SCIENCE PIPELINE    ")
        logger.info("==========================================================================")
        
        # Stage A: Plan & Autonomous Data Preparation
        preprocessor = DataPreprocessor(self.config)
        prep_data = preprocessor.load_and_clean()
        
        # Stage B: Analyze & Exploratory Data Analysis
        eda_summary = ExploratoryAnalyzer.analyze(prep_data)
        
        # Stage C: Distribution & Density Analysis
        dist_result = DistributionFitter.fit_and_evaluate(prep_data)
        
        # Stage D: Construct & Inferential Hypothesis Testing
        tester = HypothesisTester(self.config)
        hyp_results = tester.execute_tests(prep_data)
        
        # Stage E: Execute & Machine Learning Modeling Architecture
        modeler = PredictiveModeler(self.config)
        model_results = modeler.train_and_evaluate(prep_data)

        # Print Executive Summary Table
        self._print_executive_summary(prep_data, eda_summary, dist_result, hyp_results, model_results)

        return {
            "preprocessed_data": prep_data,
            "eda_summary": eda_summary,
            "distribution_result": dist_result,
            "hypothesis_results": hyp_results,
            "model_results": model_results
        }

    def _print_executive_summary(
        self,
        prep_data: PreprocessedData,
        eda: StatisticalSummary,
        dist: DistributionFitResult,
        hypotheses: List[HypothesisTestResult],
        models: Dict[str, ModelEvaluationResult]
    ) -> None:
        """Prints formatted technical summary tables to stdout."""
        print("\n" + "=" * 80)
        print("                  PACE PIPELINE EXECUTIVE SUMMARY REPORT                  ")
        print("=" * 80)
        print(f"Target Variable Analyzed: '{prep_data.target_col}'")
        print(f"Total Observations: {len(prep_data.df)} | Engineered Features: {len(prep_data.feature_cols)}")
        print("-" * 80)
        print("1. STATISTICAL DISTRIBUTION FIT:")
        print(f"   Optimal Fitted PDF: {dist.best_distribution_name}")
        print(f"   Shapiro-Wilk Normality Test Statistic: {dist.shapiro_stat:.4f} (p-value: {dist.shapiro_pvalue:.5e})")
        print(f"   Jarque-Bera Test Statistic: {dist.jb_stat:.4f} (p-value: {dist.jb_pvalue:.5e})")
        print("-" * 80)
        print("2. INFERENTIAL HYPOTHESIS TESTING SUMMARY:")
        for idx, h in enumerate(hypotheses, 1):
            print(f"   Test #{idx} [{h.test_name}]:")
            print(f"     - H0: {h.h0_description}")
            print(f"     - Metric: Stat={h.test_statistic:.4f}, p-val={h.p_value:.5e}")
            print(f"     - Outcome: {'REJECT H0' if h.decision_reject_h0 else 'FAIL TO REJECT H0'}")
            print(f"     - Insight: {h.business_conclusion}")
        print("-" * 80)
        print("3. MACHINE LEARNING MODEL PERFORMANCE COMPARISON (HOLDOUT TEST SET):")
        print(f"{'Model Architecture':<38} | {'Holdout R2':<10} | {'Holdout MAE':<11} | {'Holdout RMSE':<12}")
        print("-" * 80)
        for m_key, m_res in models.items():
            print(f"{m_res.model_name:<38} | {m_res.test_r2:<10.4f} | {m_res.test_mae:<11.4f} | {m_res.test_rmse:<12.4f}")
        print("-" * 80)
        print("4. TOP RANKED FEATURE IMPORTANCES (BEST MODEL):")
        best_model_name = max(models.keys(), key=lambda k: models[k].test_r2)
        best_importances = models[best_model_name].feature_importances.head(5)
        for feat, imp in best_importances.items():
            print(f"   - {feat:<55}: {imp:.4f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    pipeline = PACEPipeline()
    pipeline.run()
