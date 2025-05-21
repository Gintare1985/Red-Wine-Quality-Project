import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
)
from statsmodels.graphics.regressionplots import add_lowess
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper
from statsmodels.stats.outliers_influence import OLSInfluence


def get_general_df_info(df: pd.DataFrame) -> None:
    """
    Outputs general information about the dataset: number of features, if there are
    missing or duplicate values.

    Args:
        df (pd.DataFrame): The dataframe that is analyzed.
    """
    general_info = {
        "number_rows": df.shape[0],
        "number_features": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "amount_duplicated_rows": df.duplicated().sum(),
        "categorical_data": df.select_dtypes(exclude="number").columns.to_numpy(),
        "numerical_data": df.select_dtypes(include="number").columns.to_numpy(),
    }
    df.info()
    print(
        f"Dataset has {general_info['number_rows']} rows and {general_info['number_features']} features."
    )
    print(
        f"Amount of missing values in the features are: \n{general_info['missing_values']}"
    )
    print(
        f"Amount of duplicated rows is {int(general_info['amount_duplicated_rows'])}."
    )
    print(
        f"Categorical features are: {general_info['categorical_data']},\n numerical features are: {general_info['numerical_data']}"
    )


def plot_triangle_heatmap(
    data: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    annot_size: float | int,
    title: str,
    cmap: str,
    vmax: float | int,
    vmin: float | int,
    label_size: float | int = 9,
    title_size: float | int = 12,
    float_precision: str = ".1f",
) -> None:
    """
    Plots lower triangular heatmap from a given correlation matrix.

    Args:
        data (pd.DataFrame): The correlation matrix.
        figsize (tuple[float|int, float|int]): Width and height of the plot.
        annot_size (float | int): Font size for annotations inside heatmap cells.
        title (str): The title of the heatmap.
        cmap (str): Color palette for the heatmap.
        vmax (float | int): Max value to anchor to the heatmap.
        vmin (float | int): Min value to anchor to the heatmap.
        label_size (float | int, optional): Font size for axes labels. Defaults to 9.
        title_size (float | int, optional): Font size for the title. Defaults to 12.
        float_precision (str, optional): String format for displaying numerical values. Defaults to '.1f'.
    """
    mask = np.triu(np.ones_like(data, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(
        data,
        mask=mask,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        annot=True,
        fmt=float_precision,
        linewidths=0.5,
        annot_kws={"size": annot_size},
    )
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.title(title, fontsize=title_size)
    plt.grid(False)
    plt.show()


def plot_numerical_feature_distribution(
    df: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    numerical_features: list[str],
    hue: str,
    title_right: float,
    title_top: float,
    wspace: float,
    hspace: float,
    suptitle: str,
    shrink: float | int = 1,
    statistic: str = "percent",
) -> None:
    """
    Plots distribution of the metric of the categorical feature on the subplots using hue parameter.

    Args:
        df (pd.DataFrame): The original dataset.
        figsize (tuple[float|int, float|int]): The whole figure size.
        numerical_features (list[str]): The list of numerical features to plot.
        hue (str): The additional feature to analyze the metric of the main feature.
        title_right (float): The magnitude to move suptitle to the right.
        title_top (float): The magnitude to move suptitle to the top.
        wspace (float): The width of the padding between subplots.
        hspace (float): The height of the padding between subplots.
        suptitle (str): The title for the whole figure.
        shrink (float|int): The higher the coefficient, the wider the bar. Default value is 1.
        statistic(str): Statistic from those available under sns.histplot.

    """
    nrows = len(numerical_features)
    ncols = 4 if hue else 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            feature = numerical_features[row]
            if col == 0:
                sns.histplot(ax=ax, data=df, x=feature, shrink=shrink, stat=statistic)
                ax.set_xlabel(feature)
                ax.set_title(f"Distribution of {feature}", pad=20)
                ax.grid(visible=False, axis="x")
            elif col == 1:
                sns.boxplot(ax=ax, data=df, x=feature)
                ax.set_xlabel(feature)
                ax.set_title(f"Boxplot of {feature}", pad=20)
            elif col == 2:
                sns.scatterplot(ax=ax, data=df, x=feature, y="quality")
                ax.set_xlabel(feature)
                ax.set_title(f"Scatterplot of {feature} vs. {hue}", pad=20)
            elif col == 3:
                sns.boxplot(ax=ax, data=df, y=feature, hue=hue, palette="viridis")
                ax.set_ylabel(hue)
                ax.set_title(f"Boxplots of {feature} by wine {hue}", pad=20)
                ax.legend(
                    title="Wine quality level",
                    loc="upper left",
                    bbox_to_anchor=(0.98, 1.1),
                    alignment="left",
                )

    fig.suptitle(suptitle, x=title_right, y=title_top)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
    return df[numerical_features].describe().T.map(lambda x: f"{x:.2f}")


def make_polyplot(
    df: pd.DataFrame,
    feature: str,
    hue: str,
    fig_size: tuple[int | float],
    title: str,
    legend_title: str,
    legend_loc: tuple[float],
    multiple: str = "stack",
) -> None:
    """
    Makes the histogram as a polyplot.

    Args:
        df (pd.DataFrame): The dataframe of the data to be used for the plot.
        feature (str): A numerical feature on the x axis.
        hue (str): The categorical feature used as hue.
        fig_size (tuple[int | float]): Figure size.
        title (str): The title of the plot.
        legend_title (str): The title of the legend.
        legend_loc (tuple[float]): The location of the legend.
        multiple (str, optional): The display mode of the data on the plot for different hues. Defaults to 'stack'.
    """

    ax = plt.subplots(figsize=fig_size)[1]
    sns.histplot(
        data=df,
        x=feature,
        hue=hue,
        kde=False,
        element="poly",
        palette="viridis",
        multiple=multiple,
        ax=ax,
    )
    ax.set_xlabel(feature)
    ax.set_title(title, pad=20)
    sns.move_legend(
        ax,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=legend_loc,
        alignment="left",
    )
    plt.show()


def make_pairplot(
    df: pd.DataFrame,
    features: list[str],
    height: float,
    aspect: float,
    alpha: float,
    suptitle: str,
    top_right: float,
    top: float,
) -> None:
    """
    Makes pairplots for correlating features.

    Args:
        df (pd.DataFrame): The original dataset.
        features (list[str]): Numerical features.
        height (float): Height of the figure.
        aspect (float): Width ratio with respect to height.
        alpha (float): Scatterplot transparency level.
        suptitle (str): Suptitle of the figure.
        top_right (float): Shifting figure title to the right.
        top (float): Moving figure title top.
    """
    sns.pairplot(
        df.loc[:, features],
        corner=True,
        height=height,
        aspect=aspect,
        plot_kws={"alpha": alpha},
    )
    plt.suptitle(suptitle, x=top_right, y=top, fontsize=11)
    plt.show()


def make_transformed_feature_plots(
    df: pd.DataFrame,
    key: str,
    transform_func: str,
    response: str,
    fig_size: tuple[int | float],
    top_right: float,
    top: float,
    suptitle: str,
    colors: list[str] = ["#0173b2", "#de8f05"],
    pad: int = 15,
) -> None:
    """
    Plots the histogram and a scatterplot for the original feature and the transformed its version.

    Args:
        df (pd.DataFrame): The dataframe with the data to be used for plot.
        key (str): The name of the main feature.
        transform_func (str): Feature's transformation function.
        response (str): The response feature name.
        fig_size (tuple[int | float]): Figure size in inches.
        top_right (floact): Suptitle's position to the right.
        top (float): Suptitle's position to the top.
        suptitle (str): The suptitle of the subplots.
        colors (list[str], optional): The codes of the colors used for the plots. Defaults to ['#0173b2', '#de8f05'].
        pad (int, optional): Padding of the axis title. Defaults to 15.
    """
    axes = plt.subplots(2, 2, figsize=fig_size)[1]
    features = [df[key], transform_func(df[key])]
    for i, ax in enumerate(axes.flat):
        if i < 2:
            sns.scatterplot(ax=ax, x=features[i], y=df[response], color=colors[i])
            if i == 0:
                ax.set_title(f"Relationship of {key} with {response}", pad=pad)
            else:
                ax.set_title(
                    f"Relationship of transformed {key} with {response}", pad=pad
                )
        else:
            sns.histplot(ax=ax, x=features[i - 2], color=colors[i - 2])
            if i == 2:
                ax.set_title(f"Distribution of {key}", pad=pad)
            else:
                ax.set_title(f"Distribution of transformed {key}", pad=pad)

    plt.suptitle(suptitle, x=top_right, y=top)
    plt.tight_layout()
    plt.show()


def fit_linear_regression_and_get_results(
    y: pd.DataFrame, X: pd.DataFrame
) -> tuple[RegressionResultsWrapper, float, float]:
    """
    Fits OLS model, return its results class object, prints out the summary table together with MAE and RSE values.

    Args:
        y (pd.DataFrame): The dataframe of endogenous variable.
        X (pd.DataFrame): The dataframe of exogenous variables with an intercept column.

    Returns:
    results : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS results object.
    RSE : float
        Residual Standard Error.
    MAE : float
        Mean Absolute Error.
    """
    model = sm.OLS(y, X)
    results = model.fit()

    y_predict = results.fittedvalues
    MAE = mean_absolute_error(y, y_predict)
    RSE = np.sqrt(results.mse_resid)

    print(f"RSE: {np.round(RSE, 4)}, \nMAE: {np.round(MAE, 4)}")
    print(results.summary())
    return results, RSE, MAE


def customize_response(value: float) -> int:
    """
    Rounds up a float if fraction >= 0.5 or down the otherwise.

    Args:
        value (float): The value to round.

    Returns:
        int: The rounded float to integer.
    """
    fraction = value % 1
    return np.where(fraction < 0.5, np.floor(value), np.ceil(value))


def plot_residuals_fitted_values(
    model: RegressionResultsWrapper, fig_size: tuple[float | int]
) -> None:
    """
    Makes the plot of residuals vs fitted values plot.

    Args:
        model (RegressionResultsWrapper): The fitted regression model object.
        fig_size (tuple[float | int]): Figure size.
    """
    ax = plt.subplots(figsize=fig_size)[1]
    sns.residplot(
        x=customize_response(model.fittedvalues),
        y=customize_response(model.resid),
        lowess=True,
        line_kws={"color": "darkorange"},
        ax=ax,
    )
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals (Observed - Fitted values)")
    ax.set_xlim(3, 8)
    ax.set_title("Residuals Plot for Fitted Values", pad=20)
    ax.axhline(0, c="k", ls="--")
    plt.show()


def rescale_to_units(
    coef: float,
    std: float,
    change: float,
    transformation: str,
    none_transf_stats: list[float],
) -> float:
    """
    Rescales the transformed and standardized or only standardized predictor's OLS regression
    coefficient to reflect its change in the original units.

    Args:
        coef (float): Coefficient fromOLS model.
        std (float): Predictor's standard deviation.
        change (float): Marginal change.
        transformation (str): Transformation made to original predictor.
        none_transf_stats (float): Statistic of not transformed predictor.

    Returns:
        float: The magnitude in predictor original units after accounting for marginal change.
    """

    if transformation == "Log":
        # Rescale beta coefficient
        beta_log = coef / std
        # Evaluate effect on response after increasing predictor
        marginal_change = round(np.log(1 + change) * beta_log, 4)

    elif transformation == "None":
        i = 0
        # Change in predictor relative to the mean
        # Get fraction of standard deviation
        predictor_scaled = change * none_transf_stats[i][1] / none_transf_stats[i][0]
        marginal_change = round(coef * predictor_scaled, 4)
        i += 1
    return marginal_change


def rescale_to_odds(
    coef: float,
    std: float,
    change: float,
    transformation: str,
    none_transf_stats: list[float, float],
) -> tuple[float, float]:
    """
    Calculates odds ratio and percentage change in odds ratio of the log-transformed and scaled
    or only scaled predictor of defined percentage change in it.

    Args:
        coef (float): Ordinal model coefficient of a particular predictor.
        std (float): Standard deviation of the predictor.
        change (float): Percentage change in the predictor.
        transformation (str): Transformation used for the predictor.
        none_transf_stats (list[float, float]): Standard deviation and mean for only scaled predictors.

    Returns:
        tuple[float, float]: Calculated odds ratio and percentage change in odds ratio.
    """
    if transformation == "Log":
        beta_log = coef / std
        odds_ratio = round((1 + change) ** beta_log, 4)

    elif transformation == "None":
        i = 0
        beta_orig = coef * change * none_transf_stats[i][1] / none_transf_stats[i][0]
        odds_ratio = round(np.exp(beta_orig), 4)
        i += 1

    perc_change = (odds_ratio - 1) * 100
    return odds_ratio, perc_change


def plot_residuals_dist(
    residuals: pd.Series,
    figsize: tuple[float | int, float | int],
    title: str,
    shrink: float = 0.7,
    stat: str = "percent",
    decimals: str = ".1f",
    fontsize: float | int = 9,
    ax: matplotlib.axes.Axes = None,
) -> None:
    """
    _summary_

    Args:
        residuals (pd.Series): An array of residuals.
        figsize (tuple[float|int, float|int]): Width and height of the plot.
        title (str): A title of the plot.
        shrink (float, optional): The width of the bars. Defaults to 0.7.
        stat (str, optional): Used statistics for distribution. Defaults to "percent".
        decimals (str, optional): Format of the values displayed on the bars. Defaults to ".1f".
        fontsize (float | int, optional): The size of the annotations on the bars. Defaults to 9.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object. Defaults to None.
    """
    if ax is None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = ax
    sns.histplot(data=residuals, discrete=True, shrink=shrink, ax=ax, stat=stat)
    ax.set_xlabel("Residuals (observed - predicted)")
    ax.set_title(title, pad=20)
    ax.grid(False)
    for bar in ax.patches:
        height = bar.get_height()
        if height == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            height + 0.01,
            s=f"{height:{decimals}}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
    plt.show()


def make_and_customize_confusion_matrix(
    labels: list[int | str],
    y_test: pd.Series | pd.DataFrame,
    y_pred: pd.Series | pd.DataFrame,
    figsize: tuple[float | int, float | int],
    cmap: str = "Blues",
    fontsize: float | int = 9,
    titlesize: float | int = 12,
) -> None:
    """
    Makes a confusion matrix plot.

    Args:
        labels (list[int | str]): Labels for confusion matrix rows and columns.
        y_test (pd.Series | pd.DataFrame): Values of the response variable on test data.
        y_pred (pd.Series | pd.DataFrame): Values of the response predictions on the test data.
        figsize (tuple[float | int, float | int]): Matrix width and height.
        cmap (str, optional): Palette for heatmap. Defaults to "Blues".
        fontsize (float | int, optional): Size of annotations in the heatmap. Defaults to 9.
        titlesize (float | int, optional): Size of the title. Defaults to 12.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap=cmap, colorbar=False, text_kw={"fontsize": fontsize})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix", pad=15, fontsize=titlesize)
    ax.grid(False)
    plt.show()


def compare_regression_models(
    results: list[RegressionResultsWrapper],
    RSE_list: list[float],
    MAE_list: list[float],
    figsize: tuple[float | int, float | int],
    suptitle: str,
    right: float,
    top: float,
    wspace: float,
) -> None:
    """
    Plots pointplot subplots for regression model metrics.

    Args:
        results (list[RegressionResultsWrapper]): Results for regression models to compare.
        RSE_list (list[float]): A list of RSE metrics for different models.
        MAE_list (list[float]): A list of MAE metrics for different models.
        figsize (tuple[float | int, float | int]): Figure width and height.
        suptitle (str): The title of the figure.
        right (float): The shift of the suptitle to the right.
        top (float): The shift of the suptitle to top.
        wspace (float): Spacing between subplots.
    """
    AIC_list = [result.aic for result in results]
    BIC_list = [result.bic for result in results]
    metrics_df = pd.DataFrame(
        data={"AIC": AIC_list, "BIC": BIC_list, "RSE": RSE_list, "MAE": MAE_list},
        index=["Mod.1", "Mod.2", "Mod.3", "Mod.4", "Mod.5"],
    )
    metrics_df = metrics_df.reset_index().rename(columns={"index": "Models"})

    columns = metrics_df.columns[1:]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, col in enumerate(columns):
        if i < 2:
            sns.pointplot(data=metrics_df, y=col, x="Models", ax=axes[0], label=col)
            axes[0].set_title("AIC and BIC metrics")
            axes[0].set_ylabel("Metric")
        else:
            sns.pointplot(data=metrics_df, y=col, x="Models", ax=axes[1], label=col)
            axes[1].set_title("MAE and RSE metrics")
            axes[1].set_ylabel("Metric")
    for ax in axes:
        ax.legend(
            loc="upper left",
            title="Metrics",
            fontsize=9,
            title_fontsize=9,
            bbox_to_anchor=(1, 1),
            alignment="left",
        )

    fig.subplots_adjust(wspace=wspace)
    fig.suptitle(suptitle, x=right, y=top)
    plt.show()


def transform_and_scale_predictors(
    cols: list[str],
    transform_funcs: list[str],
    predictors_df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Trandsforms and scales predictors' dataframe.

    Args:
        cols (list[str]): The list of predictors' names.
        transform_funcs (list[str]): Chosen transformation functions for certain predictors.
        predictors_df (pd.DataFrame): The original dataframe of predictors.
        scaler (StandardScaler): The StandardScaler obejct.

    Returns:
        pd.DataFrame: The transformed and scaled predictors' dataframe.
    """
    df_pred_transformed = pd.DataFrame(columns=cols)
    for i, col in enumerate(cols):
        func = transform_funcs[i]
        if func is not None:
            df_pred_transformed[col] = func(predictors_df[col])
        else:
            df_pred_transformed[col] = predictors_df[col]

    scaled_data = scaler.fit_transform(df_pred_transformed)
    df_pred_transf_scaled = pd.DataFrame(
        scaled_data,
        columns=df_pred_transformed.columns,
        index=df_pred_transformed.index,
    )
    return df_pred_transf_scaled


def make_subplots_grid(
    fig: matplotlib.figure.Figure,
    width: float,
    height: float,
    alpha: float,
    color1: str,
    color2: str,
    wspace: float,
    hspace: float,
    suptitle: str,
    right: float,
    top: float,
) -> None:
    """
    Customize subplots grid.

    Args:
        fig (matplotlib.figure.Figure): The existing subplots grid object.
        width (float): The width of the figure.
        height (float): The height of the figure.
        alpha (float): Transparency level.
        color1 (str): Color used for the reference lines.
        color2 (str): Another color used for reference lines.
        wspace (float): Horizontal spacing of subplots.
        hspace (float): Vertical spacing of subplots.
        suptitle (str): The title of the figure.
        right (float): The shift of the title to the right.
        top (float): The shift of the title to top.
    """
    fig.set_size_inches(width, height)
    for ax in fig.axes:
        ax.lines[0].set_alpha(alpha)
        ax.lines[1].set_color(color1)
        ax.lines[1].set_linewidth(2)
        ax.lines[1].set_linestyle("--")
        add_lowess(ax)
        ax.lines[-1].set_color(color2)
        ax.lines[-1].set_linestyle("-")
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.suptitle(suptitle, x=right, y=top)
    plt.show()


def plot_residuals_normality(
    residuals: np.ndarray,
    key: str,
    figsize: tuple[float | int, float | int],
    wspace: float,
    suptitle: str,
    right: float,
    top: float,
    hist_shrink: float = 0.7,
    hist_stat: str = "percent",
) -> None:
    """
    Makes a histogram of residuals and QQ plot for testing residuals' normality assumption.

    Args:
        residuals (np.ndarray): The array of the residuals.
        key (str): The string use in a title.
        figsize (tuple[float | int, float | int]): The figure width and height.
        wspace (float): Horizontal spacing between subplots.
        suptitle (str): The title of the figure.
        right (float): The shift of the tail to the right.
        top (float): The shift of the tail to the top.
        hist_shrink (float, optional): The width of the bars. Defaults to 0.7.
        hist_stat (str, optional): The chosen statistic for distribution. Defaults to "percent".
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(
        data=residuals,
        discrete=True,
        shrink=hist_shrink,
        ax=axes[0],
        stat=hist_stat,
        label="Residuals",
    )
    axes[0].set_xlabel("Residuals (observed - predicted)")
    axes[0].set_title(f"Distribution of residuals for the {key} data", pad=20)
    axes[0].grid(False)
    for bar in axes[0].patches:
        height = bar.get_height()
        if height == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        axes[0].text(
            x, height + 0.01, s=f"{height:.1f}%", ha="center", va="bottom", fontsize=9
        )

    sm.qqplot(residuals, line="45", fit=True, ax=axes[1], label="Residuals")
    axes[1].set_title("QQ plot of the residuals", pad=20)

    for ax in axes:
        ax.legend(
            loc="upper left",
            title="Metrics",
            fontsize=9,
            title_fontsize=9,
            bbox_to_anchor=(1.05, 1),
            alignment="left",
        )
    fig.subplots_adjust(wspace=wspace)
    fig.suptitle(suptitle, x=right, y=top)
    plt.show()


def plot_leverage(
    infl: OLSInfluence,
    leverage_thresh: float,
    figsize: tuple[float | int, float | int],
    X: pd.DataFrame,
    line_color: str = "darkorange",
    linewidth: float = 2,
) -> int:
    """
    Makes leverage plot for the chosen predictors.
    Returns the index of the observation with the highest leverage.

    Args:
        infl (OLSInfluence): Influence object from statsmodels.
        leverage_thresh (float): Threshold for the leverage.
        figsize (tuple[float | int, float | int]): The figure width and height.
        X (pd.DataFrame): The dataframe of the predictors.
        line_color (str, optional): The color of the reference line. Defaults to "darkorange".
        linewidth (float, optional): The width of the reference line. Defaults to 2.

    Returns:
        int: The index of the observation with the highest leverage.
    """

    ax = plt.subplots(figsize=figsize)[1]
    ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag, label="Observations")
    ax.set_xlabel("Index")
    ax.set_ylabel("Leverage")
    ax.axhline(
        leverage_thresh,
        color=line_color,
        linestyle="--",
        label="Leverage threshold",
        linewidth=linewidth,
    )
    ax.set_title("Leverage statistics for observations", pad=20)
    ax.legend(
        loc="upper left",
        title="Metrics",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1, 1),
        alignment="left",
    )
    plt.show()
    max_leverage = np.argmax(infl.hat_matrix_diag)
    print(f"Index of the highest leverage observation  - {max_leverage}")
    return max_leverage


def plot_residuals_vs_leverage(
    leverage: np.ndarray,
    studentized_resid: np.ndarray,
    leverage_thresh: float,
    alpha: float,
    figsize: tuple[float | int, float | int],
    color1: str = "darkorange",
    color2: str = "firebrick",
) -> None:
    """
    Make scatterplot of leverage vs. studentized residual metrics for observations.

    Args:
        leverage (np.ndarray): The levarage score for observations.
        studentized_resid (np.ndarray): The array of studentized residuals.
        leverage_thresh (float): The threshold of the leverage.
        alpha (float): The transparency level.
        figsize (tuple[float | int, float | int]): The width and height of the figure.
        color1 (str, optional): The color of the reference lines. Defaults to "darkorange".
        color2 (str, optional): Another color of the reference lines. Defaults to "firebrick".
    """

    ax = plt.subplots(figsize=figsize)[1]
    ax.scatter(leverage, studentized_resid, alpha=alpha, label="Observations")
    ax.axhline(3, color=color1, linestyle="--", label="Outlier boundaries")
    ax.axhline(-3, color=color1, linestyle="--")
    ax.axvline(leverage_thresh, color=color2, linestyle=":", label="Leverage threshold")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized Residual")
    ax.set_title("Leverage vs Studentized Residuals", pad=20)
    ax.legend(
        loc="upper left",
        title="Metrics",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1, 1),
        alignment="left",
    )
    plt.show()


def compare_train_test_metrics(
    X_test_transf_scaled_5: pd.DataFrame,
    y_test: pd.DataFrame,
    y_test_pred: np.ndarray,
    train_metrics: list[float],
) -> pd.DataFrame:
    """
    Makes a dataframe of metrics for test and train data.

    Args:
        X_test_transf_scaled_5 (pd.DataFrame): Test data prepared for predictions (transformed and scaled).
        y_test (pd.DateFrame): Response values of the test data.
        y_test_pred (np.ndarray): Predictions of response on the test data.
        train_metrics (list[float]): Metrics on the model performance.

    Returns:
        pd.DataFrame: The dataframe of model performance metrics on different subsets.
    """
    R_sq_test = r2_score(y_test, y_test_pred)
    n = X_test_transf_scaled_5.shape[0]
    p = X_test_transf_scaled_5.shape[1]
    R_sq_adj_test = 1 - (1 - R_sq_test) * (n - 1) / (n - p - 1)

    MAE_test = mean_absolute_error(y_test, y_test_pred)
    RSE_test = np.sqrt(mean_squared_error(y_test, y_test_pred) * n / (n - p - 1))

    comparison_metrics = pd.DataFrame(
        {"Train data": train_metrics, "Test data": [R_sq_adj_test, MAE_test, RSE_test]},
        index=["R squared adjusted", "MAE", "RSE"],
    )

    comparison_metrics = comparison_metrics.reset_index().rename(
        columns={"index": "Metrics"}
    )
    comparison_metrics["Difference in metric"] = (
        comparison_metrics["Train data"] - comparison_metrics["Test data"]
    )
    return comparison_metrics


def plot_models_comparison_metrics(
    comparison_metrics: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    key: str,
    decimals: str = ".3f",
) -> None:
    """
    Creates a horizontal barplot to visualize metrics.

    Args:
        comparison_metrics (pd.DataFrame): The dataframe of the metrics.
        figsize (tuple[float | int, float | int]): The width and height of the figure.
        key (str): The string used in the title.
        decimals (str, optional): The format of the annotations on the bars. Defaults to ".3f".
    """
    comparison_metrics_long = comparison_metrics[
        ["Metrics", "Train data", "Test data"]
    ].melt(id_vars="Metrics", var_name="Dataset", value_name="Value")

    ax = plt.subplots(figsize=figsize)[1]
    sns.barplot(comparison_metrics_long, y="Metrics", x="Value", hue="Dataset", ax=ax)
    ax.legend(
        loc="upper left",
        title="Metrics",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1.05, 1),
        alignment="left",
    )
    ax.grid(False)
    for bar in ax.patches:
        width = bar.get_width()
        if width == 0:
            continue
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 0.02, y, s=f"{width:{decimals}}", ha="left", va="center", fontsize=9
        )
    ax.set_title(f"{key} evaluation metrics on train vs test data", pad=20)
    plt.show()


def make_ordinal_log_regr_model(
    y_train: pd.DataFrame, X_ord: pd.DataFrame
) -> OrderedResultsWrapper:
    """
    Makes an Ordinal Logistic Regression model.

    Args:
        y_train (pd.DataFrame): Response variable training subset.
        X_ord (pd.DataFrame): Predictors' training subset.

    Returns:
        OrderedResultsWrapper: The results wrapper for the Ordinal model.
    """
    model = OrderedModel(endog=y_train, exog=X_ord, distr="logit")
    results = model.fit(method="bfgs")
    print(f"McFadden's pseudo R-squared {results.prsquared:.4f}")
    print(results.summary())
    return results


def make_ordinal_predictions(
    results: OrderedResultsWrapper, X_test_transf_scaled: pd.DataFrame
) -> pd.Series:
    """
    Calculates predictions for the passed dataset using Logistic model.

    Args:
        results (OrderedResultsWrapper): Ordinal model wrapper.
        X_test_transf_scaled (pd.DataFrame): The test subset prepared for predicting.

    Returns:
        pd.Series: Predicted classes for the test data.
    """
    predicted_probs = results.model.predict(results.params, exog=X_test_transf_scaled)
    class_labels = np.unique(results.model.endog) + 3
    predicted_probs_df = pd.DataFrame(predicted_probs, columns=class_labels)
    y_pred = predicted_probs_df.idxmax(axis=1).astype(int)
    return y_pred


def compare_log_models(
    results: list[OrderedResultsWrapper],
    figsize: tuple[float | int, float | int],
    key: str,
) -> None:
    """
    Makes a plot of the metrics for different Ordinal models.

    Args:
        results (list[OrderedResultsWrapper]): The list of Ordinal models' wrappers.
        figsize (tuple[float | int, float | int]): The width and height of the figure.
        key (str): The string used in a title.
    """
    AIC_list = [result.aic for result in results]
    BIC_list = [result.bic for result in results]
    metrics_df = pd.DataFrame(
        data={"AIC": AIC_list, "BIC": BIC_list},
        index=[f"Mod.{i}" for i in range(1, len(results) + 1)],
    )
    metrics_df = metrics_df.reset_index().rename(columns={"index": "Models"})

    columns = metrics_df.columns[1:]
    ax = plt.subplots(figsize=figsize)[1]
    for col in columns:
        sns.pointplot(data=metrics_df, y=col, x="Models", ax=ax, label=col)
        ax.set_title(f"AIC and BIC metrics for different {key} models", pad=20)
        ax.set_ylabel("Metric")
    ax.legend(
        loc="upper left",
        title="Metrics",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1.1, 1),
        alignment="left",
    )
    plt.show()


def plot_multi_lines(
    data: pd.DataFrame,
    figsize: tuple[float | int, float | int],
    ylabel: str,
    xlabel: str,
    title: str,
    legend_title: str,
    x: str = None,
    y: str = None,
    hue: str = None,
    hline: bool = True,
    palette: str = "colorblind",
    markersize: int = 7,
) -> None:
    """
    Creates a lineplot for the dataframe.

    Args:
        data (pd.DataFrame): The dataframe of the data to be plotted.
        figsize (tuple[float | int, float | int]): The width and height of the figure.
        ylabel (str): The y axis label.
        xlabel (str): The x axis label.
        title (str): The title of the plot.
        legend_title (str): The title of legends.
        x (str, optional): The feature for the x axis. Defaults to None.
        y (str, optional): The feature for the y axis. Defaults to None.
        hue (str, optional): The feature used as hue. Defaults to None.
        hline (bool, optional): Possibility to add horizontal reference line. Defaults to True.
        palette (str, optional): The color palette of the plot. Defaults to "colorblind".
        markersize (int, optional): The size of the marker. Defaults to 7.
    """

    ax = plt.subplots(figsize=figsize)[1]
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        marker="o",
        markersize=markersize,
        ax=ax,
        linewidth=2,
        dashes=False,
        palette=palette,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=20)

    if hline == True:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.legend(
        loc="upper left",
        title=legend_title,
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1, 1),
        alignment="left",
    )
    plt.show()


def make_odds_interpretation_df(
    stds: list[float],
    coefs: pd.Series,
    transformations: list[str],
    none_transf_stats: list[tuple[float, float]],
) -> pd.DataFrame:
    """
    Creates summary summary table for interpretation of logistic regression coefficients.

    Args:
        stds (list[float]): List of predictors standard deviations.
        coefs (pd.Series): Coefficients from the logistic regression model.
        transformations (list[str]): List of transformations made on predictors.
        none_transf_stats (list[tuple[float, float]]): List of tuples for predictor's
        standard deviation and mean metrics.

    Returns:
        pd.DataFrame: The dataframe of the metrics.
    """
    changes_in_odds = []
    odds_ratios = []
    for std, coef, transf in zip(stds, coefs, transformations):
        odds_ratio, perc_change = rescale_to_odds(
            coef, std, 0.1, transf, none_transf_stats
        )
        changes_in_odds.append(perc_change)
        odds_ratios.append(odds_ratio)
    summary = (
        pd.DataFrame(
            {
                "Coefficient": coefs,
                "Transformation": transformations,
                "Odds ratio": odds_ratios,
                "Effect on odds of 10% increase in the predictor": changes_in_odds,
            }
        )
        .reset_index()
        .rename(columns={"index": "Predictors"})
    )
    return summary


def make_effects_barplot(
    df: pd.DataFrame,
    figsize: tuple[float, float],
    xlabel: str,
    title: str,
    offset: float,
    xlim: tuple[float, float] = None,
    decimals: str = ".2f",
) -> None:
    """
    Makes a horizontal barplot for predictors' effect on target variable.

    Args:
        df (pd.DataFrame): Summary metrics dataframe.
        figsize (tuple[float, float]): Width and height of the figure.
        xlabel (str): The width and height of the figure.
        title (str): The title of the figure.
        offset (float): The offset of the values from the bars on the plot.
        xlim (tuple[float, float], optional): X axis limits. Defaults to None.
        decimals (str, optional): A format of values on the bars. Defaults to '.2f'.
    """
    ax = plt.subplots(figsize=figsize)[1]
    sns.barplot(
        data=df,
        x=xlabel,
        y="Predictors",
    )
    ax.set_title(title, pad=20)
    ax.set_xlim(xlim)
    ax.grid(False)
    for bar in ax.patches:
        width = bar.get_width()
        if width == 0:
            continue

        y = bar.get_y() + bar.get_height() / 2
        offset = offset

        if width > 0:
            x = width + offset
            ha = "left"
        else:
            x = width - offset
            ha = "right"

        ax.text(x, y, s=f"{width:{decimals}}", ha=ha, va="center", fontsize=8.5)
    plt.show()
