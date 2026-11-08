


######################
### VISUALISATIONS ###
######################


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import numpy as np





import seaborn as sb

def eyeRawKDE(RawDF,titl="Raw Gaze Distributions"):
    """
    Create kernel density plots for raw eye-tracking data.

    Parameters:
    RawDF (DataFrame): Raw eye-tracking data.
    """
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(16, 10)
    fig.suptitle(titl,fontsize=20)

    # KDE plot for X coordinates
    sb.kdeplot(RawDF['LeftX'], ax=axs[0])
    sb.kdeplot(RawDF['RightX'], ax=axs[0])
    axs[0].set_title('X')
    axs[0].legend(loc='upper left', labels=['Left', 'Right'])

    # KDE plot for Y coordinates
    sb.kdeplot(RawDF['LeftY'], ax=axs[1])
    sb.kdeplot(RawDF['RightY'], ax=axs[1])
    axs[1].set_title('Y')
    axs[1].legend(loc='upper left', labels=['Left', 'Right'])

    # KDE plot for pupil diameter
    sb.kdeplot(RawDF['LeftPupil'], ax=axs[2])
    sb.kdeplot(RawDF['RightPupil'], ax=axs[2])
    axs[2].set_title('Pupil')
    axs[2].legend(loc='upper left', labels=['Left', 'Right'])

    return fig
def plotRawGaze(RawDF, gazeCoords, titl="Raw Gaze Data",show_outside=True, margin=200):

    fig = plt.figure(figsize=(20, 20))
    plt.title(titl, fontsize=20)

    plt.xlabel("Width Coords (pixels)")
    plt.ylabel("Height Coords (pixels)")

   

    plt.plot(RawDF['LeftX'], RawDF['LeftY'], label='Left',zorder=1)
    plt.plot(RawDF['RightX'], RawDF['RightY'], 'r--', label='Right',zorder=1)
    plt.legend(loc='upper center')

    ax = plt.gca()

    # Prostokąt ekranu
    rect = plt.Rectangle(
        (0, 0),
        gazeCoords[2],
        gazeCoords[3],
        linewidth=2,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(rect)

    # OKRĄG 5° visual angle (290 px średnica → 145 px promień)
    radius_px = deg2pix(3)
    circle = plt.Circle(
        (gazeCoords[2] / 2, gazeCoords[3] / 2),
        radius_px,
        fill=False,
        linewidth=2,
        zorder=11
    )
    ax.add_patch(circle)

    ax.set_aspect('equal')
    
    if show_outside:
        # zakres osi = dane + margines
        x_min = min(RawDF[['LeftX','RightX']].min()) - margin
        x_max = max(RawDF[['LeftX','RightX']].max()) + margin
        y_min = min(RawDF[['LeftY','RightY']].min()) - margin
        y_max = max(RawDF[['LeftY','RightY']].max()) + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # tylko ekran
        ax.set_xlim(0, gazeCoords[2])
        ax.set_ylim(0, gazeCoords[3])

    return fig


import numpy as np
def plotPupilTimecourse(ScreenData, title, blinks, saccades, fixations, chooseViz='011',secChan=0):
    """
    Plot pupil diameter over time with optional visualization of fixations, blinks, and saccades.

    Parameters:
    ScreenData (DataFrame): Screen data containing time points and pupil diameter.
    title (str): Title of the plot.
    blinks (DataFrame): DataFrame containing blink events.
    saccades (DataFrame): DataFrame containing saccade events.
    fixations (DataFrame): DataFrame containing fixation events.
    chooseViz (str): String of bools indicating which visualizations to include (fixations, saccades, ).
    """
    import numpy as np

    fig = plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=20)
    
    # Plot pupil diameter data
    plt.plot(ScreenData['TimePoint'], ScreenData['LeftPupil'], label="Left Pupil", color='blue')
    if secChan:
        plt.plot(ScreenData['TimePoint'], ScreenData['RightPupil'], label="Right Pupil", color='orange')

    # Add fixation rectangles
    if bool(int(chooseViz[0])):
        for _, fixation in fixations.iterrows():
            plt.axvspan(fixation['Beg'], fixation['End'], color='lightblue', alpha=0.2, label="Fixation" if _ == 0 else "")
    
    # Add saccade rectangles
    if bool(int(chooseViz[1])):
        for _, saccade in saccades.iterrows():
            plt.axvspan(saccade['Beg'], saccade['End'], color='green', alpha=0.3, label="Saccade" if _ == 0 else "")
    
    # Add blink rectangles
    if bool(int(chooseViz[2])):
        for _, blink in blinks.iterrows():
            plt.axvspan(blink['Beg'], blink['End'], color='red', alpha=0.3, label="Blink" if _ == 0 else "")
    
    # Calculate session duration in seconds
    sesDur = (ScreenData['TimePoint'].iloc[-1] - ScreenData['TimePoint'].iloc[0]) / 1000

    # Set x-axis ticks and labels
    original_ticks = np.linspace(ScreenData['TimePoint'].iloc[0], ScreenData['TimePoint'].iloc[-1], num=10)
    labels_in_seconds = (original_ticks - ScreenData['TimePoint'].iloc[0]) / 1000
    plt.xticks(
        ticks=original_ticks,
        labels=np.round(labels_in_seconds, 1)
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Pupil Diameter (a.u.)")
    plt.legend()

    return fig



def interactive_pupil_plot(ScreenData, blinks, saccades, fixations, gazeCoords, window_size=2000, chooseViz='011'):
    """
    Interactive visualization of pupil diameter and gaze position over time.
    Allows sliding window inspection with optional overlays for fixations, saccades, and blinks.
    """
    # Downsample for performance
    
    downsample_factor = max(1, len(ScreenData) // 5000)
    ScreenData = ScreenData.iloc[::downsample_factor]
        
    blinks = blinks.copy()
    saccades = saccades.copy()
    fixations = fixations.copy()
    ScreenData = ScreenData.copy()


    blinks['Beg'] -= ScreenData['TimePoint'].iloc[0]
    blinks['End'] -= ScreenData['TimePoint'].iloc[0]

    saccades['Beg'] -= ScreenData['TimePoint'].iloc[0]
    saccades['End'] -= ScreenData['TimePoint'].iloc[0]

    fixations['Beg'] -= ScreenData['TimePoint'].iloc[0]
    fixations['End'] -= ScreenData['TimePoint'].iloc[0]

    ScreenData['TimePoint'] = ScreenData['TimePoint'] - ScreenData['TimePoint'].iloc[0]

    # Prepare figure layout
    plt.subplots_adjust(bottom=0.15)
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # Main pupil plot
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_title("Pupil Diameter Time Course", fontsize=16)
    ax_main.set_xlabel("Time (ms)")
    ax_main.set_ylabel("Pupil Diameter (a.u.)")
    ax_main.plot(ScreenData['TimePoint'], ScreenData['LeftPupil'], label="Left Pupil", color='blue')
    ax_main.plot(ScreenData['TimePoint'],ScreenData['RightPupil'],label="Right Pupil",color='orange')

    # Optional overlays
    if bool(int(chooseViz[0])):
        for i, row in fixations.iterrows():
            ax_main.axvspan(row['Beg'], row['End'], color='lightblue', alpha=0.2, label="Fixation" if i == 0 else "")
    if bool(int(chooseViz[1])):
        for i, row in saccades.iterrows():
            ax_main.axvspan(row['Beg'], row['End'], color='green', alpha=0.3, label="Saccade" if i == 0 else "")
    if bool(int(chooseViz[2])):
        for i, row in blinks.iterrows():
            ax_main.axvspan(row['Beg'], row['End'], color='red', alpha=0.3, label="Blink" if i == 0 else "")

    # Add moving window rectangle
    start_time = ScreenData['TimePoint'].iloc[0]
    window_rect = Rectangle((start_time, ax_main.get_ylim()[0]), window_size, ax_main.get_ylim()[1] - ax_main.get_ylim()[0],
                            color='orange', alpha=0.3, label="Window")
    ax_main.add_patch(window_rect)

    # Gaze scatter plot
    ax_gaze = fig.add_subplot(gs[0, 1])
    ax_gaze.set_title("Raw Gaze Points in Selected Window", fontsize=16)
    ax_gaze.set_xlabel("X Coordinate (pixels)")
    ax_gaze.set_ylabel("Y Coordinate (pixels)")
    ax_gaze.set_xlim(-gazeCoords[2], 2 * gazeCoords[2])
    ax_gaze.set_ylim(-gazeCoords[3], 2 * gazeCoords[3])
    ax_gaze.set_aspect(gazeCoords[2] / gazeCoords[3])

    screen_rect = plt.Rectangle((0, 0), gazeCoords[2], gazeCoords[3], linewidth=2, edgecolor='black', facecolor='none')
    ax_gaze.add_patch(screen_rect)

    gaze_scatter = ax_gaze.scatter([], [], color='blue', label="Left Eye Gaze")
    gaze_scatter_right = ax_gaze.scatter([], [], color='orange', label="Right Eye Gaze")

    # Slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, "Time (ms)", start_time, ScreenData['TimePoint'].iloc[-1] - window_size, valinit=start_time)

    def update(val):
        start = slider.val
        end = start + window_size

        # Update rectangle
        window_rect.set_x(start)
        window_rect.set_width(window_size)

        # Update scatter plot
        window_data = ScreenData[(ScreenData['TimePoint'] >= start) & (ScreenData['TimePoint'] <= end)]
        gaze_scatter.set_offsets(np.c_[window_data['LeftX'], window_data['LeftY']])
        gaze_scatter_right.set_offsets(np.c_[window_data['RightX'], window_data['RightY']])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Show legends and plot
    ax_main.legend()
    ax_gaze.legend()
    plt.show()








###############################
### Viusalisation functions ###
###############################





###### ---- (1) Testng MW_estimate vs Tracking/Untracking ---- ######
import statsmodels.api  as sm

def plotTrialTrack(m, name, grand_df, ax, titl, paramNum=[0, 1, 2], anlysisName='Tracking',
                   firstContrast='TRACKED', lastContrast='UNTRACKED', covariates=None):
    """
    Plots model-predicted lines over trials for two groups + observed trial means.
    covariates: dict {parameter name: value} for the other predictors in the model,
                e.g. {'LemmaLen': df['LemmaLen'].mean(), 'recallfirst[T.True]': df['recallfirst'].mean()}
    """
    from scipy.stats import t
    covariates = covariates or {}
    p = m.params

    # --- predicted lines (other predictors held at the given values) ---
    trialN   = np.arange(0, grand_df['trialNum'].max() + 1)
    offset   = sum(p[k] * v for k, v in covariates.items())
    line_ref = p.iloc[paramNum[0]] + p.iloc[paramNum[2]] * trialN + offset   # firstContrast
    line_grp = line_ref + p.iloc[paramNum[1]]                                 # lastContrast

    # --- 95% CI of the predicted mean ---
    pnames = list(m.fe_params.index) if hasattr(m, 'fe_params') else list(p.index)
    col    = {n: i for i, n in enumerate(pnames)}
    Vbeta  = m.cov_params().loc[pnames, pnames].values
    dfr    = getattr(m, 'df_resid', np.inf)
    tval   = t.ppf(0.975, df=dfr) if np.isfinite(dfr) else 1.96
    grp_param = 'Tracking[T.UNTRACKED]' if anlysisName == 'Tracking' else f'C({anlysisName})[T.{lastContrast}]'

    def build_X(g):
        X = np.zeros((len(trialN), len(pnames)))
        X[:, col['Intercept']] = 1.0
        if grp_param in col:  X[:, col[grp_param]] = g
        if 'trialNum' in col: X[:, col['trialNum']] = trialN
        for k, v in covariates.items():
            if k in col: X[:, col[k]] = v
        return X

    X0, X1 = build_X(0.0), build_X(1.0)
    se_ref = np.sqrt(np.einsum('ij,jk,ik->i', X0, Vbeta, X0))
    se_grp = np.sqrt(np.einsum('ij,jk,ik->i', X1, Vbeta, X1))

    # --- plot ---
    means = grand_df.groupby([anlysisName, 'trialNum'])[name].mean()
    ax.set_title(titl)
    ax.plot(trialN, line_ref, color='blue',   label=f'Model: {anlysisName} = {firstContrast}')
    ax.plot(trialN, line_grp, color='orange', label=f'Model: {anlysisName} = {lastContrast}')
    ax.fill_between(trialN, line_ref - tval*se_ref, line_ref + tval*se_ref, color='mediumslateblue', alpha=0.25, label='95% CI')
    ax.fill_between(trialN, line_grp - tval*se_grp, line_grp + tval*se_grp, color='moccasin',        alpha=0.25, label='95% CI')
    ax.scatter(means.loc[firstContrast].index, means.loc[firstContrast], color='blue',   label=f'Trial mean ({firstContrast})')
    ax.scatter(means.loc[lastContrast].index,  means.loc[lastContrast],  color='orange', label=f'Trial mean ({lastContrast})')
    ax.set_xlabel("Trial Number"); ax.set_ylabel(name)
    ax.legend()
  

####### (3) ---- MW vs Gaze ----- #######
from scipy.stats import t

def plot_continous_results(m,currentDf,x_name,response,ax,titl ="",group_param=None):

    # --- 1) grid dla osi X ---
    x_grid = np.linspace(currentDf[x_name].min(), currentDf[x_name].max(), 200)

    # --- 2) fixed effects i ich kowariancja (FE only) ---
    fe_names = list(m.fe_params.index)
    V_fe = m.cov_params().loc[fe_names, fe_names].values
    col = {name: i for i, name in enumerate(fe_names)}

    # bezpieczeństwo: sprawdź, czy x_name jest w fe_names
    if x_name not in fe_names:
        raise KeyError(f"{x_name} not in fixed-effect names: {fe_names}")

    # --- 3) budujemy macierz modelu Xg_base (intercept + x) ---
    Xg_base = np.zeros((len(x_grid), len(fe_names)))
    Xg_base[:, col.get('Intercept', 0)] = 1.0
    Xg_base[:, col[x_name]] = x_grid

    # --- 4) ustaw pozostałe predyktory (domyślnie na ich średnie) ---
    import re
    for pname in fe_names:
        if pname in ('Intercept', x_name):
            continue

        # inny predictor: numeric -> mean if in DF, else 0
        if pname in currentDf.columns:
            Xg_base[:, col[pname]] = currentDf[pname].mean()
        else:
            Xg_base[:, col[pname]] = 0.0

    # --- 5) TEMP FIX: wyciszamy wariancję Interceptu w macierzy użytej do CI (usuwa gigantyczne pasmo) ---
    V_fe_fixed = V_fe.copy()


    # --- 6) funkcja predykcja + SE ---
    def predict_and_se(Xg):
        beta_fe = m.fe_params.values
        y_fit = Xg @ beta_fe
        var_fit = np.einsum('ij,jk,ik->i', Xg, V_fe_fixed, Xg)
        se_fit = np.sqrt(np.clip(var_fit, 0, np.inf))
        return y_fit, se_fit

    # --- 7) t-value dla 95% CI ---
    tval = t.ppf(0.975, df=m.df_resid)
    if not np.isfinite(tval):
        tval = 1.96


    # --- 8) Rysowanie ---
    ax.set_title(titl)
    ax.scatter(currentDf[x_name], currentDf[response], color='orange', alpha=0.6, s=25, label='Data')

    if group_param in col and group_param is not None:
        # rysujemy dwie linie: group=0 i group=1
        for rval, color, lab in [(0, 'blue', f'{group_param}=0'), (1, 'orange', f'{group_param}=1')]:
            Xg = Xg_base.copy()
            Xg[:, col[group_param]] = rval
            y_fit, se_fit = predict_and_se(Xg)
            ax.plot(x_grid, y_fit, color=color, linestyle='--', linewidth=2, label=f'Fit ({lab})')
            ax.fill_between(x_grid, y_fit - tval*se_fit, y_fit + tval*se_fit, color=color, alpha=0.18)
    else:
        # pojedyncza linia (kategorii trzymane na średnich)
        y_fit, se_fit = predict_and_se(Xg_base)
        ax.plot(x_grid, y_fit, 'b--', linewidth=2, label='Model fit (FE)')
        ax.fill_between(x_grid, y_fit - tval*se_fit, y_fit + tval*se_fit, color='blue', alpha=0.18, label='95% CI')

    ax.set_xlabel(x_name)
    ax.set_ylabel(response)
    ax.legend(loc='best')

    # sensowne limity Y (oparte na danych, żeby nie dopuścić do ekstremalnych rozciągnięć)
    y_min, y_max = currentDf[response].min(), currentDf[response].max()
    pad = max(0.2, 0.05 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)




def linearModelChecks(m,grand_df,formula =  "MW_Estimate ~ Tracking + trialNum"):
    # trial-level diagnostics
    resid   = m.resid                   # raw (conditional) residuals
    fitted  = m.fittedvalues            # conditional fitted values
    std_res = resid / np.sqrt(m.scale)  # ≈ studentised residuals
    fig,axs = plt.subplots(2,1,figsize=(14, 8))
    plt.suptitle("Checks for a Linear Model")

    # (1) Homoscedascisity
    axs[0].scatter(fitted, std_res, alpha=0.6, s=15,color='blue')
    axs[0].axhline(0, linestyle='--',color='orange')
    axs[0].set_xlabel("Fitted value")
    axs[0].set_ylabel("Studentised residual")
    axs[0].set_title("Residuals vs fitted")

    # (2) 
    sm.qqplot(std_res, line="45", fit=True, ax=axs[1])
    axs[1].set_title("Normal Q–Q")

    # (3) Colinearity:
    from patsy import dmatrices

    formula_fixed = formula  # your fixed part
    y, X = dmatrices(formula_fixed, data=grand_df, return_type="dataframe")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    print(vif)

     # Drop the intercept from display (its VIF is usually not meaningful)
    vif_display = vif.drop('Intercept', errors='ignore')
    
    # Convert to multiline string
    vif_text = "\n".join([f"{k}: {v:.2f}" for k, v in vif_display.items()])
    vif_text = f"VIFs\n{vif_text}"

    axs[1].text(
        0.8, 0.5, vif_text,
        transform=axs[1].transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
    )







#####################
### UTILITIES ###
#####################


def downsamplePupil(ScreenData,divisor=5,logfile=None):
    """
    Downsample Pupil Data to 100 Hz
    :param ScreenData:
    : divisor - dividing current Rate to a given count via Modullo 
    : logfile (Optional) - if you want to log things 
    :return:
    """
    # Downsample to freq (default 100) Hz
    ScreenData = ScreenData[ScreenData['TimePoint'] % divisor == 0]
    if logfile is not None:
        print(f"    Downsampling to rate {500/divisor} Hz")
        logfile.write(f"        (1) Downsampling to rate {500/divisor} Hz\n")
    return ScreenData

def deg2pix(x,gazeCoords=[0,0,1919,1079]): 
    """
    Recalculates Degrees of VIsual Angle to Pixels based on screen Coords and 

    :param x: Description
    :param gazeCoords: Description
    """

    return 92*np.tan(np.deg2rad(x)) * np.mean([gazeCoords[2]/53,gazeCoords[3]/30])


def pix2deg(x,gazeCoords=[0,0,1919,1079]):
    """
    Recalculates Pixels to Degrees of Visual Angle based on screen Coords
    """
    return np.rad2deg(np.arctan(x/(92 * np.mean([gazeCoords[2]/53,gazeCoords[3]/30]))))
