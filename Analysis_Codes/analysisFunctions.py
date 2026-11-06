


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
