"""
Created by Bing Liu
Plot the powers
"""
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox

#Plot the powers
def Display(freqs, powers, peaks_list, title="", names=[], colors=[]):
    
    l_ajust = 0.0
    h_ajust = -0.8

    x = freqs
    y = powers
    x_len = len(x)

    left, width = 0.03, 0.95
    bottom, height = 0.08, 0.08
    slider_box = [left, bottom, width, height]
    price_box = [left, bottom + height + 0.06, width, height + 0.68]

    fig = plt.figure("", figsize=(10.2,5.2))
    ax1 = plt.axes(price_box)
    ax2 = plt.axes(slider_box)
    fig.suptitle(title, x=0.8, y =0.95, fontsize=12)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.075)

    line1, = ax1.plot(x, y,'-k', color='gray', linewidth=0.5)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(min(y) + min(y)*l_ajust, max(y) + max(y)*h_ajust)

    l2x = np.arange(0, len(powers), 1)
    line2, = ax2.plot(l2x, y,'-', color = "black", linewidth = 0.3)
    ax2.set_xlim(l2x[0], l2x[-1])
    ax2.set_ylim(min(y), max(y))
    ax2.axes.get_xaxis().set_visible(False)

    if names != []:
        increase = 1.2
        for peaks, name, colored in zip(peaks_list, names, colors):
            i = 0
            for peak in peaks:
                if peak == 1:
                    ax1.annotate(" ", xy=(freqs[i],float(powers[i] + increase )), color=colored,
                                            arrowprops=dict(facecolor=colored, width = 6, headwidth=6, headlength=6))
                i += 1
            increase -= 0.4
    else:
        i = 0
        for peak in peaks_list:
            if peak == 1:
                ax1.annotate(" ", xy=(freqs[i],float(powers[i] )), color='red',
                                        arrowprops=dict(facecolor='red', width = 6, headwidth=6, headlength=6))
            i += 1


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(l2x, (int(xmin), int(xmax)+1))
        indmax = min(len(l2x) - 1, indmax) + 1

        thisx = x[indmin:indmax]
        this_date = freqs[indmin:indmax]
        thisy = y[indmin:indmax]
        line1.set_data(thisx, thisy)
        ax1.set_xlim(thisx[0], thisx[-1])
        ax1_this_y_min = min(thisy) - min(thisy)*l_ajust
        ax1_this_y_max = max(thisy) + max(thisy)*h_ajust
        ax1.set_ylim(ax1_this_y_min, ax1_this_y_max)
        fig.canvas.draw()

    hover_annotations = []
    def remove_annotations_hover():
        for child in ax1.get_children():
            if isinstance(child, matplotlib.text.Annotation):
                for annotation in hover_annotations:
                    if child is annotation:
                        child.remove()
                        hover_annotations.remove(annotation)
      
        
    def hover(event):
        if len(ax1.lines) > 1 :
            ax1.lines[-1].remove()
            ax1.lines[-1].remove()
        if event.inaxes == ax1:
            cont, ind = line1.contains(event)
            if cont:
                ind_value = ind["ind"][min(len(ind), int((len(ind)+1)/2))-1]
                xdata, ydata = line1.get_data()
                len_y = len(ydata)
                len_x = len(xdata)

                y_values = set_fixed_values(ydata[ind_value], len_y)
                x_values = set_fixed_values(xdata[ind_value], len_x)

                remove_annotations_hover()

                if len(ax1.lines) > 1 :
                    ax1.lines[-1].remove()
                    ax1.lines[-1].remove()
                ax1.axhline(ydata[ind_value],linewidth=0.5,linestyle='-',color='gray')
                ax1.axvline(xdata[ind_value],linewidth=0.5,linestyle='-',color='gray')

                for i,j in zip(xdata[ind_value:ind_value+1],ydata[ind_value:ind_value+1]):
                    display = "{:.1f}".format(j) + "db, " + str(i/1e6) + "m, "
                    annotation = ax1.annotate(display, xy=(i,j))
                hover_annotations.append(annotation)

                fig.canvas.draw()
        else:
            remove_annotations_hover()
            fig.canvas.draw()

    def key_press(event):
        if event.key == 'escape':
            span.stay_rect.set_visible(False)
            onselect(0,len(l2x))

    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax2, onselect, 'horizontal', minspan = 2, useblit=True, span_stays=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))
    fig.canvas.mpl_connect("key_press_event", key_press)
    fig.canvas.mpl_connect("motion_notify_event", hover)


    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    plt.show()
    return

#Set X Y data
def set_fixed_values(value, len):
    values = []
    for i in range (0, len):
        values.append(value)
    return values
